"""Shared counting/localization protocol. Coordinates are (y, x), in pixels."""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _points(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(value).all():
        raise ValueError('Point coordinates must be finite')
    return value


def match_counts(prediction, target, radius):
    """Maximize valid one-to-one matches, then minimize their total distance."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be positive and finite')
    prediction, target = _points(prediction), _points(target)
    if not len(prediction) or not len(target):
        return 0, len(prediction), len(target)
    distance = cdist(prediction, target)
    # A forbidden edge must cost more than ALL valid edges combined.
    # Minimizing raw distance then filtering can lose valid matches.
    penalty = min(distance.shape) + 1.0
    cost = np.where(distance <= radius, distance / radius, penalty)
    rows, cols = linear_sum_assignment(cost)
    tp = int(np.count_nonzero(distance[rows, cols] <= radius))
    return tp, len(prediction) - tp, len(target) - tp


class PointMetrics:
    def __init__(self, radii=(4, 8, 16)):
        self.radii = tuple(sorted(set(float(r) for r in radii)))
        if not self.radii or any(not math.isfinite(r) or r <= 0 for r in self.radii):
            raise ValueError('At least one positive finite radius is required')
        self.rows = []

    def update(self, image_id, prediction, target):
        prediction, target = _points(prediction), _points(target)
        error = len(prediction) - len(target)
        row = dict(image_id=str(image_id), pred_count=len(prediction),
                   gt_count=len(target), error=error, abs_error=abs(error))
        for radius in self.radii:
            tp, fp, fn = match_counts(prediction, target, radius)
            key = format(radius, 'g')
            row.update({f'tp_{key}px': tp, f'fp_{key}px': fp, f'fn_{key}px': fn})
        self.rows.append(row)

    def summary(self):
        if not self.rows:
            raise ValueError('Cannot evaluate an empty dataset')
        error = np.asarray([r['error'] for r in self.rows], dtype=np.float64)
        result = dict(samples=len(self.rows), mae=float(np.abs(error).mean()),
                      rmse=float(np.sqrt(np.square(error).mean())))
        for radius in self.radii:
            key = format(radius, 'g')
            tp, fp, fn = (sum(r[f'{name}_{key}px'] for r in self.rows)
                          for name in ('tp', 'fp', 'fn'))
            # Dataset-wide micro aggregation; undefined ratios are 0.
            result.update({f'precision_{key}px': tp / (tp + fp) if tp + fp else 0.0,
                           f'recall_{key}px': tp / (tp + fn) if tp + fn else 0.0,
                           f'f1_{key}px': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                           f'tp_{key}px': tp, f'fp_{key}px': fp, f'fn_{key}px': fn})
        return result

    def synchronize(self, expected_samples=None):
        """Gather per-image records, deduplicating DistributedSampler padding."""
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, self.rows)
            unique = {}
            for rows in gathered:
                for row in rows:
                    unique.setdefault(row['image_id'], row)
            self.rows = list(unique.values())
        if expected_samples is not None and len(self.rows) != expected_samples:
            raise ValueError(f'Expected {expected_samples} images, evaluated {len(self.rows)}; check IDs/sampler')

    def export(self, directory, summary=None, metadata=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = dict(metrics=summary or self.summary(), protocol={
            'coordinate_order': 'yx', 'distance': 'Euclidean', 'radius_unit': 'evaluated_image_pixels',
            'radii': self.radii, 'matching': 'maximum cardinality one-to-one, then minimum distance',
            'aggregation': 'dataset micro TP/FP/FN', 'undefined_precision_recall_f1': 0,
        }, metadata=metadata or {})
        (directory / 'metrics.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        with (directory / 'per_image.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return payload


def append_history(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline='') as stream:
            keys = next(csv.reader(stream))
        if keys != list(result):
            raise ValueError('Metric history schema changed; use a new experiment directory')
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        if not exists:
            writer.writeheader()
        writer.writerow(result)
