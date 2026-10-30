import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from counting_metrics import PointMetrics
from progress import progress_bar


def select_points(outputs, index, size, threshold=.5):
    height, width = size
    points = outputs['pred_points'][index].float()
    scores = outputs['pred_logits'][index].float().softmax(-1)[:, 1]
    valid = ((scores > threshold) & torch.isfinite(points).all(-1) &
             (points[:, 0] >= 0) & (points[:, 0] < width) &
             (points[:, 1] >= 0) & (points[:, 1] < height))
    # No point-radius NMS: closely spaced true leaf tips must not be merged.
    return points[valid], scores[valid]


def move_targets(targets, device):
    return [dict(target, points=target['points'].to(device, non_blocking=True)) for target in targets]


def train_epoch(model, criterion, loader, optimizer, scaler, epoch, args, device):
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    totals = defaultdict(float)
    samples = 0
    with progress_bar(loader, desc=f'Train {epoch+1}/{args.epochs}', unit='batch') as bar:
        for index, (images, targets) in enumerate(bar):
            images = images.to(device, non_blocking=True)
            targets = move_targets(targets, device)
            group_start = index // args.accum_steps * args.accum_steps
            group_size = min(args.accum_steps, len(loader) - group_start)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                output = model(images, targets)
            # Distance, matching and losses run in FP32, also under AMP training.
            losses = criterion(output, targets)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError(f'Non-finite loss in epoch {epoch+1}, batch {index+1}')
            scaler.scale(losses['loss'] / group_size).backward()
            if (index + 1) % args.accum_steps == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_size = len(targets)
            samples += batch_size
            for key, value in losses.items():
                totals[key] += value.detach().item() * batch_size
            bar.set_postfix(loss=f'{totals["loss"]/samples:.4f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.1e}', refresh=False)
    if not samples:
        raise ValueError('Empty training loader')
    return {key: value / samples for key, value in totals.items()}


@torch.inference_mode()
def evaluate(model, loader, device, output_dir, threshold=.5, radii=(4,8,16), epoch=None):
    model.eval()
    meter = PointMetrics(radii)
    start = time.perf_counter()
    with progress_bar(loader, desc='Validation', unit='image') as bar:
        for images, targets in bar:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            for index, target in enumerate(targets):
                points, scores = select_points(outputs, index, target['size'], threshold)
                # APGCC xy -> shared PET/P2R metric protocol yx.
                prediction = points.cpu().numpy()[:, ::-1].copy()
                truth = target['points'].numpy()[:, ::-1].copy()
                meter.update(target['image_id'], prediction, truth)
    report = meter.summary()
    report.update(params=sum(p.numel() for p in model.parameters()),
                  evaluation_seconds=time.perf_counter() - start)
    payload = meter.export(output_dir, report, dict(epoch_index=epoch, split=loader.dataset.split,
                          prediction_threshold=threshold, padding_predictions='removed',
                          weights='model', resized=False))
    # Source-aware diagnostics are post-hoc summaries, not separate training.
    groups = defaultdict(list)
    for row in meter.rows:
        groups[row['image_id'].split('__', 1)[0]].append(row)
    group_reports = []
    for name, rows in sorted(groups.items()):
        group_meter = PointMetrics(radii)
        group_meter.rows = rows
        group_reports.append(dict(source=name, **group_meter.summary()))
    with (Path(output_dir) / 'per_source.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_reports[0]))
        writer.writeheader()
        writer.writerows(group_reports)
    return report
