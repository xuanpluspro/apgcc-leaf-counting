"""Atomic completed-epoch checkpoints. Load only files you trust."""
import csv
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def atomic_save(state, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Visible CUDA device count changed; keep CUDA_VISIBLE_DEVICES consistent')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save(path, model, optimizer, scheduler, scaler, epoch, args, best_mae, best_epoch, fingerprint):
    atomic_save(dict(format_version=1, model=model.state_dict(), model_config=model.config.to_dict(),
                     optimizer=optimizer.state_dict(), lr_scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), epoch=epoch, args=vars(args), best_mae=best_mae,
                     best_epoch=best_epoch, rng_state=rng_state(), data_fingerprint=fingerprint), path)


def restore(state, model, optimizer, scheduler, scaler, args, fingerprint, logger):
    for key in ('model', 'optimizer', 'lr_scheduler', 'scaler', 'epoch', 'rng_state'):
        if key not in state:
            raise ValueError(f'Not a complete training checkpoint: missing {key}')
    old = state.get('args', {})
    fixed = ('epochs', 'lr', 'lr_backbone', 'lr_drop', 'lr_gamma', 'weight_decay',
             'patch_size', 'amp', 'no_aux', 'aux_weight', 'point_weight', 'seed',
             'accum_steps', 'threshold', 'loc_radii', 'clip_grad')
    for key in fixed:
        if key in old and old[key] != vars(args)[key]:
            raise ValueError(f'Resume setting differs at {key}: {old[key]!r} -> {vars(args)[key]!r}; '
                             'use the original settings (epochs is total planned epochs)')
    if old.get('eval_every', args.eval_every) != args.eval_every:
        logger.info(f'Resume changed validation interval: {old["eval_every"]} -> {args.eval_every}; '
                    'past epochs are not re-evaluated; best model is selected from recorded validations')
    for key in ('batch_size', 'workers', 'data_path'):
        if old.get(key) != vars(args).get(key):
            logger.warning(f'Resume changed {key}; exact training trajectory may differ')
    if state.get('data_fingerprint') != fingerprint:
        if not args.allow_data_change:
            raise ValueError('Dataset fingerprint changed; verify data, then use --allow-data-change if intentional')
        logger.warning('Dataset changed; continuing with --allow-data-change')
    # Check model configuration before tensors: equal tensor shapes can still
    # encode different auxiliary geometry/semantics.
    expected = json.loads(json.dumps(model.config.to_dict()))
    if json.loads(json.dumps(state['model_config'])) != expected:
        raise ValueError('Model configuration differs from checkpoint')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['lr_scheduler'])
    scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng_state'])
    return int(state['epoch']) + 1, state['best_mae'], state['best_epoch']


def trim_history(path, completed_epochs):
    path = Path(path)
    if not path.is_file():
        return
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    kept = [row for row in rows if int(row['epoch']) <= completed_epochs]
    if len(kept) == len(rows):
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
