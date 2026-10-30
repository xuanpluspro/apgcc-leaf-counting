"""Evaluate a trained model without optimizer/teacher/backbone downloads."""
import argparse
from pathlib import Path

import torch
from runtime import resolve_device
from torch.utils.data import DataLoader

from checkpointing import load
from dataset import RiceW3999, collate
from engine import evaluate
from models import APGCC, ModelConfig


def load_model(checkpoint, device):
    state = load(checkpoint)
    if 'model_config' not in state or 'model' not in state:
        raise ValueError('Expected a checkpoint exported by this repository; official checkpoints '
                         'require a separate compatibility audit (see REPRODUCTION.md)')
    model = APGCC(ModelConfig(**state['model_config']))
    model.load_state_dict(state['model'], strict=True)
    model = model.to(device).eval()
    return model, state.get('epoch')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-path', type=Path, required=True)
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--output', type=Path, default=Path('evaluation'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--loc-radii', type=float, nargs='+', default=[4.,8.,16.])
    args = p.parse_args()
    device = resolve_device(args.device)
    if not 0 < args.threshold < 1:
        p.error('threshold must be between zero and one')
    model, epoch = load_model(args.checkpoint, device)
    dataset = RiceW3999(args.data_path, args.split, augment=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=collate,
                        pin_memory=device.type=='cuda')
    print(evaluate(model, loader, device, args.output, args.threshold, args.loc_radii, epoch))


if __name__ == '__main__':
    main()
