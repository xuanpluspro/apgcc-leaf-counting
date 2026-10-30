"""Single-GPU APGCC reproduction for RiceW3999 leaf-tip points."""
import argparse
import json
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from runtime import resolve_device
from torch.utils.data import DataLoader

import checkpointing
from counting_metrics import append_history
from dataset import RiceW3999, collate, seed_worker
from engine import evaluate, train_epoch
from losses import PointCriterion
from models import APGCC, ModelConfig

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-path', type=Path, default=ROOT/'data/RiceW3999')
    p.add_argument('--pretrained', type=Path, default=ROOT/'pretrained/vgg16_bn-6c64b313.pth')
    p.add_argument('--scratch', action='store_true', help='Explicitly train without ImageNet initialization')
    p.add_argument('--tag', default='ricew3999_ep50_bs32')
    p.add_argument('--output-root', type=Path, default=ROOT/'exp')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--patch-size', type=int, default=512)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-backbone', type=float, default=1e-5)
    p.add_argument('--lr-drop', type=int, default=300)
    p.add_argument('--lr-gamma', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--clip-grad', type=float, default=.1)
    p.add_argument('--point-weight', type=float, default=.0002)
    p.add_argument('--aux-weight', type=float, default=.2)
    p.add_argument('--no-aux', action='store_true', help='Ablation only: disable auxiliary point supervision')
    p.add_argument('--amp', action='store_true', help='Optional CUDA FP16 autocast; default FP32')
    p.add_argument('--accum-steps', type=int, default=1)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--loc-radii', type=float, nargs='+', default=[4.,8.,16.])
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', type=Path, help='Trusted checkpoint or experiment output directory')
    p.add_argument('--auto-resume', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--allow-data-change', action='store_true')
    return p


def main(args):
    if min(args.batch_size, args.epochs, args.eval_every, args.accum_steps, args.lr_drop) < 1:
        raise ValueError('batch-size, epochs, eval-every, accum-steps and lr-drop must be positive')
    if args.workers < 0 or not 0 < args.threshold < 1 or args.clip_grad <= 0:
        raise ValueError('Invalid workers, probability threshold or clip-grad')
    if not args.tag or Path(args.tag).name != args.tag or args.tag in ('.', '..'):
        raise ValueError('tag must be a single directory name')
    device = resolve_device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu only for diagnostics')
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp requires CUDA')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = args.output_root / args.tag / 'output'
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('apgcc')
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(output/'train.log', mode='a')):
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)
    latest = output / 'checkpoint.pth'
    resume = args.resume or (latest if args.auto_resume and latest.is_file() else None)
    if resume and resume.is_dir():
        resume /= 'checkpoint.pth'
    if latest.exists() and not resume:
        raise FileExistsError('Existing checkpoint: use --auto-resume/--resume or a new --tag')
    state = checkpointing.load(resume) if resume else None
    # Full checkpoint already contains the backbone: no pretrained dependency on resume.
    pretrained = None if state or args.scratch else args.pretrained
    model = APGCC(ModelConfig(aux_enabled=not args.no_aux), pretrained).to(device)
    criterion = PointCriterion(args.point_weight, args.aux_weight).to(device)
    optimizer = torch.optim.Adam([
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('encoder.')]),
        dict(params=model.encoder.parameters(), lr=args.lr_backbone)],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.lr_gamma)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    training = RiceW3999(args.data_path, 'train', args.patch_size)
    validation = RiceW3999(args.data_path, 'val', args.patch_size)
    overlap = set(training.ids) & set(validation.ids)
    if overlap:
        raise ValueError(f'Train/val IDs overlap: {sorted(overlap)[:3]}')
    fingerprint = dict(train=training.fingerprint(), val=validation.fingerprint())
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              pin_memory=device.type=='cuda', worker_init_fn=seed_worker)
    val_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=args.workers,
                            collate_fn=collate, pin_memory=device.type=='cuda', worker_init_fn=seed_worker)
    start, best_mae, best_epoch = 0, float('inf'), -1
    if state:
        start, best_mae, best_epoch = checkpointing.restore(
            state, model, optimizer, scheduler, scaler, args, fingerprint, logger)
        for name in ('metrics_history.csv', 'train_history.csv'):
            checkpointing.trim_history(output/name, start)
        logger.info(f'Resumed {resume}; completed {start} epochs')
        del state
    config = dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  model=model.config.to_dict(), data_fingerprint=fingerprint,
                  precision='FP16 AMP' if args.amp else 'FP32', torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, device=str(device),
                  gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
    if start == 0:
        (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    else:
        (output/'resume_config.json').write_text(json.dumps(config, indent=2)+'\n')
    logger.info(f'Train={len(training)} val={len(validation)} | patch={args.patch_size} | '
                f'Params={sum(p.numel() for p in model.parameters())/1e6:.2f}M | '
                f'auxiliary={not args.no_aux} | {config["precision"]}')
    if start >= args.epochs:
        logger.info('Requested total epochs already completed')
        return
    for epoch in range(start, args.epochs):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        losses = train_epoch(model, criterion, train_loader, optimizer, scaler, epoch, args, device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - begin
        peak = torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0.
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        append_history(output/'train_history.csv', dict(epoch=epoch+1, **losses,
                       lr=lr, train_seconds=elapsed, train_peak_allocated_mib=peak))
        logger.info(f'Epoch {epoch+1}/{args.epochs} | loss={losses["loss"]:.5f} | '
                    f'time={elapsed:.1f}s | lr={lr:.1e} | train peak={peak:.0f}MiB')
        is_best = False
        if (epoch+1) % args.eval_every == 0 or epoch+1 == args.epochs:
            evaluation_dir = output/'evaluation'/f'epoch_{epoch+1:04d}'
            report = evaluate(model, val_loader, device, evaluation_dir,
                              args.threshold, args.loc_radii, epoch)
            append_history(output/'metrics_history.csv', dict(epoch=epoch+1,
                           train_seconds=elapsed, train_peak_allocated_mib=peak, **report))
            is_best = report['mae'] < best_mae
            if is_best:
                best_mae, best_epoch = report['mae'], epoch
            logger.info(f'Validation | MAE={report["mae"]:.3f} RMSE={report["rmse"]:.3f} '
                        f'F1@8px={report.get("f1_8px",float("nan")):.4f} | '
                        f'best MAE={best_mae:.3f} at epoch {best_epoch+1}')
        if is_best:
            checkpointing.save(output/'best_checkpoint.pth', model, optimizer, scheduler,
                               scaler, epoch, args, best_mae, best_epoch, fingerprint)
            shutil.copyfile(evaluation_dir/'metrics.json', output/'best_metrics.json')
        checkpointing.save(latest, model, optimizer, scheduler, scaler, epoch,
                           args, best_mae, best_epoch, fingerprint)
        logger.info(f'Checkpoint saved | completed epoch {epoch+1}')


if __name__ == '__main__':
    main(parser().parse_args())
