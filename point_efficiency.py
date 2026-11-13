"""Export model efficiency once, without re-running dataset evaluation."""
import csv
import json
import os
import time
from pathlib import Path


FIELDS = ('parameters', 'params_m', 'gflops', 'counted_gflops', 'flops_complete',
          'fps', 'latency_ms', 'profile_size', 'profile_batch', 'profile_precision',
          'gpu_name', 'profile_scope', 'profile_error')


def export_efficiency(model, output, pet=False):
    import torch
    from torch import nn
    output = Path(output)
    n = sum(p.numel() for p in model.parameters())
    row = dict(parameters=n, params_m=n / 1e6, flops_complete=False,
               profile_size=512, profile_batch=1, profile_precision='FP32',
               profile_scope='model forward only; seeded synthetic RGB input')
    device = next(model.parameters()).device
    modes = [(m, m.training) for m in model.modules()]
    backends = (torch.backends.cuda.matmul, torch.backends.cudnn)
    tf32 = [b.allow_tf32 for b in backends]
    errors = []
    class Forward(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            if pet:
                from util.misc import NestedTensor
                mask = torch.zeros_like(x[:, 0], dtype=torch.bool)
                return self.model(NestedTensor(x, mask), test=True)
            return self.model(x)

    try:
        model.eval()
        for b in backends:
            b.allow_tf32 = False
        with torch.inference_mode(), torch.autocast(device.type, enabled=False):
            x = torch.rand((1, 3, 512, 512), generator=torch.Generator().manual_seed(2024)).to(device)
            forward = Forward()
            if device.type == 'cuda':
                try:
                    row['gpu_name'] = torch.cuda.get_device_name(device)
                    for _ in range(20):
                        forward(x)
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    for _ in range(100):
                        forward(x)
                    torch.cuda.synchronize(device)
                    seconds = time.perf_counter() - start
                    row.update(fps=100 / seconds, latency_ms=1000 * seconds / 100)
                except Exception as exc:
                    errors.append(f'FPS: {exc}')
            try:
                from fvcore.nn import FlopCountAnalysis
                analysis = FlopCountAnalysis(forward, (x,)).unsupported_ops_warnings(False)
                analysis.uncalled_modules_warnings(False).tracer_warnings('none')
                row['counted_gflops'] = analysis.total() / 1e9
                unsupported = dict(analysis.unsupported_ops())
                row['flops_complete'] = not unsupported
                if not unsupported:
                    row['gflops'] = row['counted_gflops']
                else:
                    errors.append(f'Unsupported FLOP operators: {unsupported}')
            except ImportError:
                errors.append('Install fvcore to enable GFLOPs')
            except Exception as exc:
                errors.append(f'GFLOPs: {exc}')
    except Exception as exc:
        errors.append(str(exc))
    finally:
        for b, value in zip(backends, tf32):
            b.allow_tf32 = value
        for module, training in modes:
            module.training = training
    if errors:
        row['profile_error'] = '; '.join(errors)
    best = output / 'best_metrics.json'
    if best.is_file():
        report = json.loads(best.read_text(encoding='utf-8'))
        metrics = report.get('metrics', report)
        row.update({key: value for key, value in metrics.items()
                    if key not in FIELDS and isinstance(value, (int, float))})
        if isinstance(report.get('epoch_index'), int):
            row['best_epoch'] = report['epoch_index'] + 1
    target = output / 'results.csv'
    tmp = target.with_suffix('.csv.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[*FIELDS, *(k for k in row if k not in FIELDS)])
        writer.writeheader()
        writer.writerow(row)
    os.replace(tmp, target)
    print(f'Efficiency and validation summary: {target}', flush=True)
