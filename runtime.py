"""Resolve explicit CUDA indices before calling set_device (cuda -> cuda:0)."""
import torch


def resolve_device(name):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; --device cpu is for diagnostics only')
        device = torch.device('cuda', device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_device(device)
    elif device.type != 'cpu':
        raise ValueError('Supported devices: cpu or cuda[:index]')
    return device
