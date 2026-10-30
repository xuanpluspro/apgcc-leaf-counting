"""Single-model FP32 forward timing; excludes I/O, transfers and point matching."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from runtime import resolve_device

from evaluate import load_model


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('benchmark.json'))
    p.add_argument('--size',type=int,default=512)
    p.add_argument('--device',default='cuda')
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--flops',action='store_true')
    args=p.parse_args()
    if args.size < 32 or args.size % 16 or args.warmup < 0 or args.iterations < 1:
        p.error('Invalid size/warmup/iterations')
    torch.manual_seed(2024)
    torch.backends.cudnn.benchmark=False
    device=resolve_device(args.device)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    model,epoch=load_model(args.checkpoint,device)
    model.float().eval()
    value=torch.randn(1,3,args.size,args.size,device=device)
    def sync():
        if device.type=='cuda':
            torch.cuda.synchronize(device)
    with torch.inference_mode(),torch.autocast(device.type,enabled=False):
        for _ in range(args.warmup):
            model(value)
        sync()
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        times=[]
        for _ in range(args.iterations):
            sync();start=time.perf_counter();model(value);sync()
            times.append((time.perf_counter()-start)*1000)
    result=dict(model='APGCC reproduction',params=sum(p.numel() for p in model.parameters()),
                trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                batch_size=1,input_size=[args.size,args.size],precision='FP32',
                latency_ms=statistics.mean(times),median_latency_ms=statistics.median(times),
                fps=1000/statistics.mean(times),gflops=None,counted_gflops=None,
                torch_version=torch.__version__,cuda_version=torch.version.cuda,
                device=str(device),timing='forward only; excludes I/O, transfer, matching and visualization')
    if device.type=='cuda':
        result.update(gpu=torch.cuda.get_device_name(device),
                      peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if args.flops:
        try:
            from fvcore.nn import FlopCountAnalysis
            with torch.inference_mode():
                profile=FlopCountAnalysis(model,value)
                profile.unsupported_ops_warnings(False)
                counted=profile.total()/1e9
                unsupported=dict(profile.unsupported_ops())
            result.update(counted_gflops=counted,unsupported_ops=unsupported,
                          gflops=counted if not unsupported else None,
                          flop_convention='fvcore: one multiply-add counts as one operation')
        except Exception as error:
            result['flops_error']=str(error)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
