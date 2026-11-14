"""Diagnostic CPU-only operator timings; NOT a CUDA or full-WAV speed comparison."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from spectral_conv import SpectralConv1d, candidate_plans
from test_spectral_conv import normalized_weight


def timed(call, repeats=3):
    call()
    samples=[]
    for _ in range(repeats):
        s=perf_counter()
        call()
        samples.append((perf_counter()-s)*1000)
    return median(samples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--frames',type=int,default=3840)
    args=p.parse_args()
    torch.set_num_threads(4)
    w=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['weight']
    rng=np.random.default_rng(891771)
    rows=[]
    for block,layer,dilation in ((1,0,1),(2,2,5),(4,0,1),(5,2,5),(8,2,5)):
        prefix=f'dec.resblocks.{block}.convs1.{layer}'
        weights=normalized_weight(w,prefix)
        bias=w[prefix+'.bias'].float().numpy()
        c,_,k=weights.shape
        frames=args.frames
        x=rng.normal(size=(1,c,frames)).astype('float32')*.05
        x_t=torch.from_numpy(x)
        w_t=torch.from_numpy(weights)
        b_t=torch.from_numpy(bias)
        pad=(k-1)*dilation//2
        plans=candidate_plans(c,c,k,dilation,frames)
        plan=plans[0]
        conv=SpectralConv1d(weights,bias,dilation,plan.nfft)
        ref=F.conv1d(x_t,w_t,b_t,dilation=dilation,padding=pad).numpy()
        got=conv.forward(x,'three_real_gemm',windows_per_batch=16)
        max_delta=float(np.abs(got-ref).max())
        direct_ms=timed(lambda: F.conv1d(x_t,w_t,b_t,dilation=dilation,padding=pad))
        fft_ms=timed(lambda: conv.forward(x,'three_real_gemm',windows_per_batch=16))
        row={'layer':prefix,'channels':c,'kernel':k,'dilation':dilation,
             'frames':frames,'nfft':plan.nfft,'windows':plan.blocks,
             'flop_ratio':round(plan.ideal_flop_ratio,3),'max_abs':max_delta,
             'torch_cpu_ms':round(direct_ms,3),'numpy_fft_cpu_ms':round(fft_ms,3),
             'fft_vs_torch_cpu_ratio':round(direct_ms/fft_ms,3)}
        rows.append(row)
        print(row,flush=True)
        if max_delta>5e-5:raise SystemExit('Numerical parity failed')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(rows,indent=2),encoding='utf8')


if __name__=='__main__':main()
