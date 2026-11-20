"""Test exact graph topology of complete residual blocks on the real checkpoint."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from spectral_conv import SpectralConv1d, candidate_plans
from test_spectral_conv import normalized_weight


def compare_resblock(weights: dict, block_id: int, frames: int = 127,
                     method: str = 'three_real_gemm') -> dict:
    rng = np.random.default_rng(89123 + block_id)
    prefix = f'dec.resblocks.{block_id}'
    channels = weights[f'{prefix}.convs1.0.weight_v'].shape[0]
    inp = rng.normal(size=(1, channels, frames)).astype(np.float32) * .05
    torch_state = torch.from_numpy(inp)
    fft_state = inp.copy()
    fft_specs = []
    for l, dilation in enumerate((1, 3, 5)):
        for family, dil in (('convs1', dilation), ('convs2', 1)):
            key = f'{prefix}.{family}.{l}'
            w = normalized_weight(weights, key)
            bias = weights[key + '.bias'].float().numpy()
            k = w.shape[-1]
            plan = candidate_plans(channels,channels,k,dil,frames)[0]
            fft_specs.append((w,bias,dil,plan.nfft))
    for pair in range(3):
        w0,b0,d0,n0 = fft_specs[2*pair]
        w1,b1,d1,n1 = fft_specs[2*pair+1]
        torch_mid = F.conv1d(F.leaky_relu(torch_state, .1),torch.from_numpy(w0),
                             torch.from_numpy(b0),padding=(w0.shape[-1]-1)*d0//2,dilation=d0)
        torch_new = F.conv1d(F.leaky_relu(torch_mid,.1),torch.from_numpy(w1),
                             torch.from_numpy(b1),padding=(w1.shape[-1]-1)*d1//2,dilation=d1)
        torch_state = torch_state + torch_new
        fft_mid = SpectralConv1d(w0,b0,d0,n0).forward(
            np.where(fft_state < 0,fft_state*np.float32(.1),fft_state),method,windows_per_batch=8)
        fft_new = SpectralConv1d(w1,b1,d1,n1).forward(
            np.where(fft_mid < 0,fft_mid*np.float32(.1),fft_mid),method,windows_per_batch=8)
        fft_state += fft_new
    delta=fft_state - torch_state.numpy()
    return {'block':block_id,'channels':int(channels),'frames':frames,'algorithm':method,
        'max_abs':float(np.max(np.abs(delta))),
        'rmse':float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
        'reference_rms':float(np.sqrt(np.mean(torch_state.numpy().astype(np.float64)**2))),
        'l2_relative':float(np.linalg.norm(delta) / max(np.linalg.norm(torch_state.numpy()),1e-12))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    args=p.parse_args()
    weights = torch.load(args.checkpoint,map_location='cpu',weights_only=False)['weight']
    rows=[compare_resblock(weights,i) for i in (2,5,8,11)]
    for row in rows:print(row)
    if max(r['max_abs'] for r in rows) > 5e-5:
        raise SystemExit('Residual block parity regression exceeded 5e-5 tolerance')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
