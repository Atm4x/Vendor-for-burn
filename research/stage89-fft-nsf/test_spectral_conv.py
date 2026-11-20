"""CPU-only exact-topology tests, including real CrazyMita checkpoint weights."""

import argparse
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from spectral_conv import SpectralConv1d, candidate_plans


def normalized_weight(weight_map: dict, prefix: str) -> np.ndarray:
    v = weight_map[prefix + ".weight_v"].float()
    g = weight_map[prefix + ".weight_g"].float()
    weight = v * (g / v.square().sum(dim=(1, 2), keepdim=True).sqrt())
    return weight.numpy()


class SpectralRegression(unittest.TestCase):
    def test_exact_topology_for_odd_kernels_and_dilations(self):
        rng = np.random.default_rng(8719)
        cases = 0
        for kernel in (1, 3, 7, 11):
            for dilation in (1, 2, 3, 5):
                for frames in (1, 7, 62, 129):
                    effective = (kernel - 1) * dilation + 1
                    nfft = 1 << max(6, (effective + 5 - 1).bit_length())
                    x = rng.normal(size=(2, 3, frames)).astype(np.float32)
                    w = rng.normal(size=(4, 3, kernel)).astype(np.float32) / 6
                    bias = rng.normal(size=(4,)).astype(np.float32)
                    ref = F.conv1d(torch.from_numpy(x), torch.from_numpy(w),
                                   torch.from_numpy(bias), padding=(effective - 1) // 2,
                                   dilation=dilation).numpy()
                    conv = SpectralConv1d(w, bias, dilation, nfft)
                    for algo in ("complex", "three_real_gemm"):
                        for windows_per_batch in (1, 2, 7):
                            got = conv.forward(x, algo, windows_per_batch)
                            self.assertEqual(got.shape, ref.shape)
                            np.testing.assert_allclose(got, ref, atol=2e-5, rtol=2e-5,
                                err_msg=f"kernel={kernel} dilation={dilation} frames={frames} {algo}, windows={windows_per_batch}")
                    cases += 1
        print(f"Synthetic cases: {cases} x 2 algorithms x 3 window batches")

    def test_selector_respects_hardware_and_kernel(self):
        limited = candidate_plans(128, 128, 11, 5, 38400,
                                  max_spectral_weight_bytes=20 * 1024 * 1024)
        self.assertTrue(limited)
        self.assertTrue(all(p.spectral_weight_bytes <= 20 * 1024 * 1024 for p in limited))
        self.assertTrue(all(p.nfft >= p.effective_kernel for p in limited))
        self.assertFalse(candidate_plans(128, 128, 11, 5, 38400,
                                         max_spectral_weight_bytes=1000))
        with self.assertRaises(ValueError):
            candidate_plans(64, 64, 4, 1, 10)


def real_checkpoint_probe(path: Path, limit: int = 16) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    weights = ckpt['weight']
    rng = np.random.default_rng(8989)
    rows = []
    max_abs = 0.0
    for index in range(12):
        for family in ("convs1", "convs2"):
            for layer in range(3):
                prefix = f"dec.resblocks.{index}.{family}.{layer}"
                weight = normalized_weight(weights, prefix)
                c, _, k = weight.shape
                dilation = (1, 3, 5)[layer] if family == "convs1" else 1
                plans = candidate_plans(c, c, k, dilation, output_frames=384)
                if not plans:
                    continue
                plan = plans[0]
                row = {
                    "layer":prefix, "channels": c, "kernel": k, "dilation": dilation,
                    "nfft":plan.nfft, "valid_frames":plan.valid_frames,
                    "estimated_flop_ratio": round(plan.ideal_flop_ratio, 3),
                    "spectral_weight_mebibytes": round(plan.spectral_weight_bytes/(1024**2), 3),
                }
                if k >= 7 and c >= 64 and len([r for r in rows if 'max_abs' in r]) < limit:
                    length = 70
                    x = rng.normal(size=(1,c,length)).astype(np.float32) * .05
                    weight_t = torch.from_numpy(weight)
                    bias = weights[prefix + '.bias'].float().numpy()
                    reference = F.conv1d(torch.from_numpy(x),weight_t,
                        torch.from_numpy(bias),padding=(k-1)*dilation//2,dilation=dilation).numpy()
                    conv = SpectralConv1d(weight,bias,dilation,plan.nfft)
                    out = conv.forward(x, 'three_real_gemm', windows_per_batch=4)
                    delta = out - reference
                    row['max_abs'] = float(np.max(np.abs(delta)))
                    row['rmse'] = float(np.sqrt(np.mean(delta.astype(np.float64)**2)))
                    max_abs = max(max_abs,row['max_abs'])
                rows.append(row)
    return {"layers":rows,"tested":sum('max_abs' in r for r in rows),"max_abs":max_abs}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--real-limit',type=int,default=16)
    args=parser.parse_args()
    unittest.main(argv=[sys.argv[0]],exit=False,verbosity=2)
    if args.checkpoint:
        report=real_checkpoint_probe(args.checkpoint,args.real_limit)
        print(f"Real checkpoint: {report['tested']} layers, max_abs={report['max_abs']:.3e}")
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
