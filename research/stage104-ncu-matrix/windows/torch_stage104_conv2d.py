"""Stage104 per-case PyTorch Conv2d CUDA probe with matched deterministic tensors."""

import argparse
import json
import statistics
import time
from pathlib import Path

SHAPES = (
    ("dense3x3", 2, 32, 32, 64, 64, 3, 1, 1, 1, 1),
    ("dense3x3_wide", 1, 64, 64, 48, 48, 3, 1, 1, 1, 1),
    ("pointwise1x1", 2, 16, 32, 80, 64, 1, 0, 1, 1, 1),
    ("depthwise3x3", 1, 64, 64, 40, 48, 3, 1, 1, 1, 64),
)


def run(backend="cuda", dtype="fp32", layout="nchw", repeats=8, selected_case="all"):
    import torch
    import torch.nn.functional as F

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    if not 2 <= repeats <= 64:
        raise ValueError("repeats must be 2..64")
    device = torch.device(backend)
    precision = torch.float32 if dtype == "fp32" else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = []
    if selected_case != "all" and selected_case not in [s[0] for s in SHAPES]:
        raise ValueError(f"unknown case: {selected_case}")
    for name, b, ci, co, h, w, k, pad, stride, dilation, groups in SHAPES:
        if selected_case != "all" and name != selected_case:
            continue
        n_in = b * ci * h * w
        n_w = co * (ci // groups) * k * k
        x = ((torch.arange(n_in, device=device) * 37 % 149).float() - 70.0) * 0.005
        weight = ((torch.arange(n_w, device=device) * 53 % 211).float() - 90.0) * 0.001
        x = x.to(precision).view(b, ci, h, w)
        weight = weight.to(precision).view(co, ci // groups, k, k)
        if layout == "nhwc":
            x = x.contiguous(memory_format=torch.channels_last)
            weight = weight.contiguous(memory_format=torch.channels_last)
        times = []
        out = None
        for iteration in range(repeats):
            if backend == "cuda":
                torch.cuda.synchronize()
                if iteration > 0:
                    torch.cuda.nvtx.range_push(f"stage104_conv2d_{name}")
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    out = F.conv2d(x, weight, None, stride=stride, padding=pad, dilation=dilation, groups=groups)
                if backend == "cuda":
                    torch.cuda.synchronize()
            finally:
                if backend == "cuda" and iteration > 0:
                    torch.cuda.nvtx.range_pop()
            if iteration > 0:
                times.append((time.perf_counter() - t0) * 1000)
        vals = out.float().contiguous().flatten()
        sample_indices = [0, vals.numel() // 3, vals.numel() // 2, vals.numel() - 1]
        cases.append({
            "name": name, "shape": [b, ci, h, w], "weight_shape": [co, ci // groups, k, k],
            "padding": pad, "stride": stride, "dilation": dilation, "groups": groups,
            "output_elements": vals.numel(),
            "sample_values": [float(vals[i]) for i in sample_indices],
            "mean": float(vals.mean()), "median_ms": statistics.median(times),
            "repeats": repeats, "dtype": dtype, "layout": layout,
        })
    return {"stage": 104, "engine": "torch", "backend": backend,
            "tf32_enabled": False, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--layout", choices=("nchw", "nhwc"), default="nchw")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--case", choices=("all",) + tuple(s[0] for s in SHAPES), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.backend, args.dtype, args.layout, args.repeats, args.case)
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
