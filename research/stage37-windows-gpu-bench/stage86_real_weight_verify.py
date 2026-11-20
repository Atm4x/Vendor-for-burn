"""CPU reference check of exact tiled nonlinear NSF pair using a real checkpoint."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def loaded_pair(state, block: int, pair: int):
    result = []
    for family in ("convs1", "convs2"):
        prefix = f"dec.resblocks.{block}.{family}.{pair}"
        v = state[f"{prefix}.weight_v"].float()
        g = state[f"{prefix}.weight_g"].float()
        w = v * (g / v.flatten(1).norm(dim=1).reshape(-1, 1, 1))
        bias = state[f"{prefix}.bias"].float()
        result.append((w, bias))
    return result


def tiled_pair(x, w1, b1, w2, b2, dilation: int, tile_time: int):
    total = x.shape[-1]
    kernel1 = w1.shape[-1]
    kernel2 = w2.shape[-1]
    radius1 = (kernel1 - 1) // 2 * dilation
    radius2 = (kernel2 - 1) // 2
    result = torch.empty_like(x)
    for start in range(0, total, tile_time):
        end = min(start + tile_time, total)
        ext_start = start - radius1 - radius2
        ext_end = end + radius1 + radius2
        segment = torch.zeros((x.shape[0], x.shape[1], ext_end - ext_start), dtype=x.dtype)
        lo, hi = max(ext_start, 0), min(ext_end, total)
        if hi > lo:
            segment[..., lo - ext_start:hi - ext_start] = x[..., lo:hi]
        middle = F.conv1d(F.leaky_relu(segment, 0.1), w1, b1, dilation=dilation)
        positions = torch.arange(start-radius2, end+radius2)
        middle[..., (positions < 0) | (positions >= total)] = 0
        out = F.conv1d(F.leaky_relu(middle, 0.1), w2, b2)
        result[..., start:end] = x[..., start:end] + out
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True)
    parser.add_argument("--out", default="stage86_real_weight_results.json")
    args = parser.parse_args()
    torch.manual_seed(12345)
    torch.set_num_threads(2)
    state = torch.load(args.voice, weights_only=True, map_location="cpu")["weight"]
    cases = []
    for block, pair, dilation in [(9,0,1),(10,1,3),(11,2,5),(6,2,5),(8,0,1)]:
        (w1,b1),(w2,b2) = loaded_pair(state,block,pair)
        channels = w1.shape[0]
        for time in [17,31,64]:
            x = torch.randn(2,channels,time, dtype=torch.float32) * 0.2
            radius1 = (w1.shape[-1]-1)//2*dilation
            radius2 = (w2.shape[-1]-1)//2
            exact = x+F.conv1d(F.leaky_relu(F.conv1d(F.leaky_relu(x,0.1),w1,b1,padding=radius1,dilation=dilation),0.1),w2,b2,padding=radius2)
            candidate = tiled_pair(x,w1,b1,w2,b2,dilation,16)
            maximum = (exact-candidate).abs().max().item()
            rmse = (exact-candidate).square().mean().sqrt().item()
            cases.append({"block":block,"pair":pair,"channels":channels,"time":time,"dilation":dilation,"max_abs":maximum,"rmse":rmse})
            assert maximum < 4e-6, (block,pair,time,maximum)
    Path(args.out).write_text(json.dumps({"cases":cases,"torch":torch.__version__}, indent=2))
    print(f"PASS: {len(cases)} real-weight tiled NSF pairs (max_abs={max(x['max_abs'] for x in cases):.8g})")


if __name__ == '__main__':
    main()
