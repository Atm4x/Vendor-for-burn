"""CPU-only study of real-checkpoint NSF residual convolution precision."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def normalized_layer(weights, prefix):
    v = weights[f"{prefix}.weight_v"].float()
    g = weights[f"{prefix}.weight_g"].float()
    norms = torch.linalg.vector_norm(v.reshape(v.shape[0], -1), dim=1)
    w = v * (g.reshape(-1) / norms).reshape(-1, 1, 1)
    b = weights[f"{prefix}.bias"].float()
    return w.contiguous(), b.contiguous()


def resblock(x, layers, dilations, precision):
    x = x.to(dtype=torch.float16 if precision == "island" else torch.float32)
    for j, ((w1, b1), (w2, b2)) in enumerate(layers):
        if precision == "fp32":
            h = F.conv1d(F.leaky_relu(x, 0.1), w1, b1,
                         padding=(w1.shape[-1] - 1) * dilations[j] // 2,
                         dilation=dilations[j])
            h = F.conv1d(F.leaky_relu(h, 0.1), w2, b2,
                         padding=(w2.shape[-1] - 1) // 2)
        elif precision == "mixed":
            h = F.conv1d(F.leaky_relu(x, 0.1).half(), w1.half(), b1.half(),
                         padding=(w1.shape[-1] - 1) * dilations[j] // 2,
                         dilation=dilations[j]).float()
            h = F.conv1d(F.leaky_relu(h, 0.1).half(), w2.half(), b2.half(),
                         padding=(w2.shape[-1] - 1) // 2).float()
        elif precision == "island":
            h = F.conv1d(F.leaky_relu(x, 0.1), w1.half(), b1.half(),
                         padding=(w1.shape[-1] - 1) * dilations[j] // 2,
                         dilation=dilations[j])
            h = F.conv1d(F.leaky_relu(h, 0.1), w2.half(), b2.half(),
                         padding=(w2.shape[-1] - 1) // 2)
        else:
            raise ValueError(precision)
        x = x + h
    return x.float()


def errors(a, b):
    delta = a - b
    return {"rmse": math.sqrt(float((delta * delta).mean())),
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a).clamp_min(1e-12))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--voice", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=35)
    args = p.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.voice, map_location="cpu", weights_only=True)
    weights = checkpoint["weight"]
    config = checkpoint["config"]
    assert config[10] == [3, 7, 11] and config[11] == [[1, 3, 5]] * 3
    strides = config[12]
    assert len(strides) == 4
    generator = torch.Generator().manual_seed(87011)
    records = []
    stage_cost = []
    t = 384
    for stage in range(4):
        t *= strides[stage]
        index = stage * 3
        channels = weights[f"dec.resblocks.{index}.convs1.0.weight_v"].shape[0]
        ks = []
        for branch in range(3):
            prefix = f"dec.resblocks.{index + branch}"
            layers = [(normalized_layer(weights, f"{prefix}.convs1.{j}"),
                       normalized_layer(weights, f"{prefix}.convs2.{j}")) for j in range(3)]
            for pair in layers:
                ks.append(pair[0][0].shape[-1])
            x = torch.randn((1, channels, args.frames), generator=generator) * 0.3
            reference = resblock(x.clone(), layers, (1, 3, 5), "fp32")
            per_conv_cast = resblock(x.clone(), layers, (1, 3, 5), "mixed")
            island = resblock(x.clone(), layers, (1, 3, 5), "island")
            records.append({"stage": stage, "branch": branch, "channels": channels,
                            "weight_source_dtype": str(weights[f"{prefix}.convs1.0.weight_v"].dtype),
                            "island_vs_fp32": errors(reference, island),
                            "per_conv_cast_vs_fp32": errors(reference, per_conv_cast),
                            "island_vs_per_conv_cast": errors(per_conv_cast, island)})
        # 2 convolutions per pair * 3 pairs for each of three branches, 2 FLOP per MAC.
        macs = 2 * 3 * sum([3, 7, 11]) * channels * channels * t
        stage_cost.append({"stage": stage, "channels": int(channels), "frames_per_full_chunk": t,
                           "pair_conv_macs_per_full_chunk": macs,
                           "pair_conv_flops_per_full_chunk": macs * 2})
    result = {
        "stage": 87,
        "model": args.voice.name,
        "precision_scope": "NSF ResBlock precision islands; weights quantized once, each block receives FP16 and outputs F32; internal LeakyReLU and residual use FP16",
        "probe": "CPU PyTorch one random seeded activation per real residual branch; no real-audio activations, not a CUDA or WAV benchmark",
        "input_frames": args.frames,
        "layers": records,
        "cost": stage_cost,
        "cost_scope": "384 input latent frames for a full-sized original Rust chunk, excluding padding boundary overhead and first/last partial chunks",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"max_island_vs_fp32_rmse": max(r["island_vs_fp32"]["rmse"] for r in records),
                      "max_island_vs_per_conv_cast_rmse": max(r["island_vs_per_conv_cast"]["rmse"] for r in records),
                      "total_flops_per_full_chunk": sum(x["pair_conv_flops_per_full_chunk"] for x in stage_cost)}, indent=2))


if __name__ == "__main__":
    main()
