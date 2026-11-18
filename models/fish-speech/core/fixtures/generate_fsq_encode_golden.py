"""Torch fixture for source-level ResidualFSQ 1.14.24 encode equations (not imported package)."""

from pathlib import Path

import torch

LEVELS = [8, 5, 5, 5]


def weights(n, seed):
    return torch.tensor([((i * 13 + seed) % 37) / 40.0 - 0.45 for i in range(n)], dtype=torch.float32)


def bias(n, seed):
    return torch.tensor([((i * 7 + seed) % 13) / 30.0 - 0.2 for i in range(n)], dtype=torch.float32)


def bound(x):
    levels = torch.tensor(LEVELS, dtype=torch.float32)
    half_l = (levels - 1.0) * (1.0 + 1e-3) / 2.0
    offset = torch.where(levels.to(torch.long) % 2 == 0, 0.5, 0.0)
    shift = torch.atanh(offset / half_l)
    return (x + shift).tanh() * half_l - offset


def run(batch, groups, q, frames, width, seed):
    values = torch.tensor([((i * 11 + seed) % 103) / 22.0 - 2.2
                           for i in range(batch * groups * width * frames)], dtype=torch.float32)
    features = values.view(batch, groups * width, frames)
    indices = torch.zeros((batch, groups * q, frames), dtype=torch.int32)
    levels = torch.tensor(LEVELS, dtype=torch.int32)
    half_width = levels // 2
    basis = torch.tensor([1, 8, 40, 200], dtype=torch.int32)
    for b in range(batch):
        for g in range(groups):
            weight = weights(width * 4, seed + g).view(4, width)
            offs = bias(4, seed + g + 1)
            for t in range(frames):
                projected = torch.mv(weight, features[b, g * width:(g+1) * width, t]) + offs
                residual = bound(projected)
                for qi in range(q):
                    scale = (levels.float() - 1).pow(-qi)
                    rounded = bound(residual / scale).round()
                    normalized = rounded / half_width
                    digit = rounded.to(torch.int32) + half_width
                    indices[b, g * q + qi, t] = (digit * basis).sum()
                    residual -= normalized * scale
    return features, indices


def main():
    lines = ["# synthetic PyTorch source-formula golden; vector_quantize_pytorch package not imported"]
    for name, b, g, q, t, width, seed in [
        ("q1_g8", 1, 8, 1, 3, 5, 7),
        ("q3_g2", 2, 2, 3, 4, 3, 11),
        ("q2_g1", 1, 1, 2, 5, 4, 23),
    ]:
        features, indices = run(b, g, q, t, width, seed)
        lines.append(f"{name} {b},{g},{q},{t},{width},{seed} " + ",".join(map(str, indices.flatten().tolist())))
    Path(__file__).with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
