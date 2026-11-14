import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def make_f0(length):
    return np.asarray(
        [0.0 if i % 17 < 3 else 110.0 + (i * 73 % 481) * 0.2 for i in range(length)],
        dtype=np.float32,
    )


def reflect(index, length):
    if length <= 1:
        return 0
    period = 2 * (length - 1)
    index %= period
    return period - index if index >= length else index


def fundamental(f0, phase=0.0):
    frequency = torch.as_tensor(f0, dtype=torch.float32)
    factors = torch.arange(1, 401, dtype=torch.float32)
    rad = frequency[:, None].div(40_000.0) * factors[None, :]
    frame_end = rad[:-1, -1]
    wrapped = ((frame_end + 0.5).fmod(1.0) - 0.5)
    prefix = torch.cat((torch.zeros((1, 1)), wrapped.cumsum(0).fmod(1.0)[:, None]), 0)
    return torch.sin((rad + prefix + phase).reshape(-1) * (2.0 * torch.pi)).mul_(0.1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    text = subprocess.check_output([str(args.probe)], text=True)
    offsets = {int(first): float(value) for first, value in (line.split() for line in text.splitlines())}
    f0 = make_f0(9_000)
    reference = fundamental(f0)
    all_deltas = []
    untreated = []
    for first, offset in offsets.items():
        left = first - 40
        length = min(300, len(f0) - first)
        local_f0 = np.asarray([f0[reflect(left + i, len(f0))] for i in range(length + 84)], dtype=np.float32)
        corrected = fundamental(local_f0, offset)[40 * 400:(40 + length) * 400]
        naive = fundamental(local_f0)[40 * 400:(40 + length) * 400]
        expected = reference[first * 400:(first + length) * 400]
        all_deltas.append(corrected - expected)
        untreated.append(naive - expected)
    corrected_delta = np.concatenate(all_deltas).astype(np.float64)
    untreated_delta = np.concatenate(untreated).astype(np.float64)
    result = {
        "backend": "PyTorch CPU",
        "torch_version": torch.__version__,
        "frames": len(f0),
        "segments": len(offsets),
        "samples_compared": corrected_delta.size,
        "global_phase_max_abs": float(np.max(np.abs(corrected_delta))),
        "global_phase_rmse": float(np.sqrt(np.mean(np.square(corrected_delta)))),
        "reset_phase_max_abs": float(np.max(np.abs(untreated_delta))),
        "reset_phase_rmse": float(np.sqrt(np.mean(np.square(untreated_delta)))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    assert result["global_phase_rmse"] < result["reset_phase_rmse"] * 0.01
    assert result["global_phase_max_abs"] < 0.0001


if __name__ == "__main__":
    main()
