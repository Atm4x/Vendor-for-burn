from __future__ import annotations

import argparse
from pathlib import Path
import torch

from generate_cfm_pytorch_fixture import load_original_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_archive', type=Path)
    args = parser.parse_args()
    original = load_original_function(args.original_archive)
    fixed_noise = torch.tensor([0.2, -0.4, 0.1, 0.3, 0.55, -0.8, 0.91, 0.13, -0.6, -0.42, 0.15, 0.3,
                                0.21, -0.77, 0.48], dtype=torch.float32)
    cond = torch.tensor([0.45, 0.1, -0.8, 0.91, -0.4, 0.24] + [0.0] * 9, dtype=torch.float32)
    path = Path(__file__).resolve().parents[1] / 'fixtures/cfm_trajectory_pytorch_expected.tsv'
    with path.open('w', encoding='utf-8') as writer:
        for steps, sway, cfg in [(5, -1.0, 2.0), (16, None, 1.75), (32, 0.5, 0.0), (7, -0.3, 3.5)]:
            t = original(steps, 'cpu', torch.float32)
            if sway is not None:
                t = t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
            x = fixed_noise.clone()
            for a, b in zip(t[:-1], t[1:]):
                vc = x * 0.031 + cond * 0.021 + a * 0.17
                vu = x * 0.013 - a * 0.04
                vel = vc if cfg < 1e-5 else vc + (vc - vu) * cfg
                x = x + vel * (b - a)
            x[:6] = cond[:6]
            writer.write(f'{steps}\t{sway if sway is not None else "none"}\t{cfg}\t{",".join(str(v.item()) for v in x)}\n')
    print('Generated PyTorch CFM reference trajectories:', path)


if __name__ == '__main__':
    main()
