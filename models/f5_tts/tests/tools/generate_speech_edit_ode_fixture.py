"""Independent PyTorch numerical oracle for Euler/midpoint CFM masked speech editing."""
import json
from pathlib import Path
import struct

import torch

ROOT = Path(__file__).resolve().parents[1] / 'fixtures'


def make_timeline(steps, epss, sway):
    if epss and steps == 5:
        t = torch.tensor([0, 2, 4, 8, 16, 32], dtype=torch.float32) / 32.0
    else:
        t = torch.linspace(0, 1, steps + 1, dtype=torch.float32)
    if sway is not None:
        t = t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t


def velocity(x, condition, time, cfg):
    conditional = x * 0.11 + condition * 0.05 + time * 0.3
    unconditional = x * 0.07 - condition * 0.02 - time * 0.1
    return conditional + (conditional - unconditional) * cfg


def generate():
    frames = 8
    count = frames * 100
    initial = torch.arange(count, dtype=torch.float32).reshape(1, frames, 100) / 1000.0 - 0.3
    mel = torch.arange(count, dtype=torch.float32).reshape(1, frames, 100) / 200.0 - 0.1
    keep = [True, True, False, True, False, False, True, False]
    mask = torch.tensor(keep, dtype=torch.bool).reshape(1, frames, 1)
    condition = torch.where(mask, mel, torch.zeros_like(mel))
    configs = [
        {'method': 'euler', 'steps': 3, 'epss': False, 'sway': None, 'cfg': 1.4},
        {'method': 'midpoint', 'steps': 3, 'epss': False, 'sway': None, 'cfg': 1.4},
        {'method': 'midpoint', 'steps': 5, 'epss': True, 'sway': -1.0, 'cfg': 2.2},
    ]
    outputs = bytearray()
    for case in configs:
        times = make_timeline(case['steps'], case['epss'], case['sway'])
        state = initial.clone()
        for left, right in zip(times[:-1], times[1:]):
            delta = right - left
            v = velocity(state, condition, left, case['cfg'])
            if case['method'] == 'midpoint':
                mid = state + v * (delta * 0.5)
                v = velocity(mid, condition, left + delta * 0.5, case['cfg'])
            state = state + v * delta
        output = torch.where(mask, condition, state)
        case['offset'] = len(outputs) // 4
        case['count'] = output.numel()
        outputs.extend(output.contiguous().numpy().astype('<f4').tobytes())
    (ROOT / 'speech_edit_ode_pytorch_expected.json').write_text(
        json.dumps({'frames': frames, 'keep': keep, 'cases': configs}, separators=(',', ':'))+'\n'
    )
    (ROOT / 'speech_edit_ode_pytorch_expected.f32le').write_bytes(outputs)
    print(len(configs), 'numerical CFM trajectories,', len(outputs)//4, 'floats')


if __name__ == '__main__':
    generate()
