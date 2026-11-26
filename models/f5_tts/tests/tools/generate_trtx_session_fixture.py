import json
from pathlib import Path

import torch


def main():
    torch.set_num_threads(1)
    times = torch.tensor([0.0, 0.0625, 0.125, 0.25, 0.5, 1.0], dtype=torch.float32)
    frequencies = torch.exp(torch.arange(128, dtype=torch.float32) * (-(torch.log(torch.tensor(10000.0)) / 127.0)))
    embeddings = []
    for t in times[:-1]:
        angles = 1000.0 * t * frequencies
        embeddings.append(torch.cat([angles.sin(), angles.cos()]).tolist())
    reference = torch.full((2, 100), 0.75, dtype=torch.float32)
    state = torch.full((6, 100), 0.25, dtype=torch.float32)
    conditional = torch.tensor(0.1, dtype=torch.float32)
    unconditional = torch.tensor(-0.05, dtype=torch.float32)
    cfg = torch.tensor(2.0, dtype=torch.float32)
    velocity = conditional + (conditional - unconditional) * cfg
    trace = []
    for i, (start, stop) in enumerate(zip(times[:-1], times[1:])):
        state = state + velocity * (stop - start)
        if i == len(times) - 2:
            state[:2] = reference
        trace.append({'first_reference': float(state[0, 0]), 'first_generated': float(state[2, 0])})
    result = {
        'timeline': times.tolist(),
        'frequency_embeddings': embeddings,
        'steps': trace,
        'generated_mel': state[2:].flatten().tolist(),
    }
    target = Path(__file__).resolve().parent.parent / 'fixtures' / 'trtx_session_pytorch_expected.json'
    target.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    print(target, 'generated', len(result['generated_mel']), 'float values')


if __name__ == '__main__':
    main()
