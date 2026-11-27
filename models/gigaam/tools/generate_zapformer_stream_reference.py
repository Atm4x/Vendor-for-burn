"""Generate packetwise reference using the original streaming encoder and existing weights."""
import argparse
import json
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[1] / 'tests/fixtures/zapformer')
    args = parser.parse_args()
    package = types.ModuleType('gigaam')
    package.__path__ = [str(args.upstream.resolve() / 'gigaam')]
    sys.modules['gigaam'] = package
    from gigaam.encoder import ZapformerEncoder

    torch.set_num_threads(1)
    config = json.loads((args.fixtures / 'cases.json').read_text())[0]['config']
    model = ZapformerEncoder(**{k: v for k, v in config.items() if k != '_target_'}).eval()
    source = load_file(args.fixtures / 'causal_swish.safetensors')
    model.load_state_dict({k[len('encoder.'):]: v for k, v in source.items() if k.startswith('encoder.')}, strict=False)
    # Reuse deterministic input already present in the existing fixture; no new weights.
    x = source['reference.input'][:1].repeat(1, 5, 1)[:, :145].transpose(1, 2)
    packets = [5, 11, 18, 2, 9, 23, 7, 33, 37]
    assert sum(packets) == x.shape[-1]
    cases = []
    for chunk in (4, 8, 16):
        for left in (0, 8):
            state = model.create_stream_state(chunk, left)
            counts, output = [], []
            position = 0
            for count in packets + [0]:
                encoded, lengths, state = model(x[..., position:position+count], torch.tensor([count]), state=state, final=count == 0)
                position += count
                counts.append(int(lengths[0]))
                output.extend(encoded.transpose(1, 2).flatten().tolist())
            assert state.position == 37
            cases.append(dict(chunk=chunk, left=left, counts=counts, output=output))
    (args.fixtures / 'stream_reference.json').write_text(json.dumps(dict(packets=packets, cases=cases), separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
