import argparse
import json
from pathlib import Path

import torch

from reference_feature_heads import normalize_zip, read_hubert
from reference_hubert_transformer import reference_layer


def make_layers(count, channels, inner):
    state = {}
    for index in range(count):
        prefix = f'encoder.layers.{index}.'
        for name in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj'):
            state[prefix + name + '.weight'] = (torch.randn(channels, channels) * 0.035).half()
            state[prefix + name + '.bias'] = (torch.randn(channels) * 0.01).half()
        for name, shape in [('fc1', (inner, channels)), ('fc2', (channels, inner))]:
            state[prefix + name + '.weight'] = (torch.randn(*shape) * 0.035).half()
            state[prefix + name + '.bias'] = (torch.randn(shape[0]) * 0.01).half()
        for name in ('self_attn_layer_norm', 'final_layer_norm'):
            state[prefix + name + '.weight'] = (1.0 + torch.randn(channels) * 0.015).half()
            state[prefix + name + '.bias'] = (torch.randn(channels) * 0.01).half()
    return state


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--checkpoint', type=Path)
    modes.add_argument('--synthetic-checkpoint', type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    args = parser.parse_args()
    torch.manual_seed(20261109)
    torch.set_num_threads(1)
    if args.checkpoint:
        state = read_hubert(args.checkpoint)
        layers, channels, heads, batch, time = 12, 768, 12, 1, 7
    else:
        layers, channels, heads, batch, time = 3, 8, 2, 2, 6
        state = make_layers(layers, channels, channels * 3)
    x = torch.randn(batch, time, channels) * 0.3
    valid = torch.ones((batch, time), dtype=torch.bool)
    valid[-1, -2:] = False
    with torch.inference_mode():
        out = x
        for index in range(layers):
            out = reference_layer(state, out, valid, heads, index=index)
        unmasked = x
        for index in range(layers):
            unmasked = reference_layer(state, unmasked, None, heads, index=index)
    fixture = {
        'shape': list(x.shape), 'heads': heads, 'layers': layers,
        'input': x.flatten().tolist(), 'valid': valid.float().flatten().tolist(),
        'expected': out.flatten().tolist(), 'expected_unmasked': unmasked.flatten().tolist(),
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    if args.synthetic_checkpoint:
        torch.save({'model': state}, args.synthetic_checkpoint)
        normalize_zip(args.synthetic_checkpoint)
    print('HuBERT Transformer stack', layers, 'layers', tuple(out.shape))


if __name__ == '__main__':
    main()
