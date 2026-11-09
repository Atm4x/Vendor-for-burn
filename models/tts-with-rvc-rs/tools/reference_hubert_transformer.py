import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from reference_feature_heads import normalize_zip, read_hubert


def reference_layer(weights, x, valid, heads):
    root = 'encoder.layers.0.'

    def linear(name, value):
        return F.linear(value, weights[root + name + '.weight'].float(),
                        weights[root + name + '.bias'].float())

    def norm(name, value):
        return F.layer_norm(value, (value.shape[-1],),
                            weights[root + name + '.weight'].float(),
                            weights[root + name + '.bias'].float(), eps=1e-5)

    batch, time, channels = x.shape
    depth = channels // heads
    def split(y):
        return y.reshape(batch, time, heads, depth).transpose(1, 2)

    q = split(linear('self_attn.q_proj', x)) * (depth ** -0.5)
    k = split(linear('self_attn.k_proj', x))
    v = split(linear('self_attn.v_proj', x))
    scores = q @ k.transpose(-1, -2)
    if valid is not None:
        scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
    weights_attn = torch.softmax(scores, dim=-1)
    attended = (weights_attn @ v).transpose(1, 2).reshape(batch, time, channels)
    x = norm('self_attn_layer_norm', x + linear('self_attn.out_proj', attended))
    hidden = F.gelu(linear('fc1', x))
    return norm('final_layer_norm', x + linear('fc2', hidden))


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=Path)
    group.add_argument('--synthetic-checkpoint', type=Path)
    p.add_argument('--fixture', type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(20261109)
    torch.set_num_threads(1)
    if a.checkpoint:
        state = read_hubert(a.checkpoint)
        channels, heads, intermediate, batch, time = 768, 12, 3072, 1, 7
        weights = {k:v for k,v in state.items() if k.startswith('encoder.layers.0.')}
    else:
        channels, heads, intermediate, batch, time = 8, 2, 16, 2, 6
        root = 'encoder.layers.0.'
        weights = {}
        for key in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj']:
            weights[root + key + '.weight'] = (torch.randn(channels, channels) * 0.06).half()
            weights[root + key + '.bias'] = (torch.randn(channels) * 0.01).half()
        for key, output, input in [('fc1', intermediate, channels), ('fc2', channels, intermediate)]:
            weights[root + key + '.weight'] = (torch.randn(output, input) * 0.06).half()
            weights[root + key + '.bias'] = (torch.randn(output) * 0.01).half()
        for key in ['self_attn_layer_norm', 'final_layer_norm']:
            weights[root + key + '.weight'] = (1.0 + torch.randn(channels) * 0.03).half()
            weights[root + key + '.bias'] = (torch.randn(channels) * 0.02).half()
    x = torch.randn(batch, time, channels) * 0.35
    valid = torch.ones((batch, time), dtype=torch.bool)
    valid[-1, -2:] = False
    with torch.inference_mode():
        masked = reference_layer(weights, x, valid, heads)
        unmasked = reference_layer(weights, x, None, heads)
    fixture = {
        'shape': list(x.shape), 'heads': heads, 'input': x.flatten().tolist(),
        'valid': valid.float().flatten().tolist(),
        'expected': masked.flatten().tolist(),
        'expected_unmasked': unmasked.flatten().tolist(),
    }
    a.fixture.parent.mkdir(parents=True, exist_ok=True)
    a.fixture.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    if a.synthetic_checkpoint:
        torch.save({'model': weights}, a.synthetic_checkpoint)
        normalize_zip(a.synthetic_checkpoint)
    print(f'HuBERT Transformer block 0, channels={channels}, heads={heads}, output={tuple(masked.shape)}')


if __name__ == '__main__':
    main()
