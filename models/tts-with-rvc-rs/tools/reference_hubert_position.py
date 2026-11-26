import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from reference_feature_heads import normalize_zip, read_hubert
from reference_hubert_stack import make_layers
from reference_hubert_transformer import reference_layer


def positional_conv_reference(state, x, valid, groups):
    root = 'encoder.pos_conv.0.'
    kernel = state[root + 'weight_v'].shape[2]
    channels = x.shape[2]
    conv = torch.nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=groups)
    conv = torch.nn.utils.weight_norm(conv, name='weight', dim=2)
    with torch.no_grad():
        conv.weight_g.copy_(state[root + 'weight_g'].float())
        conv.weight_v.copy_(state[root + 'weight_v'].float())
        conv.bias.copy_(state[root + 'bias'].float())
    conv.eval()
    if valid is not None:
        x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
    conv_out = conv(x.transpose(1, 2))
    if kernel % 2 == 0:
        conv_out = conv_out[:, :, :-1]
    x = x + F.gelu(conv_out.transpose(1, 2))
    return F.layer_norm(x, (channels,),
                        state['encoder.layer_norm.weight'].float(),
                        state['encoder.layer_norm.bias'].float(), eps=1e-5)


def make_position(channels, kernel, groups):
    return {
        'encoder.pos_conv.0.weight_v': (torch.randn(channels, channels // groups, kernel) * 0.035).half(),
        'encoder.pos_conv.0.weight_g': (torch.randn(1, 1, kernel) * 0.02 + 0.65).half(),
        'encoder.pos_conv.0.bias': (torch.randn(channels) * 0.015).half(),
        'encoder.layer_norm.weight': (torch.randn(channels) * 0.015 + 1).half(),
        'encoder.layer_norm.bias': (torch.randn(channels) * 0.015).half(),
    }


def main():
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--checkpoint', type=Path)
    sources.add_argument('--synthetic-checkpoint', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20261113)
    torch.set_num_threads(1)
    if args.checkpoint:
        state = read_hubert(args.checkpoint)
        channels, heads, layers, kernel, groups, batch, time = 768, 12, 12, 128, 16, 1, 7
    else:
        channels, heads, layers, kernel, groups, batch, time = 8, 2, 3, 8, 2, 2, 6
        state = {**make_layers(layers, channels, channels * 3), **make_position(channels, kernel, groups)}

    x = torch.randn(batch, time, channels) * 0.3
    valid = torch.ones(batch, time, dtype=torch.bool)
    valid[-1, -2:] = False

    with torch.inference_mode():
        position = positional_conv_reference(state, x, valid, groups)
        position_unmasked = positional_conv_reference(state, x, None, groups)
        masked = position
        unmasked = position_unmasked
        for index in range(layers):
            masked = reference_layer(state, masked, valid, heads, index)
            unmasked = reference_layer(state, unmasked, None, heads, index)

    fixture = {
        'shape': list(x.shape), 'heads': heads, 'layers': layers,
        'input': x.flatten().tolist(), 'valid': valid.float().flatten().tolist(),
        'expected_position': position.flatten().tolist(),
        'expected_position_unmasked': position_unmasked.flatten().tolist(),
        'expected': masked.flatten().tolist(),
        'expected_unmasked': unmasked.flatten().tolist(),
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    if args.synthetic_checkpoint:
        torch.save({'model': state}, args.synthetic_checkpoint)
        normalize_zip(args.synthetic_checkpoint)
    print('Hubert position and complete encoder:', 'layers=', layers, 'shape=', list(x.shape))


if __name__ == '__main__':
    main()
