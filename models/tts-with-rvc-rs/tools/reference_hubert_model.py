import argparse
import json
from pathlib import Path

import torch

from reference_feature_heads import read_hubert, normalize_zip
from reference_hubert_frontend import run_frontend
from reference_hubert_stack import make_layers
from reference_hubert_position import make_position, positional_conv_reference
from reference_hubert_transformer import reference_layer


def synthetic_frontend(channels, embed):
    state = {
        'feature_extractor.conv_layers.0.0.weight': (torch.randn(channels, 1, 10) * 0.04).half(),
        'feature_extractor.conv_layers.0.2.weight': (1 + torch.randn(channels) * 0.02).half(),
        'feature_extractor.conv_layers.0.2.bias': (torch.randn(channels) * 0.01).half(),
        'layer_norm.weight': (1 + torch.randn(channels) * 0.02).half(),
        'layer_norm.bias': (torch.randn(channels) * 0.01).half(),
        'post_extract_proj.weight': (torch.randn(embed, channels) * 0.04).half(),
        'post_extract_proj.bias': (torch.randn(embed) * 0.01).half(),
    }
    for layer in range(1, 7):
        k = 3 if layer <= 4 else 2
        state[f'feature_extractor.conv_layers.{layer}.0.weight'] = (torch.randn(channels, channels, k) * 0.035).half()
    return state


def main():
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--checkpoint', type=Path)
    options.add_argument('--synthetic-checkpoint', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261114)
    torch.set_num_threads(1)
    if args.checkpoint:
        weights = read_hubert(args.checkpoint)
        layers, heads, groups, kernel, batch, frames = 12, 12, 16, 128, 1, 2560
    else:
        channels, embed, layers, heads, groups, kernel, batch, frames = 8, 8, 3, 2, 2, 8, 2, 2080
        weights = {**synthetic_frontend(channels, embed),
                   **make_position(embed, kernel, groups),
                   **make_layers(layers, embed, embed * 3)}

    waveform = torch.randn(batch, 1, frames) * 0.08
    with torch.inference_mode():
        projected = run_frontend(weights, waveform)
        nframe = projected.shape[1]
        valid = torch.ones(batch, nframe, dtype=torch.bool)
        valid[-1, -1:] = False
        x = positional_conv_reference(weights, projected, valid, groups)
        for index in range(layers):
            x = reference_layer(weights, x, valid, heads, index)
        unmasked = positional_conv_reference(weights, projected, None, groups)
        for index in range(layers):
            unmasked = reference_layer(weights, unmasked, None, heads, index)
    fixture = {'input_shape': list(waveform.shape), 'shape': list(x.shape),
               'heads': heads, 'layers': layers, 'waveform': waveform.flatten().tolist(),
               'valid': valid.float().flatten().tolist(),
               'expected': x.flatten().tolist(), 'expected_unmasked': unmasked.flatten().tolist()}
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    if args.synthetic_checkpoint:
        torch.save({'model': weights}, args.synthetic_checkpoint)
        normalize_zip(args.synthetic_checkpoint)
    print('HuBERT waveform to content features', 'layers', layers, 'output_shape', list(x.shape))


if __name__ == '__main__':
    main()
