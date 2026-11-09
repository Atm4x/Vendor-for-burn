import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from reference_feature_heads import read_hubert, normalize_zip


def run_frontend(weights, waveform):
    first = F.conv1d(waveform, weights['feature_extractor.conv_layers.0.0.weight'].float(), stride=5)
    channels = first.shape[1]
    first = F.group_norm(first, channels,
                         weights['feature_extractor.conv_layers.0.2.weight'].float(),
                         weights['feature_extractor.conv_layers.0.2.bias'].float(), eps=1e-5)
    features = F.gelu(first)
    for i in range(1, 7):
        features = F.gelu(F.conv1d(features, weights[f'feature_extractor.conv_layers.{i}.0.weight'].float(), stride=2))
    features = features.transpose(1, 2)
    features = F.layer_norm(features, (channels,),
                            weights['layer_norm.weight'].float(),
                            weights['layer_norm.bias'].float(), eps=1e-5)
    return F.linear(features, weights['post_extract_proj.weight'].float(),
                    weights['post_extract_proj.bias'].float())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--synthetic-checkpoint', type=Path)
    p.add_argument('--fixture', type=Path, required=True)
    a = p.parse_args()
    if bool(a.checkpoint) == bool(a.synthetic_checkpoint):
        p.error('provide exactly one checkpoint option')
    torch.manual_seed(20261109)
    torch.set_num_threads(1)
    if a.checkpoint:
        weights = read_hubert(a.checkpoint)
        channels = 512
    else:
        channels = 8
        weights = {
            'feature_extractor.conv_layers.0.0.weight': (torch.randn(channels, 1, 10) * 0.05).half(),
            'feature_extractor.conv_layers.0.2.weight': (1 + torch.randn(channels) * 0.03).half(),
            'feature_extractor.conv_layers.0.2.bias': (torch.randn(channels) * 0.03).half(),
        }
        for i in range(1, 7):
            kernel = 3 if i <= 4 else 2
            weights[f'feature_extractor.conv_layers.{i}.0.weight'] = (torch.randn(channels, channels, kernel) * 0.03).half()
        weights['layer_norm.weight'] = (1 + torch.randn(channels) * 0.03).half()
        weights['layer_norm.bias'] = (torch.randn(channels) * 0.01).half()
        weights['post_extract_proj.weight'] = (torch.randn(12, channels) * 0.04).half()
        weights['post_extract_proj.bias'] = (torch.randn(12) * 0.01).half()
    waveform = (torch.randn(1, 1, 1024) * 0.1).float()
    with torch.inference_mode():
        output = run_frontend(weights, waveform)
    fixture = {'shape': list(waveform.shape), 'waveform': waveform.flatten().tolist(),
               'expected_shape': list(output.shape), 'expected': output.flatten().tolist()}
    a.fixture.parent.mkdir(parents=True, exist_ok=True)
    a.fixture.write_text(json.dumps(fixture, separators=(',',':')), encoding='utf-8')
    if a.synthetic_checkpoint:
        torch.save({'model': weights}, a.synthetic_checkpoint)
        normalize_zip(a.synthetic_checkpoint)
    print('channels',channels,'projected_shape',tuple(output.shape),'fixture',str(a.fixture))


if __name__ == '__main__':
    main()
