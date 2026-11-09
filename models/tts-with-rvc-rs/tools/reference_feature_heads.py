import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def read_hubert(path: Path):
    class DictionaryStub:
        pass

    with torch.serialization.safe_globals([(DictionaryStub, 'fairseq.data.dictionary.Dictionary')]):
        state = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(state.get('model'), dict):
        raise ValueError('HuBERT model checkpoint has no model state')
    return state['model']


def normalize_zip(path: Path):
    repacked = path.with_suffix('.repacked.pt')
    with zipfile.ZipFile(path) as original, zipfile.ZipFile(repacked, 'w', compression=zipfile.ZIP_STORED) as normalized:
        for name in original.namelist():
            normalized.writestr(name, original.read(name))
    os.replace(repacked, path)


def reference_hubert(weights, audio):
    weight = weights['feature_extractor.conv_layers.0.0.weight'].float()
    norm_w = weights['feature_extractor.conv_layers.0.2.weight'].float()
    norm_b = weights['feature_extractor.conv_layers.0.2.bias'].float()
    x = F.conv1d(audio, weight, stride=5)
    x = F.group_norm(x, num_groups=512, weight=norm_w, bias=norm_b, eps=1e-5)
    return F.gelu(x)


def reference_rmvpe(weights, features):
    weight = weights['fc.1.weight'].float()
    bias = weights['fc.1.bias'].float()
    salience = (features @ weight.T + bias).sigmoid()
    frames = salience.numpy().reshape(-1, 360)
    mapping = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))
    padded = np.pad(frames, ((0, 0), (4, 4)))
    centers = frames.argmax(axis=1) + 4
    indices = centers[:, None] + np.arange(-4, 5)[None, :]
    weights_near = np.take_along_axis(padded, indices, axis=1)
    cents = (weights_near * mapping[indices]).sum(axis=1) / weights_near.sum(axis=1)
    cents[frames.max(axis=1) <= 0.03] = 0
    f0 = 10 * (2 ** (cents / 1200))
    f0[f0 == 10] = 0
    return salience, f0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['hubert', 'rmvpe'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--synthetic-checkpoint', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.synthetic_checkpoint):
        parser.error('provide exactly one of --checkpoint or --synthetic-checkpoint')
    torch.set_num_threads(1)
    torch.manual_seed(8675309)
    torch.serialization.DEFAULT_PROTOCOL = 2

    if args.model == 'hubert':
        if args.checkpoint:
            weights = read_hubert(args.checkpoint)
        else:
            weights = {
                'feature_extractor.conv_layers.0.0.weight': (torch.randn(512, 1, 10) * 0.04).half(),
                'feature_extractor.conv_layers.0.2.weight': (1 + torch.randn(512) * 0.1).half(),
                'feature_extractor.conv_layers.0.2.bias': (torch.randn(512) * 0.02).half(),
            }
        audio = (torch.randn(1, 1, 75) * 0.1).float()
        with torch.inference_mode():
            output = reference_hubert(weights, audio)
        fixture = {'shape': list(audio.shape), 'waveform': audio.reshape(-1).tolist(),
                   'expected_shape': list(output.shape), 'expected': output.reshape(-1).tolist()}
        if args.synthetic_checkpoint:
            torch.save({'model':weights},args.synthetic_checkpoint)
            normalize_zip(args.synthetic_checkpoint)
    else:
        if args.checkpoint:
            weights = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
        else:
            weights = {'fc.1.weight': (torch.randn(360, 512) * 0.025).half(),
                       'fc.1.bias': (torch.randn(360) * 0.01).half()}
        features = (torch.randn(1, 3, 512) * 0.2).float()
        with torch.inference_mode():
            salience, f0 = reference_rmvpe(weights, features)
        fixture = {'shape': list(features.shape), 'features': features.reshape(-1).tolist(),
                   'expected_salience': salience.reshape(-1).tolist(), 'expected_f0': f0.tolist()}
        if args.synthetic_checkpoint:
            torch.save(weights, args.synthetic_checkpoint)
            normalize_zip(args.synthetic_checkpoint)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    print(args.model, 'fixture', args.fixture, 'private', bool(args.checkpoint))


if __name__ == '__main__':
    main()
