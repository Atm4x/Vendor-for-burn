import argparse
import ast
import json
import os
import zipfile
from pathlib import Path

import torch
from torch import nn


def original_unet(source_zip: Path):
    with zipfile.ZipFile(source_zip) as archive:
        path = next(name for name in archive.namelist() if name.endswith('/infer/lib/rmvpe.py'))
        source = archive.read(path).decode('utf-8')
    root = ast.parse(source, filename=path)
    names = {'ConvBlockRes', 'ResEncoderBlock', 'Encoder', 'Intermediate', 'ResDecoderBlock', 'Decoder', 'DeepUnet'}
    definitions = [node for node in root.body if isinstance(node, ast.ClassDef) and node.name in names]
    if len(definitions) != len(names):
        raise ValueError('RMVPE original U-Net source classes are incomplete')
    env = {'torch': torch, 'nn': nn, 'List': list}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), path, 'exec'), env)
    return env['DeepUnet']


def normalize_zip(path: Path):
    alternative = path.with_suffix('.normalized.pth')
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(alternative, 'w', compression=zipfile.ZIP_STORED) as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
    os.replace(alternative, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-zip', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=Path)
    group.add_argument('--synthetic-checkpoint', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261109)
    torch.set_num_threads(1)
    DeepUnet = original_unet(args.source_zip)
    real = args.checkpoint is not None
    levels, middle, blocks, channels = (5, 4, 4, 16) if real else (2, 2, 2, 4)
    model = DeepUnet((2, 2), n_blocks=blocks, en_de_layers=levels,
                     inter_layers=middle, in_channels=1, en_out_channels=channels)
    cnn = nn.Conv2d(channels, 3, (3, 3), padding=(1, 1))
    if real:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict({k: state['unet.' + k].float() for k in model.state_dict()}, strict=True)
        cnn.load_state_dict({k: state['cnn.' + k].float() for k in cnn.state_dict()}, strict=True)
    else:
        state = {'unet.' + k: v.detach().half().cpu() for k, v in model.state_dict().items()}
        state.update({'cnn.' + k: v.detach().half().cpu() for k, v in cnn.state_dict().items()})
        args.synthetic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, args.synthetic_checkpoint)
        normalize_zip(args.synthetic_checkpoint)
        model.load_state_dict({k: state['unet.' + k].float() for k in model.state_dict()}, strict=True)
        cnn.load_state_dict({k: state['cnn.' + k].float() for k in cnn.state_dict()}, strict=True)
    model.eval()
    cnn.eval()
    h, w = (32, 64) if real else (8, 16)
    data = torch.randn(1, 1, h, w, dtype=torch.float32) * 0.18
    with torch.inference_mode():
        output = cnn(model(data))
        features = output.transpose(1, 2).flatten(-2)
    fixture = {
        'input_shape': list(data.shape), 'input': data.flatten().tolist(),
        'output_shape': list(output.shape), 'expected': output.flatten().tolist(),
        'feature_shape': list(features.shape), 'expected_features': features.flatten().tolist(),
        'encoder_levels': levels, 'intermediate_levels': middle,
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    print('RMVPE original full U-Net + CNN', list(output.shape), 'features', list(features.shape))


if __name__ == '__main__':
    main()
