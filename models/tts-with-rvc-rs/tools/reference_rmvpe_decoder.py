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
    is_real = args.checkpoint is not None
    model = DeepUnet((2, 2), n_blocks=4 if is_real else 2, en_de_layers=5 if is_real else 2,
                     inter_layers=4 if is_real else 2, in_channels=1,
                     en_out_channels=16 if is_real else 4)
    if is_real:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        weights = {name: state['unet.' + name].float() for name in model.state_dict()}
        model.load_state_dict(weights, strict=True)
    else:
        state = {'unet.' + name: value.detach().half().cpu() for name, value in model.state_dict().items()}
        args.synthetic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, args.synthetic_checkpoint)
        normalize_zip(args.synthetic_checkpoint)
        model.load_state_dict({name: state['unet.' + name].float() for name in model.state_dict()}, strict=True)
    model.eval()
    height, width = (32, 64) if is_real else (8, 16)
    x = torch.randn(1, 1, height, width, dtype=torch.float32) * 0.18
    with torch.inference_mode():
        x = model.encoder.bn(x)
        skips = []
        for layer in model.encoder.layers:
            skip, x = layer(x)
            skips.append(skip)
        x = model.intermediate(x)
        decoder_input = x
        outputs = []
        for level, layer in enumerate(model.decoder.layers):
            x = layer(x, skips[-level - 1])
            outputs.append({'shape': list(x.shape), 'output': x.flatten().tolist()})
    fixture = {
        'input_shape': list(decoder_input.shape),
        'input': decoder_input.flatten().tolist(),
        'skips': [{'shape': list(skip.shape), 'values': skip.flatten().tolist()} for skip in skips],
        'levels': outputs,
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    print('RMVPE original decoder', len(outputs), 'levels, final', outputs[-1]['shape'])


if __name__ == '__main__':
    main()
