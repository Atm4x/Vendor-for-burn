import argparse
import ast
import json
import os
import zipfile
from pathlib import Path

import torch
from torch import nn


def original_intermediate(source_zip: Path):
    with zipfile.ZipFile(source_zip) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/infer/lib/rmvpe.py'))
        code = ast.parse(archive.read(name).decode('utf-8'), filename=name)
    names = {'ConvBlockRes', 'ResEncoderBlock', 'Intermediate'}
    classes = [node for node in code.body if isinstance(node, ast.ClassDef) and node.name in names]
    if len(classes) != 3:
        raise ValueError('RMVPE source missing Intermediate classes')
    environment = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=classes, type_ignores=[]), name, 'exec'), environment)
    return environment['Intermediate']


def normalize_zip(path: Path):
    target_path = path.with_suffix('.normalized.pth')
    with zipfile.ZipFile(path) as inp, zipfile.ZipFile(target_path, 'w', compression=zipfile.ZIP_STORED) as out:
        for name in inp.namelist():
            out.writestr(name, inp.read(name))
    os.replace(target_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-zip', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=Path)
    group.add_argument('--synthetic-checkpoint', type=Path)
    parser.add_argument('--encoder-reference', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261109)
    torch.set_num_threads(1)
    Intermediate = original_intermediate(args.source_zip)

    if args.checkpoint:
        if args.encoder_reference is None:
            raise ValueError('--encoder-reference is required for a real RMVPE checkpoint')
        source = json.loads(args.encoder_reference.read_text(encoding='utf-8'))
        previous = source['stages'][-1]
        input_tensor = torch.tensor(previous['pooled'], dtype=torch.float32).reshape(previous['pooled_shape'])
        model = Intermediate(256, 512, 4, 4)
        checkpoint = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
        weights = {name: checkpoint['unet.intermediate.' + name].float() for name in model.state_dict()}
        model.load_state_dict(weights, strict=True)
    else:
        input_tensor = torch.randn(1, 4, 4, 8, dtype=torch.float32) * 0.15
        model = Intermediate(4, 8, 2, 2)
        state = {f'unet.intermediate.{key}': value.detach().half().cpu() for key, value in model.state_dict().items()}
        torch.save(state, args.synthetic_checkpoint)
        normalize_zip(args.synthetic_checkpoint)
        model.load_state_dict({name: state['unet.intermediate.' + name].float() for name in model.state_dict()}, strict=True)

    model.eval()
    with torch.no_grad():
        current = input_tensor
        expected = []
        for layer in model.layers:
            current = layer(current)
            expected.append({'shape': list(current.shape), 'output': current.reshape(-1).tolist()})
    fixture = {'input_shape': list(input_tensor.shape), 'input': input_tensor.reshape(-1).tolist(), 'levels': expected}
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    print('RMVPE original Intermediate levels',len(expected),'output',expected[-1]['shape'])


if __name__ == '__main__':
    main()
