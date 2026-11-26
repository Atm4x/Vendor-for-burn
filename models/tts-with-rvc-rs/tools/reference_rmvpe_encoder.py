import argparse
import ast
import json
import os
import zipfile
from pathlib import Path

import torch
from torch import nn


def original_encoder_classes(source_zip: Path):
    with zipfile.ZipFile(source_zip) as archive:
        path = next(name for name in archive.namelist() if name.endswith('/infer/lib/rmvpe.py'))
        source = archive.read(path).decode('utf-8')
    tree = ast.parse(source, filename=path)
    names = {'ConvBlockRes', 'ResEncoderBlock', 'Encoder'}
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if len(definitions) != 3:
        raise ValueError('RMVPE original ConvBlockRes/ResEncoderBlock/Encoder not found')
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), path, 'exec'), namespace)
    return namespace['Encoder']


def normalize_zip(path: Path):
    replacement = path.with_suffix('.normalized.pth')
    with zipfile.ZipFile(path) as original, zipfile.ZipFile(replacement, 'w', compression=zipfile.ZIP_STORED) as target:
        for name in original.namelist():
            target.writestr(name, original.read(name))
    os.replace(replacement, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-zip', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--real-checkpoint', type=Path)
    group.add_argument('--synthetic-checkpoint', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--stages', type=int, default=2)
    args = parser.parse_args()
    if args.stages not in range(1, 6):
        raise ValueError('stages must be between 1 and 5')
    torch.set_num_threads(1)
    torch.manual_seed(20261109)
    Encoder = original_encoder_classes(args.source_zip)

    if args.real_checkpoint:
        state = torch.load(args.real_checkpoint, weights_only=True, map_location='cpu')
        model = Encoder(1, 128, args.stages, (2, 2), 4, 16)
        own = model.state_dict()
        translated = {key: state['unet.encoder.' + key].float() for key in own}
        model.load_state_dict(translated, strict=True)
    else:
        model = Encoder(1, 128, args.stages, (2, 2), 2, 4)
        prefix = 'unet.encoder.'
        state = {prefix + key: value.detach().half().cpu() for key, value in model.state_dict().items()}
        torch.save(state, args.synthetic_checkpoint)
        normalize_zip(args.synthetic_checkpoint)
        model.load_state_dict({key: state[prefix + key].float() for key in model.state_dict()}, strict=True)

    model.eval()
    height, width = (32, 64) if args.real_checkpoint else (8, 16)
    input_tensor = (torch.randn(1, 1, height, width, dtype=torch.float32) * 0.18).contiguous()
    with torch.no_grad():
        x = model.bn(input_tensor)
        steps = []
        for layer in model.layers:
            skip, x = layer(x)
            steps.append({
                'skip_shape': list(skip.shape), 'skip': skip.reshape(-1).tolist(),
                'pooled_shape': list(x.shape), 'pooled': x.reshape(-1).tolist(),
            })
    reference = {
        'input_shape': list(input_tensor.shape),
        'input': input_tensor.reshape(-1).tolist(),
        'stages': steps,
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(reference, separators=(',', ':')), encoding='utf-8')
    print(f'RMVPE original Encoder stage 0..{len(steps)-1}, output {steps[-1]["pooled_shape"]}, synthetic={args.synthetic_checkpoint is not None}')


if __name__ == '__main__':
    main()
