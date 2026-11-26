import argparse
import ast
import json
import os
import zipfile
from pathlib import Path

import torch
from torch import nn


def original_classes(source_zip: Path):
    with zipfile.ZipFile(source_zip) as archive:
        path = next(name for name in archive.namelist() if name.endswith('/infer/lib/rmvpe.py'))
        source = archive.read(path).decode('utf-8')
    names = {'BiGRU', 'ConvBlockRes', 'ResEncoderBlock', 'Encoder', 'Intermediate', 'ResDecoderBlock', 'Decoder', 'DeepUnet'}
    root = ast.parse(source, filename=path)
    classes = [node for node in root.body if isinstance(node, ast.ClassDef) and node.name in names]
    if len(classes) != len(names):
        raise ValueError('Incomplete original RMVPE source')
    env = {'torch': torch, 'nn': nn, 'List': list}
    exec(compile(ast.Module(body=classes, type_ignores=[]), path, 'exec'), env)
    return env['BiGRU'], env['DeepUnet']


def normalize_zip(path: Path):
    alternative = path.with_suffix('.normalized.pth')
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(alternative, 'w', compression=zipfile.ZIP_STORED) as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
    os.replace(alternative, path)


def f0_from_salience(values: torch.Tensor, threshold=0.03):
    cents = 1997.3794084376191 + 20.0 * torch.arange(360, dtype=torch.float64)
    output = []
    for row in values.reshape(-1, 360):
        index = row.argmax().item()
        if row[index].item() <= threshold:
            output.append(0.0)
            continue
        left = max(0, index - 4)
        right = min(359, index + 4)
        segment = row[left:right + 1].double()
        mean = (segment * cents[left:right + 1]).sum() / segment.sum()
        output.append(float(10.0 * 2.0 ** (mean.item() / 1200)))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-zip', required=True, type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=Path)
    group.add_argument('--synthetic-checkpoint', type=Path)
    p.add_argument('--fixture', required=True, type=Path)
    args = p.parse_args()
    torch.manual_seed(20261109)
    torch.set_num_threads(1)
    BiGRU, DeepUnet = original_classes(args.source_zip)
    real = args.checkpoint is not None
    bins, levels, middle, blocks, channels, frames = (128, 5, 4, 4, 16, 32) if real else (16, 2, 2, 2, 4, 8)
    unet = DeepUnet((2, 2), n_blocks=blocks, en_de_layers=levels, inter_layers=middle,
                    in_channels=1, en_out_channels=channels)
    cnn = nn.Conv2d(channels, 3, 3, padding=1)
    gru = BiGRU(3 * bins, 256, 1)
    head = nn.Linear(512, 360)
    if real:
        state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
        unet.load_state_dict({key: state['unet.' + key].float() for key in unet.state_dict()}, strict=True)
        cnn.load_state_dict({key: state['cnn.' + key].float() for key in cnn.state_dict()}, strict=True)
        gru.load_state_dict({key: state['fc.0.' + key].float() for key in gru.state_dict()}, strict=True)
        head.load_state_dict({key: state['fc.1.' + key].float() for key in head.state_dict()}, strict=True)
    else:
        state = {'unet.' + k: v.detach().half().cpu() for k, v in unet.state_dict().items()}
        state.update({'cnn.' + k: v.detach().half().cpu() for k, v in cnn.state_dict().items()})
        state.update({'fc.0.' + k: v.detach().half().cpu() for k, v in gru.state_dict().items()})
        state.update({'fc.1.' + k: v.detach().half().cpu() for k, v in head.state_dict().items()})
        args.synthetic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, args.synthetic_checkpoint)
        normalize_zip(args.synthetic_checkpoint)
        unet.load_state_dict({k: state['unet.' + k].float() for k in unet.state_dict()}, strict=True)
        cnn.load_state_dict({k: state['cnn.' + k].float() for k in cnn.state_dict()}, strict=True)
        gru.load_state_dict({k: state['fc.0.' + k].float() for k in gru.state_dict()}, strict=True)
        head.load_state_dict({k: state['fc.1.' + k].float() for k in head.state_dict()}, strict=True)
    for module in (unet, cnn, gru, head):
        module.eval()
    feature = torch.randn(1, 5, 3 * bins) * 0.15
    mel = torch.randn(1, bins, frames) * 0.15
    with torch.inference_mode():
        gru_out = gru(feature)
        gru_salience = torch.sigmoid(head(gru_out))
        cnn_features = cnn(unet(mel.transpose(-1, -2).unsqueeze(1))).transpose(1, 2).flatten(-2)
        integrated_gru = gru(cnn_features)
        integrated_salience = torch.sigmoid(head(integrated_gru))
    fixture = {
        'gru_input_shape': list(feature.shape),
        'gru_input': feature.flatten().tolist(),
        'gru_output_shape': list(gru_out.shape),
        'expected_gru': gru_out.flatten().tolist(),
        'expected_salience': gru_salience.flatten().tolist(),
        'expected_f0': f0_from_salience(gru_salience),
        'mel_shape': list(mel.shape),
        'mel': mel.flatten().tolist(),
        'unet_levels': levels,
        'middle_levels': middle,
        'expected_features_shape': list(cnn_features.shape),
        'expected_features': cnn_features.flatten().tolist(),
        'expected_full': integrated_salience.flatten().tolist(),
        'expected_full_shape': list(integrated_salience.shape),
        'expected_full_f0': f0_from_salience(integrated_salience),
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    print('Original RMVPE BiGRU', list(gru_out.shape), 'E2E', list(integrated_salience.shape))


if __name__ == '__main__':
    main()
