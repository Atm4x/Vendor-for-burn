import argparse
import ast
import json
import math
import zipfile
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def original_classes(source_zip: Path):
    with zipfile.ZipFile(source_zip) as archive:
        name = next(path for path in archive.namelist() if path.endswith('/infer/lib/rmvpe.py'))
        source = archive.read(name).decode('utf-8')
    needed = {'MelSpectrogram', 'BiGRU', 'ConvBlockRes', 'ResEncoderBlock', 'Encoder',
              'Intermediate', 'ResDecoderBlock', 'Decoder', 'DeepUnet', 'E2E'}
    tree = ast.parse(source, filename=name)
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in needed]
    if len(definitions) != len(needed):
        raise ValueError('Original RMVPE source incomplete')
    env = {'torch': torch, 'nn': nn, 'F': F, 'np': np, 'List': list,
           'mel': librosa.filters.mel, 'os': __import__('os')}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), name, 'exec'), env)
    return env['MelSpectrogram'], env['E2E']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-zip', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    torch.manual_seed(20261109)
    torch.set_num_threads(1)
    MelSpectrogram, E2E = original_classes(args.source_zip)
    samples = 4096
    t = np.arange(samples, dtype=np.float64) / 16000.0
    audio = (0.16 * np.sin(2 * math.pi * 230 * t) +
             0.06 * np.sin(2 * math.pi * (410 * t + 180 * t ** 2)) +
             0.035 * np.cos(2 * math.pi * 720 * t) +
             0.012 * np.random.default_rng(20261109).standard_normal(samples)).astype(np.float32)
    extractor = MelSpectrogram(False, 128, 16000, 1024, 160, None, 30, 8000)
    with torch.inference_mode():
        mel = extractor(torch.from_numpy(audio).unsqueeze(0), center=True)
    fixture = {
        'sample_rate': 16000, 'audio': audio.tolist(),
        'mel_shape': list(mel.shape), 'expected_mel': mel.flatten().tolist(),
    }
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model = E2E(4, 1, (2, 2))
        model.load_state_dict({k: state[k].float() for k in model.state_dict()}, strict=True)
        model.eval()
        n_frames = mel.shape[-1]
        padded = F.pad(mel, (0, (-n_frames) % 32))
        with torch.inference_mode():
            salience = model(padded)[:, :n_frames, :]
        fixture['expected_salience'] = salience.flatten().tolist()
        fixture['expected_salience_shape'] = list(salience.shape)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    print('Original Mel',list(mel.shape),'full',fixture.get('expected_salience_shape'))


if __name__ == '__main__':
    main()
