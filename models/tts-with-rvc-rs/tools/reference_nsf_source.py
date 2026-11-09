import argparse
import json
import math
import os
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_STORED, ZipFile

import torch

from reference_flow import import_original


def flatten(x):
    return x.detach().float().contiguous().cpu().flatten().tolist()


def normalize_pytorch_zip(path):
    temporary = path.with_suffix('.tmp')
    with ZipFile(path, 'r') as source, ZipFile(temporary, 'w', compression=ZIP_STORED) as dest:
        for name in source.namelist():
            dest.writestr(name, source.read(name))
    os.replace(temporary, path)


def controlled_random(phase, noise):
    return (patch('torch.rand', side_effect=lambda *a, **kw: phase.clone()),
            patch('torch.randn_like', side_effect=lambda *a, **kw: noise.clone()))


def source_reference(models, checkpoint, harmonics, output):
    sr, up = 40000, 4
    if harmonics != 0:
        raise ValueError('The original fork has an inplace broadcasting bug for harmonic_num>0')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)['weight']
    model = models.SourceModuleHnNSF(sr, 0, is_half=False).eval().float()
    model.load_state_dict({key.removeprefix('dec.m_source.'): value.float() for key, value in saved.items() if key.startswith('dec.m_source.')})
    f0 = torch.tensor([[0., 110., 187.5, 199.4, 0., 130.2]], dtype=torch.float32)
    dim = harmonics + 1
    phases = torch.rand(1, 1, dim)
    phases[..., 0] = 0.0
    normal = torch.randn(1, f0.shape[1] * up, dim)
    patches = controlled_random(phases, normal)
    with patches[0], patches[1], torch.inference_mode():
        merged, _, _ = model(f0, up)
    patches = controlled_random(phases, normal)
    with patches[0], patches[1], torch.inference_mode():
        sines, voiced, actual_noise = model.l_sin_gen(f0, up)
    obj = {
        'sample_rate': sr,
        'harmonic_num': harmonics,
        'upsample': up,
        'f0_shape': list(f0.shape),
        'f0': flatten(f0),
        'initial_phases': flatten(phases),
        'gaussian_noise': flatten(normal),
        'expected_sines': flatten(sines),
        'expected_voiced': flatten(voiced),
        'expected_noise': flatten(actual_noise),
        'expected_merged': flatten(merged),
    }
    output.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    return obj


def waveform_reference(models, checkpoint, output, real_checkpoint=None):
    from tts_with_rvc.infer.lib.infer_pack.models import GeneratorNSF

    if real_checkpoint is None:
        cfg = dict(rates=[2, 2], kernels=[3, 7], dilation=[[1, 3, 5]] * 2, channels=4, initial=8, gin=3, sr=40000, kernels_up=[4, 4])
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)['weight']
        frames = 6
        g = torch.randn(1, cfg['gin'], 1) * .13
    else:
        checkpoint_data = torch.load(real_checkpoint, map_location='cpu', weights_only=True)
        c = checkpoint_data['config']
        cfg = dict(rates=c[12], kernels=c[10], dilation=c[11], channels=c[2], initial=c[13], gin=c[16], sr=c[17], kernels_up=c[14])
        ckpt = checkpoint_data['weight']
        frames = 2
        g = ckpt['emb_g.weight'][0].float().reshape(1, cfg['gin'], 1)
    generator = GeneratorNSF(cfg['channels'], '1', cfg['kernels'], cfg['dilation'], cfg['rates'], cfg['initial'], cfg['kernels_up'], cfg['gin'], cfg['sr'], is_half=False).float().eval()
    generator.load_state_dict({k.removeprefix('dec.'): v.float() for k,v in ckpt.items() if k.startswith('dec.')}, strict=True)
    x = torch.randn(1, cfg['channels'], frames) * .12
    if frames == 2:
        f0 = torch.tensor([[131.2, 0.]], dtype=torch.float32)
    else:
        f0 = torch.tensor([[0., 110., 187.5, 199.4, 0., 130.2]], dtype=torch.float32)
    upsample = math.prod(cfg['rates'])
    phase = torch.rand(1, 1, 1)
    phase[..., 0] = 0
    gaussian = torch.randn(1, x.shape[2] * upsample, 1)
    patches = controlled_random(phase, gaussian)
    with patches[0], patches[1], torch.inference_mode():
        waveform = generator(x, f0, g)
    obj = {
        'sample_rate': cfg['sr'],
        'upsample_rates': cfg['rates'],
        'kernels_per_stage': len(cfg['kernels']),
        'dilations': cfg['dilation'][0],
        'input_shape': list(x.shape),
        'condition_shape': list(g.shape),
        'input': flatten(x),
        'condition': flatten(g),
        'f0': flatten(f0),
        'initial_phases': flatten(phase),
        'gaussian_noise': flatten(gaussian),
        'expected_waveform': flatten(waveform),
    }
    output.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-zip', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--real-checkpoint', type=Path)
    parser.add_argument('--source-fixture', type=Path, required=True)
    parser.add_argument('--waveform-fixture', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(2071)
    torch.set_num_threads(1)
    temporary, _ = import_original(args.source_zip)
    with temporary:
        from tts_with_rvc.infer.lib.infer_pack import models
        src = source_reference(models, args.base_checkpoint, 0, args.source_fixture)
        waveform = waveform_reference(models, args.base_checkpoint, args.waveform_fixture, args.real_checkpoint)
    print('source_samples',len(src['expected_sines']),'waveform_samples',len(waveform['expected_waveform']))


if __name__ == '__main__':
    main()
