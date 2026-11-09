import argparse
import json
import math
from pathlib import Path

import torch

from reference_flow import import_original


def flatten(tensor):
    return tensor.detach().float().contiguous().reshape(-1).tolist()


def make_model(checkpoint):
    from tts_with_rvc.infer.lib.infer_pack.models import GeneratorNSF
    if checkpoint is None:
        rates, kernels, dils = [2, 2], [3, 7, 11], [[1, 3, 5]] * 3
        model = GeneratorNSF(4, '1', kernels, dils, rates, 8, [4, 4], 3, 40000, is_half=False)
        g = torch.randn(1, 3, 1) * 0.1
        x = torch.randn(1, 4, 4) * 0.1
    else:
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        cfg = data['config']
        rates, kernels, dils = cfg[12], cfg[10], cfg[11]
        model = GeneratorNSF(cfg[2], cfg[9], kernels, dils, rates, cfg[13], cfg[14], cfg[16], cfg[17], is_half=False)
        state = {name.removeprefix('dec.'): value.float() for name, value in data['weight'].items() if name.startswith('dec.')}
        model.load_state_dict(state, strict=True)
        g = data['weight']['emb_g.weight'][0].float().reshape(1, cfg[16], 1)
        x = torch.randn(1, cfg[2], 2) * 0.1
    harmonic = torch.randn(1, 1, x.shape[-1] * math.prod(rates)) * 0.07
    model.float().eval()
    return model, x, g, harmonic, rates, kernels, dils


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-zip', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--synthetic-checkpoint', type=Path)
    p.add_argument('--fixture', type=Path, required=True)
    a = p.parse_args()
    if bool(a.checkpoint) == bool(a.synthetic_checkpoint):
        p.error('exactly one of --checkpoint and --synthetic-checkpoint is required')
    torch.manual_seed(1143)
    torch.set_num_threads(1)
    temporary, _ = import_original(a.source_zip)
    with temporary:
        model, x, g, harmonic, rates, kernels, dils = make_model(a.checkpoint)
        with torch.inference_mode():
            def provided_harmonic(f0, upp):
                assert upp == math.prod(rates)
                return harmonic.transpose(1, 2), None, None
            model.m_source.forward = provided_harmonic
            f0_dummy = torch.zeros(x.shape[0], x.shape[-1])
            waveform = model.forward(x, f0_dummy, g=g)
            assert waveform.shape[-1] == harmonic.shape[-1]
            assert torch.isfinite(waveform).all()
        fixture = {
            'upsample_rates': rates,
            'kernels_per_stage': len(kernels),
            'dilations': dils[0],
            'input_shape': list(x.shape),
            'source_shape': list(harmonic.shape),
            'condition_shape': list(g.shape),
            'input': flatten(x),
            'harmonic_source': flatten(harmonic),
            'condition': flatten(g),
            'expected_waveform': flatten(waveform),
        }
        if a.synthetic_checkpoint is not None:
            state = {}
            for name, value in model.state_dict().items():
                name = name.replace('.parametrizations.weight.original0', '.weight_g')
                name = name.replace('.parametrizations.weight.original1', '.weight_v')
                state['dec.' + name] = value.detach().float()
            torch.save({'weight':state,'version':'v2'},a.synthetic_checkpoint)
    a.fixture.parent.mkdir(parents=True, exist_ok=True)
    a.fixture.write_text(json.dumps(fixture,indent=2),encoding='utf-8')
    print('channels',x.shape[1],'rates',rates,'waveform_samples',waveform.numel(),'mean_abs',waveform.abs().mean().item())


if __name__ == '__main__':
    main()
