import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from reference_flow import import_original


def flatten(x):
    return x.detach().float().contiguous().view(-1).tolist()


def make_stage(args):
    torch.manual_seed(7629)
    torch.set_num_threads(1)
    from tts_with_rvc.infer.lib.infer_pack.models import GeneratorNSF
    if args.checkpoint is None:
        model = GeneratorNSF(4, '1', [3, 7, 11], [[1, 3, 5]] * 3, [2], 8, [4], 0, 40000, is_half=False)
        stride, future_stride = 2, 1
        input_tensor = torch.randn(1, 8, 5) * 0.25
        source = torch.randn(1, 1, 10) * 0.07
        stage = 0
    else:
        data = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
        config = data['config']
        model = GeneratorNSF(config[2], config[9], config[10], config[11], config[12], config[13], config[14], config[16], config[17], is_half=False)
        params = {name.removeprefix('dec.'): value.float() for name, value in data['weight'].items() if name.startswith('dec.')}
        model.load_state_dict(params, strict=True)
        stage = 0
        stride = int(config[12][stage])
        future_stride = math.prod(config[12][stage + 1:])
        input_tensor = torch.randn(1, int(config[13]), 2) * 0.15
        source = torch.randn(1, 1, input_tensor.shape[-1] * math.prod(config[12])) * 0.05
    model.eval().float()
    with torch.inference_mode():
        upsampled = model.ups[stage](F.leaky_relu(input_tensor, 0.1))
        noise = model.noise_convs[stage](source)
        assert upsampled.shape == noise.shape
        before_residual = upsampled + noise
        output = sum(model.resblocks[stage * model.num_kernels + i](before_residual) for i in range(model.num_kernels)) / model.num_kernels
        assert torch.isfinite(output).all()
    return model, {
        'stage': stage,
        'stride': stride,
        'future_stride': future_stride,
        'num_kernels': model.num_kernels,
        'dilations': [1, 3, 5],
        'input_shape': list(input_tensor.shape),
        'source_shape': list(source.shape),
        'input': flatten(input_tensor),
        'harmonic_source': flatten(source),
        'expected_upsampled': flatten(upsampled),
        'expected_output': flatten(output),
    }


def save_synthetic_pth(model, output):
    state = {}
    for name, tensor in model.state_dict().items():
        if not (name.startswith('ups.0.') or name.startswith('noise_convs.0.') or name.startswith('resblocks.')):
            continue
        name = name.replace('.parametrizations.weight.original0', '.weight_g')
        name = name.replace('.parametrizations.weight.original1', '.weight_v')
        state['dec.' + name] = tensor.detach().float()
    torch.save({'weight': state, 'version': 'v2'}, output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-zip', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--synthetic-checkpoint', type=Path)
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    if bool(args.checkpoint) == bool(args.synthetic_checkpoint):
        p.error('pass exactly one of --synthetic-checkpoint or --checkpoint')
    temporary, _ = import_original(args.source_zip)
    with temporary:
        model, fixture = make_stage(args)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture, indent=2), encoding='utf-8')
    if args.synthetic_checkpoint:
        save_synthetic_pth(model, args.synthetic_checkpoint)
    print('stage0', fixture['input_shape'], fixture['source_shape'], max(map(abs,fixture['expected_output'])))


if __name__ == '__main__':
    main()
