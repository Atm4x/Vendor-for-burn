import argparse
import json
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import torch


def import_original(source_zip):
    temporary = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(temporary.name)
    matches = list(Path(temporary.name).glob('*/tts_with_rvc/infer/lib/infer_pack/models.py'))
    if len(matches) != 1:
        raise ValueError('Original RVC archive layout is unexpected')
    root = matches[0].parents[3]
    for name, path in (
        ('tts_with_rvc', root),
        ('tts_with_rvc.infer', root / 'infer'),
        ('tts_with_rvc.infer.lib', root / 'infer' / 'lib'),
        ('tts_with_rvc.infer.lib.infer_pack', root / 'infer' / 'lib' / 'infer_pack'),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    from tts_with_rvc.infer.lib.infer_pack.models import ResidualCouplingBlock
    return temporary, ResidualCouplingBlock


def values(tensor):
    return tensor.detach().float().contiguous().cpu().reshape(-1).tolist()


def generate(model_class, checkpoint):
    torch.manual_seed(2407)
    torch.set_num_threads(1)
    if checkpoint:
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if data.get('version') != 'v2':
            raise ValueError('Expected RVC v2 checkpoint')
        config = data['config']
        channels, hidden, gin = config[2], config[3], config[-2]
        model = model_class(channels, hidden, 5, 1, 3, n_flows=4, gin_channels=gin)
        state = {key.removeprefix('flow.'): value.float() for key, value in data['weight'].items() if key.startswith('flow.')}
        model.load_state_dict(state, strict=True)
        condition = data['weight']['emb_g.weight'][0].float().reshape(1, gin, 1)
        batch, time, length = 1, 8, 6
    else:
        channels, hidden, gin = 8, 4, 3
        model = model_class(channels, hidden, 5, 1, 3, n_flows=2, gin_channels=gin)
        for flow in model.flows:
            if hasattr(flow, 'post'):
                torch.nn.init.normal_(flow.post.weight, mean=0.0, std=0.015)
                torch.nn.init.normal_(flow.post.bias, mean=0.0, std=0.015)
        condition = torch.randn(1, gin, 1) * 0.2
        batch, time, length = 1, 8, 6

    model.float().eval()
    x = torch.randn(batch, channels, time) * 0.13
    x[:, :, length:] = 0.0
    mask = (torch.arange(time) < length).float().reshape(1, 1, time)
    with torch.inference_mode():
        expected = model(x, mask, g=condition, reverse=False)
        reverse = model(expected, mask, g=condition, reverse=True)
        assert torch.isfinite(expected).all()
        assert torch.allclose(reverse[:, :, :length], x[:, :, :length], atol=1e-4)

    return model, {
        'batch': batch,
        'channels': channels,
        'time': time,
        'gin_channels': gin,
        'x': values(x),
        'mask': values(mask),
        'condition': values(condition),
        'expected_forward': values(expected),
        'expected_reverse': values(reverse),
    }


def write_checkpoint(model, output):
    state = {}
    for name, tensor in model.state_dict().items():
        name = name.replace('.parametrizations.weight.original0', '.weight_g')
        name = name.replace('.parametrizations.weight.original1', '.weight_v')
        state['flow.' + name] = tensor.detach().to(dtype=torch.float32)
    torch.save({'weight': state, 'version': 'v2'}, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-zip', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--synthetic-checkpoint', type=Path)
    parser.add_argument('--real-checkpoint', type=Path)
    args = parser.parse_args()
    if bool(args.synthetic_checkpoint) == bool(args.real_checkpoint):
        parser.error('specify either --synthetic-checkpoint or --real-checkpoint')
    temporary, model_class = import_original(args.source_zip)
    with temporary:
        model, fixture = generate(model_class, args.real_checkpoint)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(json.dumps(fixture), encoding='utf-8')
    if args.synthetic_checkpoint:
        args.synthetic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        write_checkpoint(model, args.synthetic_checkpoint)
    print(f"Reference: channels={fixture['channels']}, time={fixture['time']}, max={max(map(abs, fixture['expected_forward'])):.7g}")


if __name__ == '__main__':
    main()
