import argparse
import json
import math
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F


def import_original(source_zip: Path):
    temp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(temp.name)
    folders = list(Path(temp.name).glob('*/tts_with_rvc/infer/lib/infer_pack/models.py'))
    if len(folders) != 1:
        raise ValueError('Expected original TTS-with-RVC source layout')
    root = folders[0].parents[3]
    for name, path in (
        ('tts_with_rvc', root),
        ('tts_with_rvc.infer', root / 'infer'),
        ('tts_with_rvc.infer.lib', root / 'infer' / 'lib'),
        ('tts_with_rvc.infer.lib.infer_pack', root / 'infer' / 'lib' / 'infer_pack'),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    from tts_with_rvc.infer.lib.infer_pack.models import TextEncoder
    return temp, TextEncoder


def floats(values):
    return values.detach().contiguous().to(dtype=torch.float32, device='cpu').flatten().tolist()


def make_fixture(TextEncoder, checkpoint: Path | None):
    torch.set_num_threads(1)
    torch.manual_seed(5271)
    if checkpoint:
        obj = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if obj.get('version') != 'v2' or obj.get('f0') != 1:
            raise ValueError('This fixture generator expects an RVC v2 F0 checkpoint')
        cfg = obj['config']
        hidden, filter_channels, heads, n_layers, kernel, out_channels = (
            cfg[2], cfg[4], cfg[5], cfg[6], cfg[7], cfg[3]
        )
        input_channels = obj['weight']['enc_p.emb_phone.weight'].shape[1]
        batch, time, lengths = 1, 14, [12]
    else:
        input_channels, hidden, filter_channels, heads = 4, 4, 8, 2
        n_layers, kernel, out_channels = 6, 3, 2
        batch, time, lengths = 2, 13, [13, 10]

    model = TextEncoder(
        input_channels, out_channels, hidden, filter_channels,
        heads, n_layers, kernel, 0.0, f0=True,
    ).float().eval()
    if checkpoint:
        weights = {
            key.removeprefix('enc_p.'): value.float().contiguous()
            for key, value in obj['weight'].items()
            if key.startswith('enc_p.')
        }
        model.load_state_dict(weights, strict=True)

    phone = torch.randn(batch, time, input_channels) * 0.11
    pitch = torch.randint(0, 256, (batch, time), dtype=torch.int64)
    seq_lengths = torch.tensor(lengths, dtype=torch.int64)
    sequence_mask = (torch.arange(time).unsqueeze(0) < seq_lengths.unsqueeze(1)).float().unsqueeze(1)

    with torch.inference_mode():
        front_values = F.leaky_relu(
            (model.emb_phone(phone) + model.emb_pitch(pitch)) * math.sqrt(hidden),
            negative_slope=0.1,
        ).transpose(1, 2)
        mean, logs, mask = model(phone, pitch, seq_lengths)
        skip_head = 2
        skip_mean, skip_logs, skip_mask = model(
            phone, pitch, seq_lengths, skip_head=torch.tensor(skip_head)
        )

        def layer_payload(i, layer_input, attention_output, layer_output):
            attn = model.encoder.attn_layers[i]
            ffn = model.encoder.ffn_layers[i]
            norm_a = model.encoder.norm_layers_1[i]
            norm_b = model.encoder.norm_layers_2[i]
            record = {
                'batch': batch,
                'time': time,
                'channels': hidden,
                'heads': heads,
                'window': model.encoder.window_size,
                'input': floats(layer_input),
                'lengths': lengths,
                'expected': floats(attention_output),
                'relative_keys': floats(attn.emb_rel_k),
                'relative_values': floats(attn.emb_rel_v),
                'filter_channels': filter_channels,
                'kernel': kernel,
                'norm_attention_gamma': floats(norm_a.gamma),
                'norm_attention_beta': floats(norm_a.beta),
                'norm_feed_forward_gamma': floats(norm_b.gamma),
                'norm_feed_forward_beta': floats(norm_b.beta),
                'conv1_weight': floats(ffn.conv_1.weight),
                'conv1_bias': floats(ffn.conv_1.bias),
                'conv2_weight': floats(ffn.conv_2.weight),
                'conv2_bias': floats(ffn.conv_2.bias),
                'expected_encoder': floats(layer_output * sequence_mask),
            }
            for suffix, name in [('q', 'query'), ('k', 'key'), ('v', 'value'), ('o', 'output')]:
                conv = getattr(attn, f'conv_{suffix}')
                record[f'{name}_weight'] = floats(conv.weight)
                record[f'{name}_bias'] = floats(conv.bias)
            return record

        x = front_values * sequence_mask
        layers = []
        attention_mask = sequence_mask.unsqueeze(2) * sequence_mask.unsqueeze(-1)
        for i in range(n_layers):
            attention = model.encoder.attn_layers[i](x, x, attention_mask)
            intermediate = model.encoder.norm_layers_1[i](x + attention)
            feed_forward = model.encoder.ffn_layers[i](intermediate, sequence_mask)
            next_x = model.encoder.norm_layers_2[i](intermediate + feed_forward)
            layers.append(layer_payload(i, x, attention, next_x))
            x = next_x
        x = x * sequence_mask
        stats = model.proj(x) * sequence_mask
        expected_stats = torch.cat([mean, logs], dim=1)
        assertion_error = (stats - expected_stats).abs().max().item()
        if assertion_error > 1e-5:
            raise ValueError(f'Independent original-module forward differs by {assertion_error}')

        front = {
            'batch': batch,
            'time': time,
            'in_channels': input_channels,
            'hidden_channels': hidden,
            'phone': floats(phone),
            'pitch': pitch.flatten().tolist(),
            'lengths': lengths,
            'phone_weight': floats(model.emb_phone.weight),
            'phone_bias': floats(model.emb_phone.bias),
            'pitch_weight': floats(model.emb_pitch.weight),
            'expected_features': floats(front_values),
            'expected_mask': floats(sequence_mask),
        }
        return {
            'front': front,
            'layers': layers,
            'output_channels': out_channels,
            'stats_weight': floats(model.proj.weight),
            'stats_bias': floats(model.proj.bias),
            'expected_mean': floats(mean),
            'expected_log_scale': floats(logs),
            'expected_mask': floats(mask),
            'skip_head': skip_head,
            'expected_skip_mean': floats(skip_mean),
            'expected_skip_log_scale': floats(skip_logs),
            'expected_skip_mask': floats(skip_mask),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-zip', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    temp, model_type = import_original(args.source_zip)
    try:
        fixture = make_fixture(model_type, args.checkpoint)
    finally:
        temp.cleanup()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    print(args.output, args.output.stat().st_size, len(fixture['layers']))


if __name__ == '__main__':
    main()
