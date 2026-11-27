"""Export reference tensors for acoustic subgraphs from the supplied Silero checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from torch.package import PackageImporter
from safetensors.torch import save_file


def run(checkpoint: Path, output: Path) -> None:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    package = PackageImporter(str(checkpoint)).load_pickle('tts_models', 'model').packages[0]
    model = package.models[0]
    model.eval()
    symbols = json.loads((output / 'config.json').read_text())['symbols']
    characters = '|привет мир~'
    tokens = torch.tensor([[symbols.index(c) for c in characters]], dtype=torch.int64)
    speaker = torch.tensor([0], dtype=torch.int64)
    padding = torch.zeros((1, tokens.shape[1]), dtype=torch.bool)
    duration = model.dur_predictor(tokens, speaker, padding)
    pitch = model.pitch_predictor(tokens, speaker, padding)
    embedded = model.dur_predictor.dur_pred.embedding(tokens)
    speaker_emb = model.dur_predictor.dur_pred.speaker_embedding(speaker)[:, None, :]
    predictor_input = embedded + speaker_emb
    predictor_pos = model.dur_predictor.dur_pred.transformer.pos_encoder(predictor_input.transpose(0, 1)).transpose(0, 1)
    layer0 = model.dur_predictor.dur_pred.transformer.layers._modules["0"](predictor_pos, padding)
    encoder_emb = model.tacotron.embedding(tokens)
    encoded = model.tacotron.encoder(encoder_emb, padding)
    duration_frames = torch.tensor([[1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1]], dtype=torch.float32)
    mel = model.tacotron(tokens, speaker, padding, duration_frames, pitch)
    t = model.tacotron
    mixed = encoded + t.speaker_embedding(speaker)[:, None, :] + t.pitch_proj(pitch).transpose(1, 2)
    expanded = t.len_reg(mixed, duration_frames)
    decoder = t.decoder
    positioned = decoder.pos_encoder(expanded.transpose(0, 1)).transpose(0, 1)
    pre = decoder.pre_vanilla_layers._modules["0"](positioned)
    pad = (3 - pre.shape[1] % 3) % 3
    padded = torch.nn.functional.pad(pre, (0, 0, 0, pad))
    down = decoder.downsample(padded)
    short0 = decoder.shorten_layers._modules["0"](down)
    short1 = decoder.shorten_layers._modules["1"](down)
    up = decoder.upsample(short1)
    post_input = (up + padded)[:, :pre.shape[1], :]
    post = decoder.post_vanilla_layers._modules["0"](post_input)
    decoded = decoder.norm(post)
    full_spectral, full_duration = model(tokens, speaker)
    full_pitch = model.pitch_predictor(tokens, speaker, padding)
    full_mel = model.tacotron(tokens, speaker, padding, full_duration, full_pitch)
    full_pcm8000 = package.wrapped_jit_v(full_spectral, 8000)
    output.mkdir(exist_ok=True, parents=True)
    data = {
        'tokens': tokens,
        'speaker': speaker,
        'duration_log': duration,
        'pitch': pitch,
        'predictor_pos': predictor_pos,
        'predictor_block0': layer0,
        'acoustic_encoded': encoded,
        'dur_frames': duration_frames,
        'mel': mel,
        'mixed': mixed,
        'expanded': expanded,
        'decoder_pos': positioned,
        'decoder_pre': pre,
        'decoder_down': down,
        'decoder_short0': short0,
        'decoder_short1': short1,
        'decoder_up': up,
        'decoder_post_input': post_input,
        'decoder_post': post,
        'decoder_final': decoded,
        'full_duration': full_duration,
        'full_mel': full_mel,
        'full_magnitude': full_spectral[0],
        'full_cos': full_spectral[1],
        'full_sin': full_spectral[2],
        'full_pcm8000': full_pcm8000,
    }
    save_file({k: v.detach().cpu().contiguous().float() for k, v in data.items() if v.is_floating_point()}, str(output / 'acoustic-parity.safetensors'))
    (output / 'acoustic-input.json').write_text(json.dumps({
        'tokens': tokens[0].tolist(), 'speaker': speaker.tolist(), 'lengths': duration_frames[0].tolist(),
        'keys': {k: list(v.shape) for k, v in data.items()},
    }, indent=2))
    print(json.dumps({k: list(v.shape) for k, v in data.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.out)
