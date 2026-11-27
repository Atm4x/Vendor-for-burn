"""Reference checkpoint SSML inference, including durations and PCM."""

import argparse
import json
import warnings
from pathlib import Path

import torch
from torch.package import PackageImporter
from safetensors.torch import save_file


def run(checkpoint: Path, destination: Path):
    torch.set_grad_enabled(False)
    package = PackageImporter(str(checkpoint)).load_pickle('tts_models', 'model').packages[0]
    package.unpack_q_model()
    reference = json.loads((destination / 'ssml-parity.json').read_text(encoding='utf-8'))
    output = []
    pcm = {}
    mel_cases = {}
    for index in [0, 1, 8, 12]:
        case = reference[index]
        ssml = case['ssml']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_output = package.models[0](
                sequence=torch.tensor([case['sequence']], dtype=torch.long),
                speaker_ids=torch.tensor([0], dtype=torch.long),
                sr=8000,
                symb_durs={int(key): int(value) for key, value in case['pauses'].items()},
                durs_rate=torch.tensor([case['rates']], dtype=torch.float),
                pitch_coefs=torch.tensor([case['pitches']], dtype=torch.float),
                device='cpu',
            )
        wave, dur = model_output
        if isinstance(wave, tuple):
            wave = package.wrapped_jit_v(wave, 8000)
        tokens = torch.tensor([case['sequence']], dtype=torch.long)
        speaker = torch.tensor([0], dtype=torch.long)
        pitch_coef = torch.tensor([case['pitches']], dtype=torch.float)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        raw_pitch = package.models[0].pitch_predictor(tokens, speaker, mask, 1.0)
        adjusted_pitch = package.models[0].update_pitch_coef(raw_pitch, pitch_coef, speaker)
        mel = package.models[0].tacotron(tokens, speaker, mask, dur, adjusted_pitch, None)
        if case['pauses']:
            mel = package.models[0].fx_pauses(mel, dur, {int(k): v for k, v in case['pauses'].items()})
        mel_cases[f'mel_{index}'] = mel.detach().cpu().float().contiguous()
        mel_cases[f'pitch_{index}'] = adjusted_pitch.detach().cpu().float().contiguous()
        wave = wave.detach().cpu().float().flatten().contiguous()
        key = f'pcm_{index}'
        pcm[key] = wave
        output.append({'ssml': ssml, 'key': key, 'samples': wave.numel(),
                       'durations': dur.detach().cpu().float().flatten().tolist()})
        print(f'SSML case {index}: {wave.numel()} PCM samples, {len(output[-1]["durations"])} durations', flush=True)
    (destination / 'ssml-audio-parity.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    save_file(pcm, str(destination / 'ssml-audio-parity.safetensors'))
    save_file(mel_cases, str(destination / 'ssml-mel-parity.safetensors'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
