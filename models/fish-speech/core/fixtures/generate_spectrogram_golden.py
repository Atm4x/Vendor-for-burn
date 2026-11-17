"""Generate portable log-mel fixtures from the unmodified Fish 1.5 spectrogram module."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import math

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'vendor/Fish-speech-pipeline-main/fish_speech_lib/fish_speech/utils/spectrogram.py'
spec = importlib.util.spec_from_file_location('fish_reference_mel', SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
torch.set_num_threads(1)


def waveform(count: int) -> torch.Tensor:
    return torch.tensor([
        0.37 * math.sin(2 * math.pi * 440 * i / 44100)
        + 0.21 * math.cos(2 * math.pi * 1300 * i / 44100)
        + 0.03 * math.sin(2 * math.pi * 9000 * i / 44100)
        + 0.02 * (((i * 17) % 31) / 31.0 - 0.5)
        for i in range(count)
    ], dtype=torch.float32)


def main() -> None:
    mel = module.LogMelSpectrogram(sample_rate=44100, n_mels=160, n_fft=2048, hop_length=512, win_length=2048).eval()
    lines = ["# generated from original spectrogram.py with torch CPU / torchaudio Slaney mel"]
    with torch.no_grad():
        for length in (769, 1024, 3073, 4096):
            result = mel(waveform(length).unsqueeze(0))
            assert result.shape == (1, 160, length // 512)
            lines.append(f'{length} {result.shape[2]} ' + ' '.join(f'{v:.9g}' for v in result.flatten().tolist()))
    destination = pathlib.Path(__file__).with_suffix('.txt')
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('created',destination)


if __name__ == '__main__':
    main()
