import json
from pathlib import Path

import torch
import torchaudio


def main():
    torch.manual_seed(131)
    size = 3079
    time = torch.arange(size, dtype=torch.float32)
    wave = 0.03 * torch.randn(size) + 0.13 * torch.sin(time * 0.017) + 0.07 * torch.cos(time * 0.04)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=100,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    )(wave.unsqueeze(0)).clamp(min=1e-5).log().squeeze(0).transpose(0, 1)
    fixture = {
        "frames": mel.shape[0],
        "wave": wave.tolist(),
        "mel": mel.flatten().tolist(),
    }
    dest = Path(__file__).parents[1] / "fixtures" / "vocos_mel_expected.json"
    dest.write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")
    print(f"generated {dest}: {mel.shape}")


if __name__ == "__main__":
    main()
