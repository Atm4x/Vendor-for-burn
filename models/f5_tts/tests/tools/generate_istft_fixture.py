from pathlib import Path
import json
import torch

frames = 8
generator = torch.Generator().manual_seed(27)
real = torch.randn((513, frames), generator=generator) / 15
imag = torch.randn((513, frames), generator=generator) / 15
imag[0] = 0
imag[-1] = 0
spectrum = torch.complex(real, imag)
padded = torch.cat((spectrum, spectrum[:, -1:]), dim=1)
wave = torch.istft(padded, n_fft=1024, hop_length=256, window=torch.hann_window(1024), center=True)
out = {
    "frames": frames,
    "real": real.T.contiguous().flatten().tolist(),
    "imag": imag.T.contiguous().flatten().tolist(),
    "wave": wave.tolist(),
}
path = Path(__file__).resolve().parent.parent / "fixtures" / "vocos_istft_expected.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"Wrote {path} ({len(out['wave'])} samples)")
