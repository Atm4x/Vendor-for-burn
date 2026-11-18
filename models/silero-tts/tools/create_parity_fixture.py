"""Generate reproducible PyTorch reference tensors for native Rust parity tests."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def run(checkpoint: Path, output: Path) -> None:
    from torch.package import PackageImporter

    output.mkdir(parents=True, exist_ok=True)
    model = PackageImporter(str(checkpoint)).load_pickle("tts_models", "model").packages[0]
    torch.manual_seed(123)
    mel = torch.randn(1, 192, 5) * 0.3
    with torch.no_grad():
        mag, cosine, sine = model.models[0].vocoder(mel, 48000, 0.0, True)
        result = {"mel": mel, "mag": mag, "cos": cosine, "sin": sine}
        for sample_rate in (48000, 24000, 8000):
            result[f"wave{sample_rate}"] = model.wrapped_jit_v(
                (mag, cosine, sine), sample_rate
            )
    save_file({key: tensor.detach().cpu().contiguous() for key, tensor in result.items()},
              str(output / "parity.safetensors"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    run(arguments.checkpoint, arguments.output)
