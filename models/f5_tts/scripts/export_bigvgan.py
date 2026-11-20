"""Convert official BigVGAN v2 PyTorch generator checkpoints to native F5-TTS safetensors."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


EXPECTED_CONFIG = {
    "sampling_rate": 24000,
    "num_mels": 100,
    "hop_size": 256,
    "upsample_initial_channel": 1536,
    "upsample_rates": [4, 4, 2, 2, 2, 2],
    "upsample_kernel_sizes": [8, 8, 4, 4, 4, 4],
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5]] * 3,
    "resblock": "1",
    "activation": "snakebeta",
    "snake_logscale": True,
    "use_tanh_at_final": False,
    "use_bias_at_final": False,
}


def validate_config(path: Path) -> None:
    """Reject models with architecture or preprocessing incompatible with the native decoder."""
    config = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"BigVGAN configuration mismatch for {key}: {actual!r}, expected {expected!r}")


def export_checkpoint(input_path: Path, output_path: Path, config_path: Path | None = None) -> None:
    """Export generator tensors as contiguous float32 without changing normalization keys."""
    if input_path.is_dir():
        model_dir = input_path
        input_path = model_dir / "bigvgan_generator.pt"
    else:
        model_dir = input_path.parent
    if config_path is None:
        config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"BigVGAN model config required: {config_path}")
    validate_config(config_path)
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("generator"), dict):
        raise ValueError("Expected official BigVGAN checkpoint with a generator state_dict")
    tensors = checkpoint["generator"]
    required = ("conv_pre.", "ups.0.0.", "resblocks.0.", "activation_post.", "conv_post.")
    for prefix in required:
        if not any(name.startswith(prefix) for name in tensors):
            raise ValueError(f"Missing BigVGAN tensor prefix: {prefix}")
    converted = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Invalid checkpoint tensor {name!r}")
        converted[name] = tensor.detach().cpu().float().contiguous()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, output_path)
    print(f"Exported {len(converted)} BigVGAN tensors to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="BigVGAN model directory or bigvgan_generator.pt")
    parser.add_argument("output", type=Path, help="Destination safetensors file")
    parser.add_argument("--config", type=Path, help="Override official config.json location")
    args = parser.parse_args()
    export_checkpoint(args.source, args.output, args.config)


if __name__ == "__main__":
    main()
