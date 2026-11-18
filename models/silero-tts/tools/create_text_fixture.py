"""Capture full Silero V5 text-to-PCM references for independent Rust parity."""

import argparse
import json
from pathlib import Path

import torch
from torch.package import PackageImporter
from safetensors.torch import save_file


def run(checkpoint: Path, destination: Path) -> None:
    torch.set_grad_enabled(False)
    package = PackageImporter(str(checkpoint)).load_pickle("tts_models", "model").packages[0]
    cases = []
    audio = {}
    for idx, text in enumerate(["Привет, мир!", "На двери замок."]):
        result = package.apply_tts(text=text, speaker="aidar", sample_rate=8000).detach().cpu().float().contiguous()
        audio[f"pcm_{idx}"] = result
        cases.append({"text": text, "accented": package.accentor(package.prepare_text_input(text)[0]),
            "speaker": "aidar", "sample_rate": 8000, "key": f"pcm_{idx}"})
    destination.mkdir(parents=True, exist_ok=True)
    save_file(audio, str(destination / "full-text-parity.safetensors"))
    (destination / "full-text-parity.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Full-text references:", [(x["text"], audio[x["key"]].numel()) for x in cases])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
