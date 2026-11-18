"""Export ordered symbol-duration SSML and audio references from Silero V5 RU."""

import argparse
import json
import warnings
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.package import PackageImporter


CASES = [
    ("Да.", [("а", 75)]),
    ("Привет!", [(".", 50), ("!", 75)]),
    ("Мама.", [("а", 75), ("ма", 150)]),
    ("Да.", [("Да", 100), ("а", 50)]),
    ("На двери замок.", [("замок", 125)]),
    ("Это!", [("!", 250)]),
    ("Мир.", [("и", 80), ("р", 120)]),
    ("Да.", [("", 50)]),
]


def run(checkpoint: Path, destination: Path) -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    package = PackageImporter(str(checkpoint)).load_pickle("tts_models", "model").packages[0]
    package.unpack_q_model()
    fixtures = []
    for index, (text, replacements) in enumerate(CASES):
        ssml = package.transform_durs_to_ssml(text, dict(replacements))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sentences, clean, breaks, rates, pitches, _ = package.prepare_tts_model_input(
                ssml, ssml=True, speaker_ids=[0]
            )
            ids, pauses, rate_tensor, pitch_tensor = package.merge_batch_model(
                sentences, breaks, rates, pitches
            )
        fixtures.append({
            "text": text,
            "symbol_durs": replacements,
            "ssml": ssml,
            "segments": package.process_ssml(ssml),
            "accented": package.accentor(" ".join(sentences)),
            "ids": ids.squeeze().tolist(),
            "pauses": {str(key): value for key, value in pauses.items()},
            "rates": rate_tensor.squeeze().tolist(),
            "pitches": pitch_tensor.squeeze().tolist(),
        })
        print(f"symbol_durs {index}: tokens={ids.numel()}, pauses={pauses}", flush=True)

    audio = {}
    for rate in (8000, 24000, 48000):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcm = package.apply_tts(
                text=CASES[0][0], symbol_durs=dict(CASES[0][1]),
                speaker="aidar", sample_rate=rate,
            )
        audio[f"pcm_{rate}"] = pcm.detach().cpu().float().flatten().contiguous()
        print(f"symbol_durs audio {rate}: {audio[f'pcm_{rate}'].numel()} samples", flush=True)

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "symbol-durations-parity.json").write_text(
        json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_file(audio, str(destination / "symbol-durations-audio.safetensors"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
