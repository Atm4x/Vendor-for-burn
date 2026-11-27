"""Generate native frontend flag regression fixtures from the bundled SileroStress."""

import argparse
import json
from pathlib import Path

from torch.package import PackageImporter


def run(checkpoint: Path, output: Path) -> None:
    package = PackageImporter(str(checkpoint)).load_pickle("tts_models", "model").packages[0]
    package.unpack_q_model()
    fixtures = []
    options = [
        {},
        {"put_stress_homo": False},
        {"put_yo_homo": False},
        {"put_stress_homo": False, "put_yo_homo": False},
        {"put_accent": False, "put_yo": False},
        {"put_accent": False, "put_stress_homo": False},
        {"put_yo": False},
        {"stress_single_vowel": False},
    ]
    for text in ["Замок стоит у реки.", "Двери и елки в новом доме.", "Я несу мел и кофе."]:
        sentence = package.prepare_text_input(text)[0]
        for flags in options:
            kwargs = dict(flags)
            if "put_accent" in kwargs:
                kwargs["put_stress"] = kwargs.pop("put_accent")
            result = package.accentor(sentence, **kwargs)
            fixtures.append({"text": text, "options": flags, "accented": result})
    output.mkdir(parents=True, exist_ok=True)
    (output / "stress-flags-parity.json").write_text(
        json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Stress flags fixture:", len(fixtures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
