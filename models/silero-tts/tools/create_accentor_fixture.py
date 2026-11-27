"""Build reference Accentor feature and text fixtures from the supplied checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from torch.package import PackageImporter


def run(checkpoint: Path, destination: Path) -> None:
    torch.set_grad_enabled(False)
    package = PackageImporter(str(checkpoint)).load_pickle("tts_models", "model").packages[0]
    accentor = package.accentor.accentor
    model = accentor.model
    words = [
        "", "привет", "ёлка", "елка", "мир", "недалеко", "человек", "пришел", "мне", "замок",
        "я", "и", "двери", "абракадабра", "кто", "да", "объём", "кофе", "утро", "поезд",
        "красивый", "окно", "москва", "проверка", "флаг", "торжество", "здание",
    ]
    sentences = [
        "Привет, мир!", "Один человек пришёл домой.", "Елка и ёлка",
        "Кто-то мне сказал: всё хорошо.", "Я - кот, а ты?", "Есть замок на двери.",
        "ПРОВЕРКА  ПРОВЕРКА!", "я уже пью кофе", "недалеко красивый дом",
        "У меня есть к+офе", "с+ёстры и елки", "Привет -- мир", "лёд и много елок",
        "", "+а", "hello", "вс+е", "пере+езд",
    ]
    cases = []
    for sentence in sentences:
        try:
            actual = accentor(sentence)
            cases.append({"sentence": sentence, "result": actual})
        except Exception as exc:
            cases.append({"sentence": sentence, "error": str(exc)})
    with torch.no_grad():
        embeddings = model.embedding(words)
        stress, yo = model(words)
    result = {
        "words": words,
        "embeddings": embeddings.tolist(),
        "stress_logits": stress.tolist(),
        "yo_logits": yo.tolist(),
        "cases": cases,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "accentor-parity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Accentor fixture: {len(words)} words, {len(cases)} sentences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
