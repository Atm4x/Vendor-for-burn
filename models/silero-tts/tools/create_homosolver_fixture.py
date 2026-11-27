"""Export real HomoSolver marker-token inputs and PyTorch reference outputs."""

import argparse
import json
from pathlib import Path

import torch
from torch.package import PackageImporter


def run(checkpoint: Path, destination: Path) -> None:
    torch.set_grad_enabled(False)
    package = PackageImporter(str(checkpoint)).load_pickle("tts_models", "model").packages[0]
    package.unpack_q_model()
    solver = package.accentor.homosolver
    sentences = [
        "На двери замок.", "Она плачет о душе.", "Кто тому верит?", "Он открыл замок ключом.",
        "Есть мел на столе.", "Части города всё еще закрыты.", "Один человек вышел.",
        "Сегодня вечером будет вечер.", "Этому человеку нужен мел.",
    ]
    cases = []
    sentence_results = [{"sentence": text, "result": solver(text), "fully_accented": package.accentor(text)} for text in sentences]
    for sentence in sentences:
        tagged = solver._find_and_tag_homos(sentence)
        for _start, _end, _word, _, raw_mark in tagged:
            if raw_mark is None:
                continue
            ids = solver.tokenizer(raw_mark)
            start = ids.index(solver.tokenizer.homo_start_id)
            end = ids.index(solver.tokenizer.homo_end_id)
            tensor = torch.tensor([ids])
            with torch.no_grad():
                embeddings = solver.model.bert.embeddings(tensor)
                hidden = solver.model.bert(tensor)[0]
                logit = solver.model(tensor, torch.tensor([start]), torch.tensor([end]))
            cases.append({
                "raw_mark": raw_mark,
                "tokens": ids,
                "start": start,
                "end": end,
                "embedding_slice": embeddings.flatten()[:32].tolist(),
                "hidden_slice": hidden.flatten()[:32].tolist(),
                "logit": float(logit[0, 0]),
            })
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "homosolver-parity.json").write_text(
        json.dumps({"cases": cases, "sentences": sentence_results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"HomoSolver fixture: {len(cases)} tagged sentences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
