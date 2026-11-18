"""Export Silero v5_ru.pt without retaining PyTorch as a runtime dependency.

This conversion runs once on a developer machine with PyTorch and safetensors.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def run(checkpoint: Path, output: Path) -> None:
    from torch.package import PackageImporter

    output.mkdir(parents=True, exist_ok=True)
    package = PackageImporter(str(checkpoint)).load_pickle("tts_models", "model")
    if len(package.packages) != 1:
        raise ValueError("Unexpected multiple Silero packages")
    model_package = package.packages[0]
    if len(model_package.models) != 1:
        raise ValueError("Unexpected multiple acoustic models")

    networks = {
        "tts": model_package.models[0],
        "accentor": model_package.accentor.accentor.model,
        "homosolver": model_package.accentor.homosolver.model,
    }
    summary = {}
    for name, model in networks.items():
        state = {
            key: tensor.detach().cpu().contiguous()
            for key, tensor in model.state_dict().items()
        }
        save_file(state, str(output / f"{name}.safetensors"))
        summary[name] = {
            "tensors": len(state),
            "parameters": sum(t.numel() for t in state.values()),
            "dtypes": sorted(set(str(t.dtype) for t in state.values())),
        }

    pqmf = model_package.wrapped_jit_v
    save_file({
        "analysis_2": pqmf.pqmf_2.H.detach().cpu().contiguous(),
        "analysis_6": pqmf.pqmf_6.H.detach().cpu().contiguous(),
    }, str(output / "pqmf.safetensors"))

    ngrams = model_package.accentor.accentor.model.embedding.ngram_dict
    if "UNK" not in ngrams or len(ngrams) == 0:
        raise ValueError("Silero Accentor's subword ngram vocabulary is incomplete")
    (output / "accentor-ngrams.json").write_text(
        json.dumps(ngrams, ensure_ascii=False), encoding="utf-8"
    )

    tokenizer = model_package.accentor.homosolver.tokenizer
    vocab = sorted(tokenizer.vocab.items(), key=lambda item: item[1])
    (output / "vocab.json").write_text(
        json.dumps(dict(vocab), ensure_ascii=False), encoding="utf-8"
    )
    (output / "accentor-exceptions.json").write_text(
        json.dumps(model_package.accentor.accentor.exceptions, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "homographs.json").write_text(
        json.dumps(model_package.accentor.homosolver.homodict, ensure_ascii=False),
        encoding="utf-8",
    )
    config = {
        "format_version": 1,
        "source_checkpoint": checkpoint.name,
        "speakers": model_package.speaker_to_ids[0],
        "symbols": model_package.symbols,
        "alphabet": "".join(model_package.alphabet),
        "sos": model_package.sos_token,
        "eos": model_package.eos_token,
        "phons": model_package.phons,
        "sample_rates": [8000, 24000, 48000],
        "networks": summary,
        "tokenizer": {
            "unk_token": tokenizer.unk_token,
            "cls_token": tokenizer.cls_token,
            "sep_token": tokenizer.sep_token,
            "pad_token": tokenizer.pad_token,
            "homo_start_id": tokenizer.homo_start_id,
            "homo_end_id": tokenizer.homo_end_id,
            "never_split": sorted(tokenizer.never_split),
            "tokenize_chinese_chars": tokenizer.tokenize_chinese_chars,
            "model_max_length": tokenizer.model_max_length,
        },
    }
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
