"""Offline trusted Fish Speech 1.5 checkpoint exporter; never used during inference."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unpack_state(checkpoint, component: str):
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint contains no state_dict")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("checkpoint must contain a tensor-only state_dict")
    if component == "semantic":
        prefixed = [key.startswith("model.") for key in state]
        if any(prefixed):
            if not all(prefixed):
                raise ValueError("mixed semantic model. prefixes are ambiguous")
            state = {key.removeprefix("model."): value for key, value in state.items()}
    elif component == "firefly":
        prefixed = {key: value for key, value in state.items() if key.startswith("generator.")}
        if prefixed:
            state = {key.removeprefix("generator."): value for key, value in prefixed.items()}
    else:
        raise ValueError("component must be semantic or firefly")
    return dict(state)


def pack_qkv(state):
    packed = dict(state)
    for key in tuple(state):
        if not key.endswith("wq.weight"):
            continue
        prefix = key[: -len("wq.weight")]
        parts = [prefix + part + ".weight" for part in ("wq", "wk", "wv")]
        if any(part not in packed for part in parts):
            raise ValueError(f"partial legacy QKV triplet: {prefix}")
        out_key = prefix + "wqkv.weight"
        if out_key in packed:
            raise ValueError(f"QKV collision: {out_key}")
        q, k, v = (packed.pop(part) for part in parts)
        if not all(tensor.ndim == 2 and tensor.shape[1] == q.shape[1] for tensor in (q, k, v)):
            raise ValueError(f"QKV incompatible input dimensions: {prefix}")
        packed[out_key] = torch.cat([q, k, v], dim=0)
    if any(key.endswith(("wk.weight", "wv.weight")) for key in packed):
        raise ValueError("orphan K/V tensor without Q")
    return packed


def flatten_weight_norm(state):
    packed = dict(state)
    patterns = (
        (".parametrizations.weight.original0", ".parametrizations.weight.original1"),
        (".weight_g", ".weight_v"),
    )
    for g_suffix, v_suffix in patterns:
        for g_key in tuple(state):
            if not g_key.endswith(g_suffix):
                continue
            stem = g_key[: -len(g_suffix)]
            v_key = stem + v_suffix
            result_key = stem + ".weight"
            if v_key not in packed or result_key in packed:
                raise ValueError(f"weight_norm incomplete or colliding: {stem}")
            g, v = packed.pop(g_key), packed.pop(v_key)
            if v.ndim < 2 or g.numel() != v.shape[0]:
                raise ValueError(f"weight_norm dim=0 extent mismatch: {stem}")
            packed[result_key] = torch._weight_norm(v, g, 0)
    if any(key.endswith(suffix) for key in packed for suffix in (
        ".parametrizations.weight.original0", ".parametrizations.weight.original1",
        ".weight_g", ".weight_v",
    )):
        raise ValueError("unpaired weight_norm parameterization")
    return packed


def canonicalize(checkpoint, component: str):
    state = unpack_state(checkpoint, component)
    if component == "semantic":
        state = pack_qkv(state)
    else:
        state = flatten_weight_norm(state)
    result = {}
    for key, tensor in state.items():
        if tensor.layout != torch.strided or tensor.is_sparse or tensor.is_quantized:
            raise ValueError(f"unsupported tensor layout: {key}")
        tensor = tensor.detach().cpu().contiguous()
        if tensor.numel() == 0 or tensor.ndim == 0 or tensor.dtype not in (
            torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64
        ):
            raise ValueError(f"unsupported dtype or empty tensor: {key}")
        result[key] = tensor
    return result


def validate_firefly_source_config(config_path: Path):
    if config_path.suffix.lower() != ".json":
        raise ValueError("Firefly exporter requires normalized JSON from normalize_firefly_config.py")
    reference = json.loads((Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "firefly_1_5.json").read_text(encoding="utf-8"))
    try:
        supplied = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("invalid normalized Firefly JSON configuration") from exc
    if supplied != reference:
        raise ValueError("Firefly config does not match the supported source variant")


def export(checkpoint_path: Path, config_path: Path, destination: Path, component: str, source_license: str):
    if not source_license.strip():
        raise ValueError("source checkpoint license must be supplied explicitly")
    if component == "firefly":
        validate_firefly_source_config(config_path)
    source_sha = hash_file(checkpoint_path)
    config_sha = hash_file(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    tensors = canonicalize(checkpoint, component)
    if not tensors:
        raise ValueError("empty model checkpoint")
    destination.mkdir(parents=True, exist_ok=True)
    shard = component + ".safetensors"
    file_path = destination / shard
    save_file(tensors, str(file_path))
    metadata = {}
    for key, tensor in sorted(tensors.items()):
        raw_bytes = tensor.view(torch.uint8).numpy().tobytes()
        dtype = {
            torch.float32: "F32",
            torch.float16: "F16",
            torch.bfloat16: "BF16",
            torch.int32: "I32",
            torch.int64: "I64",
        }[tensor.dtype]
        metadata[key] = {
            "shard": shard,
            "dtype": dtype,
            "shape": list(tensor.shape),
            "sha256": digest(raw_bytes),
        }
    manifest = {
        "schema_version": 1,
        "model_family": "fish-speech-1.5",
        "component": component,
        "source_checkpoint_sha256": source_sha,
        "source_license": source_license,
        "config_sha256": config_sha,
        "shards": {shard: hash_file(file_path)},
        "tensors": metadata,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("semantic", "firefly"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-license", required=True)
    args = parser.parse_args()
    manifest = export(args.checkpoint, args.config, args.out, args.component, args.source_license)
    print(f"exported {len(manifest['tensors'])} tensors to {args.out}; not GPU-validated")


if __name__ == "__main__":
    main()
