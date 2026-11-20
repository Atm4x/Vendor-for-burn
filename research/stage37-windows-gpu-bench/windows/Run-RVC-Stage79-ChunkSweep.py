#!/usr/bin/env python3
"""A/B long-WAV chunk geometry with pinned RVC compute policies (stdlib only)."""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import wave
import zipfile


MODES = {
    "direct": (None, "direct"),
    "tf32x2": ("RVC_RMVPE_UNET_COMPENSATED_TF32", "compensated_tf32x2"),
    "batched": ("RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32", "compensated_tf32x2_batched"),
}
CLEAR_FLAGS = (
    "RVC_GPU_FAISS", "RVC_GPU_F0", "RVC_GLOBAL_F0", "RVC_PROFILE",
    "RVC_DIAGNOSTIC_TRACE", "RVC_NSF_CONV_KERNEL", "RVC_NSF_CONV1D_GEMM",
    "RVC_NSF_TRANSPOSE_GEMM", "RVC_RMVPE_FUSED_READBACK", "RVC_RMVPE_COMPACT_READBACK",
    "RVC_RMVPE_PLANE_READBACK", "RVC_HUBERT_FRONTEND_GEMM", "RVC_RMVPE_UNET_GEMM",
    "RVC_RMVPE_UNET_CONV2D_GEMM", "RVC_RMVPE_UNET_TRANSPOSE2D_GEMM",
    "RVC_RMVPE_UNET_COMPENSATED_TF32", "RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32",
    "RVC_F0_REFERENCE_DIR", "RVC_F0_REFERENCE_MODE", "RVC_F0_AUDIT_DIR",
)


def grid_parse(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        core_text, separator, batch_text = item.strip().partition(":")
        if not separator:
            raise ValueError(f"Expected CORE:BATCH, got {item!r}")
        core, batch = int(core_text), int(batch_text)
        if not (25 <= core <= 900 and 1 <= batch <= 32 and core + 2 * 40 + 4 <= 1000):
            raise ValueError(f"Invalid RVC core/batch geometry: {core}:{batch}")
        if (core, batch) in result:
            raise ValueError(f"Repeated RVC geometry: {core}:{batch}")
        result.append((core, batch))
    if not result:
        raise ValueError("The geometry grid is empty")
    return result


def work_estimate(duration_seconds: float, core: int) -> dict[str, int | float]:
    total_frames = math.ceil(duration_seconds * 100)
    step = core - 12
    if step <= 0:
        raise ValueError("Core must be greater than 12 overlap frames")
    start, chunks, processed = 0, 0, 0
    while True:
        end = min(total_frames, start + core)
        processed += end - start + 2 * 40 + 4
        chunks += 1
        if end == total_frames:
            break
        start += step
    return {
        "core_frames": core, "chunks_estimate": chunks,
        "model_frames_estimate": processed,
        "model_seconds_estimate": processed / 100,
        "relative_to_input": processed / total_frames,
    }


def wav_pcm16(path: Path) -> array.array:
    with wave.open(str(path), "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2 or f.getframerate() != 40000:
            raise ValueError(f"Expected 40 kHz mono PCM16, got {path}")
        payload = f.readframes(f.getnframes())
    result = array.array("h")
    result.frombytes(payload)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def compare_pcm(left_path: Path, right_path: Path) -> dict[str, float | int]:
    left, right = wav_pcm16(left_path), wav_pcm16(right_path)
    if len(left) != len(right):
        raise ValueError(f"PCM lengths differ: {len(left)} vs {len(right)}")
    count, max_delta, squared, signal = 0, 0, 0, 0
    for a, b in zip(left, right):
        delta = a - b
        count += delta != 0
        max_delta = max(max_delta, abs(delta))
        squared += delta * delta
        signal += a * a
    n = len(left)
    rmse = math.sqrt(squared / n)
    return {
        "samples": n, "different_samples": count, "max_abs_pcm16": max_delta,
        "rmse_pcm16": rmse,
        "snr_db": 10 * math.log10(signal / squared) if squared else None,
    }


def resolve(path: str, base: Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="tts-with-rvc-rs.exe")
    parser.add_argument("--voice", default="embed/CrazyMita.pth")
    parser.add_argument("--hubert", default="embed/hubert_base.pt")
    parser.add_argument("--rmvpe", default="embed/rmvpe.pt")
    parser.add_argument("--index", default="embed/CrazyMita.index")
    parser.add_argument("--input-wav", default="embed/MitaKind.wav")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--grid", default="300:2,300:1,600:1,900:1")
    parser.add_argument("--modes", default="direct,tf32x2")
    parser.add_argument("--out", default="stage79_chunk_sweep_results")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    grid = grid_parse(args.grid)
    modes = args.modes.split(",")
    if len(set(modes)) != len(modes) or "direct" not in modes or any(m not in MODES for m in modes):
        parser.error("--modes must include direct and may include tf32x2,batched, without repeats")
    if not 2 <= args.runs <= 16:
        parser.error("--runs must be 2..16")
    if not 0 <= args.index_rate <= 1:
        parser.error("--index-rate must be 0..1")
    paths = {name: resolve(value, base) for name, value in {
        "exe": args.exe, "voice": args.voice, "hubert": args.hubert,
        "rmvpe": args.rmvpe, "index": args.index, "input_wav": args.input_wav,
    }.items()}
    for name in ("exe", "voice", "hubert", "rmvpe", "input_wav"):
        if not paths[name].is_file():
            parser.error(f"Missing required {name}: {paths[name]}")
    if args.index_rate > 0 and not paths["index"].is_file():
        parser.error(f"Missing index: {paths['index']}")
    out = resolve(args.out, base)
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    for key in CLEAR_FLAGS:
        env_base.pop(key, None)
    env_base.update({
        "RVC_CONV_WEIGHT_PREPACK": "1", "RVC_NSF_IMPLICIT_GEMM": "1",
        "RVC_RMVPE_GRU_PERSISTENT": "1",
    })
    rows, failures, parities = [], [], []
    try:
        nvml = subprocess.run(["nvidia-smi", "-q"], capture_output=True, timeout=15, check=False)
        (out / "nvidia-smi.txt").write_bytes(nvml.stdout + nvml.stderr)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    for core, batch in grid:
        for mode in modes:
            case_id = f"c{core}_b{batch}_{mode}"
            wav = out / f"{case_id}.wav"
            env = env_base.copy()
            flag, expected = MODES[mode]
            if flag:
                env[flag] = "1"
            command = [
                str(paths["exe"]), "infer-wav-full-benchmark", str(paths["voice"]),
                str(paths["hubert"]), str(paths["rmvpe"]),
                str(paths["index"]) if args.index_rate > 0 else "-",
                str(paths["input_wav"]), str(wav), format(args.index_rate, ".8g"),
                "cuda", str(args.runs), str(core), str(batch),
            ]
            print(f"Stage79: {case_id} ({args.runs} full passes)", flush=True)
            t0 = time.monotonic()
            try:
                result = subprocess.run(command, cwd=base, env=env, capture_output=True)
                (out / f"{case_id}.stdout.log").write_bytes(result.stdout)
                (out / f"{case_id}.stderr.log").write_bytes(result.stderr)
                if result.returncode:
                    raise RuntimeError(f"EXE exit {result.returncode}")
                report = json.loads(Path(f"{wav}.benchmark.json").read_text(encoding="utf-8-sig"))
                if (
                    report.get("rmvpe_unet_conv_execution") != expected
                    or report.get("backend") != "cuda"
                    or report.get("graph_replay") is not False
                    or report.get("nsf_execution") != "sequential"
                    or report.get("nsf_conv_execution") != "implicit_gemm"
                    or report.get("rmvpe_gru_execution") != "persistent"
                    or len(report.get("inference_ms", [])) != args.runs
                    or not report.get("iterations_exact_f32")
                ):
                    raise RuntimeError("Benchmark policy mismatch, or inference is not deterministic")
                if len(wav_pcm16(wav)) != report["output_samples"]:
                    raise RuntimeError("PCM16 sample count mismatch")
                row = {
                    "id": case_id, "mode": mode, "core_frames": core, "batch_size": batch,
                    "warm_median_ms": report["warm_median_ms"],
                    "warm_rtf": report["warm_rtf"], "first_inference_ms": report["first_inference_ms"],
                    "load_ms": report["load_ms"], "source_seconds": report["source_seconds"],
                    "output_samples": report["output_samples"], "wall_elapsed_ms": (time.monotonic() - t0) * 1000,
                    "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                    "chunk_counts": report["chunk_counts"],
                    "inference_ms": report["inference_ms"],
                    **work_estimate(report["source_seconds"], core),
                }
                rows.append(row)
                print(f"  warm={row['warm_median_ms']:.1f} ms  chunks={row['chunk_counts']}  RTF={row['warm_rtf']:.4f}", flush=True)
                if mode != "direct":
                    ref_id = f"c{core}_b{batch}_direct"
                    ref_file = out / f"{ref_id}.wav"
                    if ref_file.exists():
                        parities.append({"left": ref_id, "right": case_id, **compare_pcm(ref_file, wav)})
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                failures.append({"case": case_id, "error": str(error)})
                print(f"  FAILED: {error}", file=sys.stderr, flush=True)
    (out / "summary.json").write_text(json.dumps({
        "stage": 79, "purpose": "chunk-geometry-and-original-comparison",
        "source_basis": "PR #2 Stage78 and upstream tts-with-rvc-main; no PyTorch timing in this run",
        "device": "cuda", "graph_replay": False, "runs": args.runs,
        "grid": grid, "modes": modes,
        "caveat": "PCM comparisons valid only within identical chunk geometry. Cross-core PCM will differ from context and RNG changes; do not use as hard correctness gate.",
        "cases": rows, "parity": parities, "failures": failures,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "times.csv").open("w", encoding="utf-8", newline="") as dest:
        fields = ["id", "mode", "core_frames", "batch_size", "warm_median_ms", "warm_rtf", "chunks_estimate", "model_seconds_estimate", "relative_to_input", "first_inference_ms", "load_ms"]
        writer = csv.DictWriter(dest, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_zip:
        archive = out.parent / f"RVC_STAGE79_CHUNK_SWEEP_RESULTS.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as destination:
            for file in sorted(out.iterdir()):
                if file.is_file():
                    destination.write(file, file.name)
        print(f"Results: {archive}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
