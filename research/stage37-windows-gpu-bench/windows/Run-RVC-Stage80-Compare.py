#!/usr/bin/env python3
"""Run original PyTorch CUDA and native Rust full-WAV benchmarks without modifying sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


WINDOWS_TORCH_PYTHON = r"D:\Projects\tts-with-rvc\.venv\Scripts\python.exe"
ORIGINAL_SOURCE_SHA256 = {
    "tts_with_rvc/infer/vc/pipeline.py": "8e25cddb6d9a616a1d2f9911146e2a1d1c77900d40421c8d17ebeea16ce1736e",
    "tts_with_rvc/infer/vc/modules.py": "4d3bada97629b2a61a1497cd0c8f92a707e78a355c1dbc2729556ec70fee926a",
    "tts_with_rvc/infer/vc/config.py": "4470db3c5f9b1d3542cf8fdfcdc02d658be2368b10d5dd12c79f16dd1a15c82f",
    "tts_with_rvc/infer/lib/rmvpe.py": "da5d5d311f220a18164b47ce08a48baebf93f4ab6b86941a68d8e73f04d3ba92",
    "tts_with_rvc/infer/lib/infer_pack/models.py": "9a642acdad8edf497afb5afc76ce0c722564a04e9d049605e1096d2585fffa4e",
}


def original_fingerprint(root: Path) -> dict:
    result = {}
    for name, expected in ORIGINAL_SOURCE_SHA256.items():
        source = root / name
        actual = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if source.is_file() else None
        result[name] = {"expected_sha256": expected, "actual_sha256": actual, "matches_user_original_zip": actual == expected}
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute(path: str, base: Path) -> Path:
    item = Path(path)
    return item.resolve() if item.is_absolute() else (base / item).resolve()


def run_case(command: list[str], workdir: Path, name: str, output: Path) -> dict:
    print(f"Stage80 compare: {name}", flush=True)
    result = subprocess.run(command, cwd=workdir, capture_output=True, check=False)
    (output / f"{name}.stdout.log").write_bytes(result.stdout)
    (output / f"{name}.stderr.log").write_bytes(result.stderr)
    if result.returncode:
        print(f"FAILED: {name}, exit {result.returncode}; see logs", file=sys.stderr, flush=True)
    else:
        print(f"SUCCESS: {name}", flush=True)
    return {"name": name, "exit_code": result.returncode, "command_executable": command[0]}


def summarize(result_dir: Path, runs: list[dict]) -> dict:
    source = result_dir / "native" / "summary.json"
    native = json.loads(source.read_text(encoding="utf-8-sig")) if source.is_file() else {}
    original = {}
    for prec in ("fp32", "fp16"):
        path = result_dir / f"torch_{prec}" / "summary.json"
        if path.is_file():
            j = json.loads(path.read_text(encoding="utf-8-sig"))
            original[prec] = {
                "warm_median_ms": j["warm_median_ms"],
                "warm_rtf": j["warm_rtf"],
                "config": j["original_config"],
                "output_samples": j["output_samples"],
                "all_iteration_sha256_equal": len(set(j["sha256"])) == 1,
                "cuda_matmul_allow_tf32": j["cuda_matmul_allow_tf32"],
                "cudnn_allow_tf32": j["cudnn_allow_tf32"],
            }
    comparison = []
    for row in native.get("cases", []):
        for prec, timings in original.items():
            comparison.append({
                "native_case": row["id"], "torch_precision": prec,
                "native_warm_ms": row["warm_median_ms"],
                "torch_warm_ms": timings["warm_median_ms"],
                "torch_over_native_ratio_non_equivalent": timings["warm_median_ms"] / row["warm_median_ms"],
            })
    return {
        "stage": 80, "kind": "reference-vs-native-full-wav-benchmark",
        "note": "Not a matched-graph numerical or latency benchmark: different precision, chunk length, F0 extent, RNG, preprocessing and synchronization. Compare end-to-end engineering throughput, not individual kernels.",
        "execution": runs, "native_cases": native.get("cases", []),
        "native_parity_within_geometry": native.get("parity", []),
        "native_failures": native.get("failures", []),
        "torch": original, "indicative_warm_comparisons": comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-python", default=WINDOWS_TORCH_PYTHON)
    parser.add_argument("--original-root", default="", help="Default: root above .venv/Scripts/python.exe")
    parser.add_argument("--native-exe", default="tts-with-rvc-rs.exe")
    parser.add_argument("--assets-dir", default="embed")
    parser.add_argument("--voice", default="CrazyMita.pth")
    parser.add_argument("--hubert", default="hubert_base.pt")
    parser.add_argument("--rmvpe", default="rmvpe.pt")
    parser.add_argument("--index", default="CrazyMita.index")
    parser.add_argument("--input-wav", default="MitaKind.wav")
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--grid", default="300:2,600:1,900:1")
    parser.add_argument("--native-modes", default="direct,gemm,tf32x2")
    parser.add_argument("--torch-precision", choices=("both", "fp16", "fp32"), default="both")
    parser.add_argument("--out", default="stage80_original_comparison")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    torch_python = absolute(args.torch_python, here)
    original_root = absolute(args.original_root, here) if args.original_root else here / "reference_upstream"
    native_exe = absolute(args.native_exe, here)
    assets = absolute(args.assets_dir, here)
    for key, path in {
        "torch-python": torch_python, "native-exe": native_exe,
        "original-source": original_root / "tts_with_rvc" / "infer" / "vc" / "pipeline.py",
        "voice": assets / args.voice, "hubert": assets / args.hubert,
        "rmvpe": assets / args.rmvpe, "input-wav": assets / args.input_wav,
    }.items():
        if not path.is_file():
            parser.error(f"Missing {key}: {path}")
    if args.index_rate > 0 and not (assets / args.index).is_file():
        parser.error(f"Missing index: {assets / args.index}")
    if not (2 <= args.runs <= 16) or not (0 <= args.index_rate <= 1):
        parser.error("Runs must be 2..16 and index-rate must be 0..1")
    native_script = here / "Run-RVC-Stage80-ChunkSweep.py"
    original_script = here / "Run-RVC-OriginalTorch-Stage80.py"
    for script in (native_script, original_script):
        if not script.is_file():
            parser.error(f"Missing runner: {script}")
    fingerprint = original_fingerprint(original_root)
    if not all(item["matches_user_original_zip"] for item in fingerprint.values()):
        parser.error("Original source fingerprint mismatch. Use the bundled Stage80 reference_upstream/ folder")
    result_dir = absolute(args.out, here)
    if result_dir.exists() and any(result_dir.iterdir()):
        parser.error(f"Output directory must be empty: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    native_out = result_dir / "native"
    native_command = [
        str(torch_python), str(native_script), "--exe", str(native_exe),
        "--voice", str(assets / args.voice), "--hubert", str(assets / args.hubert),
        "--rmvpe", str(assets / args.rmvpe), "--index", str(assets / args.index),
        "--input-wav", str(assets / args.input_wav),
        "--index-rate", str(args.index_rate), "--runs", str(args.runs),
        "--grid", args.grid, "--modes", args.native_modes,
        "--out", str(native_out), "--no-zip",
    ]
    runs.append(run_case(native_command, here, "native", result_dir))
    precisions = ("fp32", "fp16") if args.torch_precision == "both" else (args.torch_precision,)
    for precision in precisions:
        original_command = [
            str(torch_python), str(original_script),
            "--original-root", str(original_root), "--assets-dir", str(assets),
            "--voice", args.voice, "--hubert", args.hubert, "--rmvpe", args.rmvpe,
            "--index", args.index, "--input-wav", args.input_wav,
            "--index-rate", str(args.index_rate), "--runs", str(args.runs),
            "--precision", precision, "--output", str(result_dir / f"torch_{precision}"),
        ]
        runs.append(run_case(original_command, here, f"torch_{precision}", result_dir))
    summary = summarize(result_dir, runs)
    summary["original_source_fingerprint"] = original_fingerprint(original_root)
    if not all(item["matches_user_original_zip"] for item in summary["original_source_fingerprint"].values()):
        raise RuntimeError("Original code fingerprint mismatch")
    summary["inputs"] = {
        "source_wav_sha256": sha256(assets / args.input_wav),
        "voice_sha256": sha256(assets / args.voice),
        "hubert_sha256": sha256(assets / args.hubert),
        "rmvpe_sha256": sha256(assets / args.rmvpe),
        "index_sha256": sha256(assets / args.index) if args.index_rate else None,
        "index_rate": args.index_rate, "native_grid": args.grid, "native_modes": args.native_modes,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.no_zip:
        artifact = result_dir.parent / "RVC_STAGE79_ORIGINAL_COMPARISON_RESULTS.zip"
        with zipfile.ZipFile(artifact, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
            for item in sorted(result_dir.rglob("*")):
                if item.is_file():
                    zip_file.write(item, item.relative_to(result_dir))
        print(f"Benchmark archive: {artifact}", flush=True)
    return int(any(x["exit_code"] for x in runs) or bool(summary["native_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
