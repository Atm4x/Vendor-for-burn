"""Analyze repeated resident RVC benchmarks and PCM16 output parity."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import wave
from array import array
from collections import defaultdict
from pathlib import Path


def read_pcm16(path: Path) -> tuple[int, array]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM: {path}")
        sample_rate = handle.getframerate()
        data = handle.readframes(handle.getnframes())
    values = array("h")
    values.frombytes(data)
    if __import__("sys").byteorder != "little":
        values.byteswap()
    return sample_rate, values


def pcm_metrics(reference: Path, target: Path) -> dict:
    rate_a, a = read_pcm16(reference)
    rate_b, b = read_pcm16(target)
    if len(a) != len(b) or rate_a != rate_b:
        return {"compatible": False, "sample_count_a": len(a),
                "sample_count_b": len(b), "sample_rate_a": rate_a,
                "sample_rate_b": rate_b}
    if not a:
        raise ValueError("Empty comparison WAV")
    sum_err_sq = 0
    sum_signal_sq = 0
    max_err = 0
    unequal = 0
    for va, vb in zip(a, b):
        difference = int(va) - int(vb)
        unequal += difference != 0
        sum_err_sq += difference * difference
        sum_signal_sq += int(va) * int(va)
        max_err = max(max_err, abs(difference))
    rmse = math.sqrt(sum_err_sq / len(a))
    return {
        "compatible": True,
        "sample_count": len(a),
        "unequal_samples": unequal,
        "rmse_pcm16": rmse,
        "max_abs_pcm16": max_err,
        "snr_db": (10 * math.log10(sum_signal_sq / sum_err_sq)
                   if sum_err_sq > 0 and sum_signal_sq > 0 else None),
    }


def analyze(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    order = manifest["order"]
    if manifest.get("input_duration_seconds", 0) < 60:
        raise ValueError("Stage101 long-WAV gate requires input >=60s")
    measurements = []
    for number, mode in enumerate(order, 1):
        label = f"{number:02d}_{mode}"
        path = root / f"{label}.wav.benchmark.json"
        sample = json.loads(path.read_text(encoding="utf-8-sig"))
        if (sample.get("backend") != "cuda" or
            sample.get("tensor_batch_size") != manifest["tensor_batch_size"] or
            sample.get("core_frames") != manifest["core_frames"] or
            sample.get("iterations_exact_f32") is not True):
            raise ValueError(f"Unmatched runtime or nonrepeatable output: {label}")
        times = sample["inference_ms"]
        if len(times) != manifest["runs_per_process"]:
            raise ValueError(f"Unexpected benchmark repetition count: {label}")
        expected = {
            "reference_direct":"direct", "candidate_direct":"direct", "fused_direct":"direct",
            "fused_spatial":"spatial_quad",
            "fused_winograd":"winograd_f2x2_3x3",
            "fused_tf32x2_batched":"compensated_tf32x2_batched",
        }[mode]
        if sample.get("rmvpe_unet_conv_execution") != expected:
            raise ValueError(f"Runtime ignored requested {mode} setting: {label}")
        expected_bn = mode.startswith("fused_")
        actual_bn = sample.get("rmvpe_unet_bn_folded", False)
        if actual_bn is not expected_bn or (mode != "reference_direct" and "rmvpe_unet_bn_folded" not in sample):
            raise ValueError(f"BatchNorm folding inactive or mismatched: {label}")
        measurements.append({"label":label,"mode":mode,"warm_ms":times[1:],
                             "warm_median_ms":statistics.median(times[1:]),
                             "first_ms":times[0],
                             "chunk_counts":sample.get("chunk_counts"),
                             "planned_cut_frames":sample.get("planned_cut_frames"),
                             "wav":str(root/f"{label}.wav")})
    references = [r for r in measurements if r["mode"] == "reference_direct"]
    if not references:
        raise ValueError("Baseline missing")
    baseline = references[0]
    baseline_cut = baseline["planned_cut_frames"]
    baseline_chunks = baseline["chunk_counts"]
    for sample in measurements:
        if (sample["planned_cut_frames"] != baseline_cut or
            sample["chunk_counts"] != baseline_chunks):
            raise ValueError(f"Different chunking: {sample['label']}")
        sample["pcm_vs_direct"] = pcm_metrics(Path(baseline["wav"]), Path(sample["wav"]))
        if not sample["pcm_vs_direct"]["compatible"]:
            raise ValueError(f"Incompatible WAV lengths: {sample['label']}")
    by_mode = defaultdict(list)
    for entry in measurements:
        by_mode[entry["mode"]].extend(entry["warm_ms"])
    baseline_ms = statistics.median(by_mode["reference_direct"])
    modes = {}
    for mode, durations in by_mode.items():
        med = statistics.median(durations)
        entries = [e for e in measurements if e["mode"] == mode]
        modes[mode] = {
            "warm_samples_ms":durations,
            "median_ms":med,
            "min_ms":min(durations),
            "max_ms":max(durations),
            "latency_change_pct_vs_direct":100 * (med/baseline_ms - 1),
            "speedup_x_vs_direct":baseline_ms/med,
            "pcm_vs_direct":entries[0]["pcm_vs_direct"],
        }
    torch_ms = manifest.get("torch_fp16_reference_ms")
    for mode, item in modes.items():
        item["vs_torch_fp16_ratio"] = item["median_ms"] / torch_ms if torch_ms and torch_ms > 0 else None
    return {
        "stage":101, "manifest":manifest,
        "baseline_warm_median_ms":baseline_ms,
        "modes":modes, "runs":measurements,
        "provisional_1_2x_goal_met": (
            torch_ms is not None and modes["fused_spatial"]["median_ms"] <= 1.2 * torch_ms
        ) if "fused_spatial" in modes else False,
        "warning": "Historical PyTorch FP16 baseline is not a paired comparison; only matched PyTorch and Rust runs can confirm the 1.2x goal. GPU measurements only.",
    }


def write_report(result: dict, output: Path) -> None:
    output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    lines = ["# Stage 101 — Long WAV RVC algorithm + BN fusion A/B", "",
             "Full RVC resident long-WAV benchmark; fixed plan, CUDA, identical model files and index.",
             "First run of each process is excluded; all subsequent warm times are included.",
             "", "| Mode | Warm median ms | Latency change vs direct | RMSE PCM16 | Max abs PCM16 |",
             "|---|---:|---:|---:|---:|"]
    for mode, item in result["modes"].items():
        parity = item["pcm_vs_direct"]
        lines.append(f"| {mode} | {item['median_ms']:.3f} | {item['latency_change_pct_vs_direct']:+.2f}% | {parity['rmse_pcm16']:.3f} | {parity['max_abs_pcm16']} |")
    torch_ms = result["manifest"].get("torch_fp16_reference_ms")
    if torch_ms:
        lines.extend(["", f"PyTorch FP16 reference: {torch_ms:.3f} ms (historical, not a matched CUDA-operator workload).", "",
                      "| Mode | Rust / PyTorch ratio |", "|---|---:|"])
        for mode, item in result["modes"].items():
            lines.append(f"| {mode} | {item['vs_torch_fp16_ratio']:.3f}x |")
    lines.extend(["", "A smaller RMSE does not prove perceptual parity; evaluate generated audio.",
                  "Only long input (>60s) is eligible for the stage101 performance target.",
                  "PyTorch ratio from historical benchmark is indicative, not strict workload parity.",
                  "For each candidate, verify identical chunk boundaries and output length."])
    (output/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.input)
    write_report(result,args.output)
    print((args.output/"comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
