"""Cross-backend full-tensor FP32 parity and matched wall-time summaries."""
import argparse
import array
import json
import math
import statistics
from pathlib import Path


def read_floats(path):
    values = array.array("f")
    with path.open("rb") as handle:
        values.frombytes(handle.read())
    return values


def validate(reference, candidate):
    if not reference or len(reference) != len(candidate):
        raise ValueError("empty or different length output tensors")
    absmax = normalized = square = 0.0
    exact = 0
    for a, b in zip(reference, candidate):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("nonfinite FP32 tensor")
        error = abs(a - b)
        absmax = max(absmax, error)
        normalized = max(normalized, error / (1 + abs(a)))
        square += error * error
        exact += a == b
    return {
        "elements": len(reference),
        "max_abs_error": absmax,
        "max_normalized_error": normalized,
        "rmse": math.sqrt(square / len(reference)),
        "exact_equal_fraction": exact / len(reference),
        "passed": normalized <= 0.0002,
    }


def mean_of_medians(values):
    return statistics.mean(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = args.results
    names = ["rust_prepacked", "rust_baseline", "torch"]
    data = {}
    for round_no in (1, 2):
        for name in names:
            path = results / f"round{round_no}_{name}.json"
            if not path.is_file():
                parser.error(f"missing report {path}")
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            if name == "torch":
                if report["backend"] != "pytorch-cuda" or report["tf32_allowed"]:
                    raise ValueError("PyTorch GPU baseline unavailable or TF32 enabled")
            else:
                if report["stage"] != 52 or report["backend"] != "cuda" or not report["basic_graph_hardware"]:
                    raise ValueError(f"Rust CUDA Graph not hardware on {path}")
            data[(round_no, name)] = {int(case["frames"]): case for case in report["cases"]}
    frames = sorted(data[(1, "torch")])
    if not frames or any(sorted(cases) != frames for cases in data.values()):
        raise ValueError("frame count mismatch")
    comparisons = []
    for round_no in (1, 2):
        for name in ("rust_prepacked", "rust_baseline"):
            for frames_count in frames:
                case = data[(round_no, name)][frames_count]
                if not case["hardware_graph"] or not case["bitwise_equal"] or case["distinct_outputs"] < 2:
                    raise ValueError(f"Rust correctness failed: round {round_no} {name} {frames_count}")
                for step in range(4):
                    reference = read_floats(results / f"round{round_no}_torch-{frames_count}-{step}.f32le")
                    candidate = read_floats(results / f"round{round_no}_{name}-{frames_count}-{step}.f32le")
                    comparison = validate(reference, candidate)
                    comparisons.append({"round": round_no, "variant": name, "frames": frames_count,
                                        "input_step": step, **comparison})
                    if not comparison["passed"]:
                        raise ValueError(f"FP32 parity failed: {comparisons[-1]}")
    summary = []
    for frames_count in frames:
        means = {}
        for name in names:
            a = data[(1, name)][frames_count]
            b = data[(2, name)][frames_count]
            if name == "torch":
                eager = mean_of_medians([a["median_eager_ms"], b["median_eager_ms"]])
                graph = mean_of_medians([a["median_graph_replay_ms"], b["median_graph_replay_ms"]])
                refresh = mean_of_medians([a["median_refresh_ms"], b["median_refresh_ms"]])
            else:
                eager = mean_of_medians([a["median_normal_ms"], b["median_normal_ms"]])
                graph = mean_of_medians([a["median_replay_ms"], b["median_replay_ms"]])
                refresh = mean_of_medians([a["median_refresh_ms"], b["median_refresh_ms"]])
            stage_sum = [statistics.median([m["refresh_ms"] + (m["graph_replay_and_readback_ms"] if name == "torch" else m["replay_and_readback_ms"]) for m in data[(round_no, name)][frames_count]["measurements"]]) for round_no in (1, 2)]
            means[name] = {"eager_and_readback_ms": eager,
                           "graph_and_readback_ms": graph,
                           "refresh_ms": refresh,
                           "refresh_plus_graph_ms": mean_of_medians(stage_sum),
                           "round1_graph_ms": a["median_graph_replay_ms"] if name == "torch" else a["median_replay_ms"],
                           "round2_graph_ms": b["median_graph_replay_ms"] if name == "torch" else b["median_replay_ms"]}
        rust = means["rust_prepacked"]
        torch = means["torch"]
        summary.append({"frames": frames_count, "variants": means,
                        "torch_over_rust_prepacked_graph_ratio": torch["graph_and_readback_ms"] / rust["graph_and_readback_ms"],
                        "torch_over_rust_prepacked_eager_ratio": torch["eager_and_readback_ms"] / rust["eager_and_readback_ms"],
                        "torch_over_rust_prepacked_refresh_plus_graph_ratio": torch["refresh_plus_graph_ms"] / rust["refresh_plus_graph_ms"]})
    result = {"stage": 53, "status": "pass", "reference": "PyTorch/cuDNN FP32 strict; same synthetic NSF checkpoint and changing input sequence", 
              "caveat": "Separate Python/Rust host runtime costs and distinct memory allocation/copy implementations; ratios are host+readback latency, not GPU kernel-only execution. Refresh+Graph uses per-iteration sums and process-level medians.",
              "parity_comparisons": comparisons, "summary": summary}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for case in summary:
        print(f"{case['frames']} frames: Rust graph {case['variants']['rust_prepacked']['graph_and_readback_ms']:.4f}ms, "
              f"Torch graph {case['variants']['torch']['graph_and_readback_ms']:.4f}ms, "
              f"Torch/Rust = {case['torch_over_rust_prepacked_graph_ratio']:.3f}x")
    print(f"FP32 full-tensor comparison PASS for {len(comparisons)} samples")


if __name__ == "__main__":
    main()
