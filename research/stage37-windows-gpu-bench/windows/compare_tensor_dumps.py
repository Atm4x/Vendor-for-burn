import argparse
import array
import json
import math
from pathlib import Path


def read_f32(path):
    data = array.array("f")
    with path.open("rb") as stream:
        data.frombytes(stream.read())
    return data


def compare(expected, actual):
    if len(expected) != len(actual):
        raise ValueError("Tensor lengths differ")
    if not expected:
        raise ValueError("Empty tensor")
    max_abs = 0.0
    max_normalized = 0.0
    sum_sq = 0.0
    sum_abs = 0.0
    equal = 0
    for ref, value in zip(expected, actual):
        if not math.isfinite(ref) or not math.isfinite(value):
            raise ValueError("Non-finite tensor element")
        delta = abs(ref - value)
        max_abs = max(max_abs, delta)
        max_normalized = max(max_normalized, delta / (1.0 + abs(ref)))
        sum_abs += delta
        sum_sq += delta * delta
        equal += ref == value
    count = len(expected)
    return {
        "elements": count,
        "max_abs_error": max_abs,
        "max_normalized_error": max_normalized,
        "mean_abs_error": sum_abs / count,
        "rmse": math.sqrt(sum_sq / count),
        "exact_equal_fraction": equal / count,
        "passed": max_normalized <= 0.0002,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare full FP32 GPU output tensors")
    parser.add_argument("--cubecl-dir", type=Path, required=True)
    parser.add_argument("--torch-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for name in ("nsf_k3", "nsf_k11_d5", "nsf_k11"):
        reference = read_f32(args.torch_dir / f"{name}_pytorch.f32le")
        for variant in ("direct", "packed", "quad"):
            candidate = read_f32(args.cubecl_dir / f"{name}_{variant}.f32le")
            entry = {"case": name, "variant": variant, **compare(reference, candidate)}
            results.append(entry)
            print(f"{name}/{variant}: max_abs={entry['max_abs_error']:.8g} rmse={entry['rmse']:.8g} pass={entry['passed']}")
    passed = all(result["passed"] for result in results)
    report = {"status": "pass" if passed else "parity_failed", "reference": "PyTorch TF32 disabled", "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
