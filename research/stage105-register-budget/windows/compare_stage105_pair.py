"""Strict matched-case contract for experimental FP32 spatial-pair convolution."""
import argparse
import json
from pathlib import Path


def compare(direct, pair):
    if direct.get("strategy") != "direct" or pair.get("strategy") != "pair":
        raise ValueError("Expected direct/pair strategies")
    if direct.get("backend") != "cuda" or pair.get("backend") != "cuda":
        raise ValueError("Only matched CUDA runs are comparable")
    left = {case["name"]: case for case in direct["cases"]}
    right = {case["name"]: case for case in pair["cases"]}
    if left.keys() != right.keys():
        raise ValueError("Mismatched case sets")
    results = []
    for name, baseline in left.items():
        candidate = right[name]
        for key in ("shape", "weight_shape", "groups", "padding", "stride", "dilation", "dtype", "layout"):
            if baseline[key] != candidate[key]:
                raise ValueError(f"{name}: mismatched {key}")
        supported = candidate["groups"] == 1 and candidate["weight_shape"][2:] == [3, 3]
        if supported:
            parity = candidate.get("full_parity")
            if not parity or not parity.get("passed") or parity["elements"] != candidate["output_elements"]:
                raise ValueError(f"{name}: missing full-output parity")
            if parity["max_abs"] > 1e-4 or parity["max_abs"] < 0:
                raise ValueError(f"{name}: excessive numerical drift")
        else:
            parity = None
        if baseline["median_ms"] <= 0 or candidate["median_ms"] <= 0:
            raise ValueError("Non-positive benchmark duration")
        results.append({"name": name, "direct_ms": baseline["median_ms"], "pair_ms": candidate["median_ms"],
                        "speedup": baseline["median_ms"] / candidate["median_ms"], "full_parity": parity})
    return {"stage": 105, "strategy": "pair", "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.direct.read_text(encoding="utf-8-sig")),
                     json.loads(args.pair.read_text(encoding="utf-8-sig")))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for item in result["results"]:
        print(f"{item['name']}: pair/direct speedup {item['speedup']:.3f}x")


if __name__ == "__main__":
    main()
