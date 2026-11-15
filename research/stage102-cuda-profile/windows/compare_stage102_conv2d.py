"""Validate matched Conv2d contracts and compare synchronized CPU/GPU timings."""

import argparse
import json
import math
from pathlib import Path


def compare(rust, torch, *, atol=0.005, rtol=0.002):
    if rust.get("engine") != "burn" or torch.get("engine") != "torch":
        raise ValueError("Expected Burn and PyTorch results")
    if rust.get("backend") != torch.get("backend"):
        raise ValueError("Backend mismatch")
    rust_cases = {c["name"]: c for c in rust["cases"]}
    torch_cases = {c["name"]: c for c in torch["cases"]}
    if set(rust_cases) != set(torch_cases):
        raise ValueError("Different case sets")
    if len(rust_cases) != len(rust["cases"]) or len(torch_cases) != len(torch["cases"]):
        raise ValueError("Duplicate case names")
    records = []
    for name, a in rust_cases.items():
        b = torch_cases[name]
        for key in ("shape", "weight_shape", "stride", "padding", "dilation", "groups", "output_elements"):
            if a[key] != b[key]:
                raise ValueError(f"{name}: incompatible {key}: {a[key]} != {b[key]}")
        if a.get("dtype") != b.get("dtype") or a.get("layout") != b.get("layout"):
            raise ValueError(f"{name}: precision/layout mismatch; do not compare FP32 and FP16 or NCHW and NHWC as matched")
        av, bv = a["sample_values"], b["sample_values"]
        if len(av) != len(bv) or not av:
            raise ValueError(f"{name}: invalid output sample probes")
        errors = [abs(x - y) for x, y in zip(av, bv)]
        correct = all(math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, abs_tol=atol, rel_tol=rtol)
                      for x, y in zip(av, bv))
        if not (math.isfinite(a["mean"]) and math.isfinite(b["mean"])):
            correct = False
        if not math.isclose(a["mean"], b["mean"], abs_tol=atol, rel_tol=rtol):
            correct = False
        rust_ms, torch_ms = float(a["median_ms"]), float(b["median_ms"])
        if not all(math.isfinite(t) and t > 0 for t in (rust_ms, torch_ms)):
            raise ValueError(f"{name}: invalid timings")
        records.append({
            "name": name, "burn_ms": rust_ms, "torch_ms": torch_ms,
            "burn_over_torch": rust_ms / torch_ms,
            "max_probe_abs_error": max(errors), "probe_correct": correct,
            "precision": a["dtype"], "layout": a["layout"],
            "remark": "Synchronized per-call host time incl. launch overhead; GPU kernel duration requires NSYS/NCU",
        })
    return {"stage": 102, "matched_inputs_and_operator": True,
            "all_sample_probes_correct": all(r["probe_correct"] for r in records),
            "cases": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--torch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.rust.read_text(encoding="utf-8-sig")),
                     json.loads(args.torch.read_text(encoding="utf-8-sig")))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["all_sample_probes_correct"]:
        raise SystemExit("Numerical mismatch: see comparison JSON")


if __name__ == "__main__":
    main()
