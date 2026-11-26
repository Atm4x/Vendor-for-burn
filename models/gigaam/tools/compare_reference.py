"""Compare Rust diagnostic float32 dumps against original Python GigaAM."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="reference_dir", type=Path, required=True)
    parser.add_argument("--rust", dest="rust_dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads((args.reference_dir / "reference.json").read_text(encoding="utf-8"))
    results = {}
    for name in ("mel", "encoder", "logits"):
        source = args.reference_dir / f"{name}.f32"
        actual = args.rust_dir / f"{name}.f32"
        if not source.is_file() or not actual.is_file():
            continue
        reference = np.fromfile(source, dtype="<f4").astype(np.float64)
        rust = np.fromfile(actual, dtype="<f4").astype(np.float64)
        shape = metadata[f"{name}_shape"]
        expected = int(np.prod(shape))
        if len(reference) != expected or len(rust) != expected:
            results[name] = {"shape": shape, "reference_values": len(reference), "rust_values": len(rust), "shape_match": False}
            continue
        diff = rust - reference
        mse = float(np.mean(np.square(diff)))
        ref_mse = float(np.mean(np.square(reference)))
        results[name] = {
            "shape": shape,
            "shape_match": True,
            "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(mse)),
            "snr_db": float(10 * np.log10(ref_mse / mse)) if mse > 0 and ref_mse > 0 else None,
            "exact_match": bool(mse == 0.0),
        }
    if not results:
        parser.error("No matching .f32 files between reference and Rust directories")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(not result["shape_match"] for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
