import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubecl", type=Path, required=True)
    parser.add_argument("--torch", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("comparison.json"))
    args = parser.parse_args()
    cubecl = json.loads(args.cubecl.read_text(encoding="utf-8"))
    torch = json.loads(args.torch.read_text(encoding="utf-8"))
    by_name = {r["name"]: r for r in torch.get("results", [])}
    rows = []
    for row in cubecl.get("results", []):
        reference = by_name.get(row["name"])
        if reference is None:
            continue
        normalized_errors = [abs(a - b) / (1.0 + abs(b)) for a, b in zip(row["direct_probe"], reference["direct_probe"])]
        maximum = max(normalized_errors)
        rows.append({
            "name": row["name"],
            "cubecl_direct_ms": row["direct_ms"],
            "cubecl_packed_ms": row["packed_ms"],
            "pytorch_cudnn_ms": reference["conv_readback_ms"],
            "cubecl_vs_pytorch_direct": reference["conv_readback_ms"] / row["direct_ms"],
            "cubecl_vs_pytorch_packed": reference["conv_readback_ms"] / row["packed_ms"],
            "max_normalized_probe_error": maximum,
            "probe_parity": maximum <= 0.001,
        })
        print(f'{row["name"]}: CubeCL direct {row["direct_ms"]:.3f}ms, packed {row["packed_ms"]:.3f}ms, PyTorch {reference["conv_readback_ms"]:.3f}ms, probe parity {maximum:.2g}')
    result = {"cubecl": cubecl["backend"], "torch": torch["backend"], "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if any(not r["probe_parity"] for r in rows):
        raise SystemExit("Numeric probe mismatch between CubeCL and PyTorch")


if __name__ == "__main__":
    main()
