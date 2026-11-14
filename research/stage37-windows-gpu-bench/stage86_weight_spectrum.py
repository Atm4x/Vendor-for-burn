"""Inspect rank/compute tradeoffs of RVC NSF Conv1d weights without modifying them."""
import argparse
import json
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    state = torch.load(args.voice, map_location="cpu", weights_only=True)["weight"]
    rows = []
    for block in range(12):
        for family in ("convs1", "convs2"):
            for layer in range(3):
                prefix = f"dec.resblocks.{block}.{family}.{layer}"
                v = state[f"{prefix}.weight_v"].float()
                c_out, c_in, kernel = v.shape
                if c_out > 128:
                    continue
                gain = state[f"{prefix}.weight_g"].float()
                normalized = v * (gain / v.flatten(1).norm(dim=1).reshape(-1, 1, 1))
                singular = torch.linalg.svdvals(normalized.reshape(c_out, -1))
                energy = singular.square()
                fractions = energy.cumsum(0) / energy.sum()
                def rank(retained_energy):
                    return int((fractions < retained_energy).sum().item()) + 1
                rows.append({
                    "layer": prefix, "channels": c_out, "input_channels": c_in, "kernel": kernel,
                    "full_multiplies": c_out * c_in * kernel,
                    "rank_99_percent": rank(0.99), "rank_95_percent": rank(0.95),
                })
    groups = []
    for channels in sorted({row["channels"] for row in rows}):
        subset = [row for row in rows if row["channels"] == channels]
        baseline = sum(row["full_multiplies"] for row in subset)
        def compressed_fraction(key):
            return baseline / sum(row[key] * (row["input_channels"] * row["kernel"] + row["channels"]) for row in subset)
        groups.append({
            "channels": channels, "layers": len(subset),
            "mean_rank_99": statistics.mean(row["rank_99_percent"] for row in subset),
            "mean_rank_95": statistics.mean(row["rank_95_percent"] for row in subset),
            "compute_reduction_99": compressed_fraction("rank_99_percent"),
            "compute_reduction_95": compressed_fraction("rank_95_percent"),
        })
    out = {"groups": groups, "method": "singular-value-energy of normalized Conv1d [Cout,Cin*K], rank-aware FLOP ratio"}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
