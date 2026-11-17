"""Export every Nsight Compute action/metric via NVIDIA's Python SDK, not CLI --page raw."""

import argparse
import csv
import os
import sys
from pathlib import Path


def export(report_path: Path, output: Path, sdk_dir: Path) -> int:
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    if not sdk_dir.is_dir():
        raise FileNotFoundError(f"Nsight Compute Python SDK not found at {sdk_dir}")
    sys.path.insert(0, str(sdk_dir))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(sdk_dir))
    import ncu_report

    report = ncu_report.load_report(str(report_path))
    fields = ["Range", "ActionID", "KernelName"]
    unique = set(fields)
    actions = []
    for r_idx in range(report.num_ranges()):
        target_range = report.range_by_idx(r_idx)
        for a_idx in range(target_range.num_actions()):
            action = target_range.action_by_idx(a_idx)
            row = {"Range": r_idx, "ActionID": a_idx, "KernelName": action.name()}
            for metric in action.metric_names():
                if metric not in unique:
                    fields.append(metric)
                    unique.add(metric)
                try:
                    row[metric] = action.metric_by_name(metric).value()
                except (TypeError, ValueError, RuntimeError):
                    row[metric] = ""
            actions.append(row)
    if not actions or any("bigru_" not in str(row["KernelName"]) for row in actions):
        raise ValueError("NCU did not capture exclusively BiGRU kernels")
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(actions)
    print(f"NCU: {len(actions)} GRU actions; {len(fields)-3} metrics; {output}")
    return len(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sdk-dir", required=True, type=Path)
    args = parser.parse_args()
    export(args.report, args.output, args.sdk_dir)


if __name__ == "__main__":
    main()
