import csv
import os
import sys
import tempfile
import zipfile
from pathlib import Path

SDK = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\extras\python")
if not SDK.is_dir():
    raise SystemExit(f"Nsight Compute Python SDK not found: {SDK}")
sys.path.insert(0, str(SDK))
if hasattr(os, "add_dll_directory"):
    _dll_handle = os.add_dll_directory(str(SDK))
import ncu_report


def write_metrics(report_path: Path, destination: Path) -> None:
    report = ncu_report.load_report(str(report_path))
    rows = []
    metric_names = []
    seen = set()
    for range_index in range(report.num_ranges()):
        section = report.range_by_idx(range_index)
        for action_index in range(section.num_actions()):
            action = section.action_by_idx(action_index)
            row = {
                "Range": range_index,
                "ActionID": action_index,
                "KernelName": action.name(),
            }
            for name in action.metric_names():
                if name not in seen:
                    metric_names.append(name)
                    seen.add(name)
                try:
                    value = action.metric_by_name(name).value()
                    row[name] = value if isinstance(value, (int, float, str)) else str(value)
                except Exception:
                    row[name] = ""
            rows.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Range", "ActionID", "KernelName", *metric_names])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} kernels, {len(metric_names)} metrics -> {destination}")
    for row in rows:
        print(f"  Range {row['Range']} Action {row['ActionID']}: {row['KernelName']}")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python Export-Stage121-NCU-Metrics.py <stage121.zip|resblock-kernels.ncu-rep> [output.csv]")
    source = Path(sys.argv[1]).resolve()
    if not source.is_file():
        raise SystemExit(f"Cannot find report: {source}")
    destination = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else source.with_name("stage121_ncu_metrics.csv")
    if source.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="stage121_ncu_") as temp:
            with zipfile.ZipFile(source) as archive:
                matches = [name for name in archive.namelist() if name.endswith(".ncu-rep")]
                if len(matches) != 1:
                    raise SystemExit(f"Expected one NCU report in ZIP, got {len(matches)}")
                local = Path(temp) / "capture.ncu-rep"
                local.write_bytes(archive.read(matches[0]))
            write_metrics(local, destination)
    else:
        write_metrics(source, destination)


if __name__ == "__main__":
    main()
