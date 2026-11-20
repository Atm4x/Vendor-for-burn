"""Validate GRU CUDA events against the synchronized warm NVTX interval."""

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path


KERNEL_NAME = re.compile(r"bigru_(?:split_|gate_parallel_)?kernel")


def inspect_nsys(path: Path, expected_block: int, expected_kernel: str) -> dict:
    db = sqlite3.connect(str(path))
    try:
        intervals = db.execute(
            "SELECT start, end FROM NVTX_EVENTS "
            "WHERE text='stage102_rust_measured' AND end IS NOT NULL "
            "ORDER BY start"
        ).fetchall()
        if not intervals:
            raise ValueError("No synchronized warm NVTX interval: stage102_rust_measured")
        integrity = db.execute("PRAGMA quick_check(1)").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"NSYS SQLite failed integrity check: {integrity}")
        names = dict(db.execute("SELECT id, value FROM StringIds"))
        warm = []
        for start, end in intervals:
            if start >= end:
                raise ValueError("Invalid warm NVTX interval")
            events = db.execute(
                "SELECT start, end, gridX, gridY, gridZ, blockX, blockY, blockZ, "
                "registersPerThread, staticSharedMemory, dynamicSharedMemory, shortName "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? "
                "AND gridY=2 AND gridZ=1 AND blockX=? AND blockY=1 AND blockZ=1 "
                "ORDER BY start", (start, end, expected_block)
            ).fetchall()
            matched = []
            for event in events:
                *_, name_id = event
                name = names.get(name_id)
                if name is not None and expected_kernel not in name:
                    continue
                matched.append(event)
            if not matched:
                raise ValueError(f"No GRU GPU events in warm interval [{start}, {end}]")
            if not names and any(event[8] is None for event in matched):
                raise ValueError("No StringIds and missing register metadata; cannot identify GRU")
            if not names and any(event[8] != 40 for event in matched):
                raise ValueError("Ambiguous GRU fallback: expected 40 registers/thread")
            if len(matched) != 1:
                raise ValueError(f"Expected one short-segment warm GRU launch, found {len(matched)}")
            warm.append({
                "warm_nvtx_ms": (end-start)/1e6,
                "kernel_count": len(matched),
                "gpu_kernel_sum_ms": round(sum(e[1]-e[0] for e in matched)/1e6, 6),
                "grid_x_values": sorted(set(e[2] for e in matched)),
                "block_x": expected_block,
                "registers_per_thread": sorted(set(e[8] for e in matched)),
                "static_shared_bytes": sorted(set(e[9] for e in matched)),
                "dynamic_shared_bytes": sorted(set(e[10] for e in matched)),
            })
        return {
            "stage": 132,
            "source": path.name,
            "kernel_identity": "CUDA name + launch geometry" if names else "launch geometry only; StringIds unavailable; verify NCU name separately",
            "warm_passes": warm,
            "aggregate_gpu_kernel_ms": round(sum(x["gpu_kernel_sum_ms"] for x in warm), 6),
            "scope": "CUDA GPU kernel durations inside synchronized warm NVTX, not NVTX CPU duration or NCU replay",
        }
    finally:
        db.close()


def inspect_ncu(path: Path, expected_kernel: str) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next((line for line in reader if "KernelName" in line and "ActionID" in line), None)
        if header is None:
            raise ValueError("NCU metric export is empty or lacks KernelName/ActionID")
        rows = [dict(zip(header, row)) for row in reader if len(row) == len(header)]
    if not rows:
        raise ValueError("NCU profile contains no kernel data rows")
    matching = [row for row in rows if expected_kernel in row["KernelName"]]
    if len(matching) != len(rows):
        raise ValueError(f"NCU captured {len(rows)-len(matching)} non-GRU kernels")
    if not matching:
        raise ValueError("NCU did not capture target GRU kernels")
    keys = [
        "gpu__time_duration.avg", "gpu__time_duration.sum", "launch__registers_per_thread",
        "launch__shared_mem_per_block_static", "launch__shared_mem_per_block_dynamic",
        "launch__occupancy_limit_registers", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "sass__inst_executed_local_loads", "sass__inst_executed_local_stores",
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    ]
    return {
        "source": path.name,
        "capture_count": len(matching),
        "kernel_names": sorted(set(row["KernelName"] for row in matching)),
        "replay_not_wall": True,
        "metrics": [{key: row.get(key, "") for key in keys} for row in matching],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys", type=Path)
    parser.add_argument("--ncu", type=Path)
    parser.add_argument("--block-size", type=int, choices=[256,512,768,1024], required=True)
    parser.add_argument("--kernel", choices=["bigru_kernel","bigru_split_kernel","bigru_gate_parallel_kernel"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.nsys and not args.ncu:
        parser.error("Supply --nsys or --ncu")
    result = {"stage": 132, "block_size": args.block_size, "kernel": args.kernel}
    if args.nsys:
        result["nsys"] = inspect_nsys(args.nsys, args.block_size, args.kernel)
    if args.ncu:
        result["ncu"] = inspect_ncu(args.ncu, args.kernel)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
