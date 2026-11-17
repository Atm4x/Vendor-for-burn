"""Summarize full CUDA event coverage and targeted NCU hardware counters without conflating replay and wall time."""

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path

METRICS = {
    "replay_kernel_duration_ns": "gpu__time_duration.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "active_warps_per_sm_cycle": "sm__warps_active.avg.per_cycle_active",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex_throughput_pct": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_throughput_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "long_scoreboard_stall_per_issue": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "local_loads": "sass__inst_executed_local_loads",
    "local_stores": "sass__inst_executed_local_stores",
    "grid_x": "launch__grid_size",
    "block_x": "launch__block_size",
}


def as_number(value):
    if value in (None, "", "N/A", "nan"):
        return None
    try:
        return float(value.replace(",", ""))
    except (ValueError, AttributeError):
        return value


def ncu(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        selected.append({
            "kernel_name": row.get("KernelName", ""),
            "metrics": {display: as_number(row.get(full)) for display, full in METRICS.items()},
        })
    return dict(kind="ncu", source=path.name, replay_not_wall=True,
                selected_kernels=len(rows), kernels=selected,
                warning="Review launch geometry and kernel names: NCU replay is NOT warm duration")


def sqlite_timeline(path):
    db = sqlite3.connect(str(path))
    try:
        kernels = db.execute("SELECT start,end,streamId,shortName,gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
        string_count = db.execute("SELECT COUNT(*) FROM StringIds").fetchone()[0]
        names = {}
        if string_count:
            names = dict(db.execute("SELECT id,value FROM StringIds"))
        duration_ns = sum(end-start for start,end,*_ in kernels)
        by_kernel = {}
        for start,end,stream,name_id,gx,gy,gz,bx,by,bz,reg in kernels:
            name = names.get(name_id, "<missing-string-id:{}>".format(name_id))
            key=(name,gx,gy,gz,bx,by,bz,reg)
            item=by_kernel.setdefault(key,{"name":name,"grid":[gx,gy,gz],"block":[bx,by,bz],"registers":reg,"count":0,"gpu_duration_ms":0.0})
            item["count"]+=1
            item["gpu_duration_ms"]+=(end-start)/1e6
        ranked=sorted(by_kernel.values(),key=lambda x:-x["gpu_duration_ms"])
        return dict(kind="nsys",source=path.name,kernel_launches=len(kernels),streams=len({x[2] for x in kernels}),gpu_kernel_sum_ms=duration_ns/1e6,
                    gpu_activity_span_ms=(max((x[1] for x in kernels),default=0)-min((x[0] for x in kernels),default=0))/1e6,
                    string_table_entries=string_count,
                    warn_no_names=string_count==0,
                    top_kernels=ranked[:25],
                    warning="CUDA sum is not elapsed latency; overlapping kernels and host gaps are not included")
    finally:
        db.close()


def nsys_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle)
        rows=list(reader)
    if not rows:
        return dict(kind="nsys_summary",source=path.name,warning="CSV empty")
    names=rows[0].keys()
    name_key=next((x for x in names if x.lower().strip()=="name"),None)
    total_key=next((x for x in names if re.search(r"total.*time",x,re.I)),None)
    return dict(kind="nsys_summary",source=path.name,headers=list(names),
                top=[{"kernel_name":r.get(name_key),"total_time":r.get(total_key)} for r in rows[:30]],
                warning=None if name_key and total_key else "Unknown NSYS summary format; inspect original CSV")


def analyze(folder):
    rows=[]
    for path in sorted(folder.rglob("*.ncu.csv")):
        try:rows.append(ncu(path))
        except Exception as e:rows.append({"source":path.name,"error":str(e)})
    for path in sorted(folder.rglob("*.nsys.sqlite")):
        try:rows.append(sqlite_timeline(path))
        except Exception as e:rows.append({"source":path.name,"error":str(e)})
    for path in sorted(folder.rglob("*cuda_gpu_kern_sum*.csv")):
        try:rows.append(nsys_csv(path))
        except Exception as e:rows.append({"source":path.name,"error":str(e)})
    return dict(stage=127,profiles=rows,
                gpu_analysis_contract="NSYS event timeline shows real kernel GPU durations. NCU uses replay; do not divide warm elapsed by NCU sum.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    result=analyze(args.directory)
    dest=args.directory/"nsight_analysis.json"
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Nsight analysis:",dest)


if __name__=="__main__":main()
