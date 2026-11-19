#!/usr/bin/env python3
"""Benchmark repeated Silero Flex inference through the persistent JSONL worker."""

import argparse
import base64
import json
import os
import selectors
import statistics
import subprocess
import sys
import time
from pathlib import Path


def exchange(process, selector, request, timeout):
    started = time.perf_counter()
    process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
    process.stdin.flush()
    if not selector.select(timeout):
        raise TimeoutError(f"Worker timed out on {request['command']}")
    payload = process.stdout.readline()
    if not payload:
        raise RuntimeError(f"Worker closed stdout: exit code {process.poll()}")
    elapsed = time.perf_counter() - started
    result = json.loads(payload)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Unknown worker failure"))
    return result["data"], elapsed


def run(args):
    env = os.environ.copy()
    if args.threads is not None:
        env["RAYON_NUM_THREADS"] = str(args.threads)
    if args.exact_gelu:
        env["SILERO_EXACT_GELU"] = "1"
    if args.stages:
        env["SILERO_TIMING"] = "1"
    process = subprocess.Popen(
        [str(args.binary.resolve())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=0,
        env=env,
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        capabilities, _ = exchange(process, selector, {"command": "capabilities", "request_id": "cap"}, args.timeout)
        if "flex" not in capabilities["available_backends"]:
            raise RuntimeError("Benchmark binary does not contain the Flex backend")
        _, load_elapsed = exchange(process, selector, {
            "command": "load", "request_id": "load", "model_dir": str(args.model_dir.resolve()), "backend": "flex"
        }, args.timeout)
        runs = []
        last_pcm = None
        for index in range(args.repeats):
            data, total = exchange(process, selector, {
                "command": "infer", "request_id": f"infer-{index}",
                "input": {"kind": "text", "text": args.text},
                "speaker": args.speaker, "sample_rate": args.sample_rate,
                "output": {"kind": "inline_f32le"},
            }, args.timeout)
            pcm = base64.b64decode(data["audio"]["data_base64"])
            last_pcm = pcm
            if len(pcm) != data["sample_count"] * 4:
                raise RuntimeError("PCM length does not match declared sample count")
            seconds = data["sample_count"] / args.sample_rate
            seconds_elapsed = data["elapsed_ms"] / 1000
            record = {
                "index": index, "sample_count": data["sample_count"],
                "audio_seconds": seconds, "inference_seconds": seconds_elapsed,
                "external_seconds": total, "rtf": seconds_elapsed / seconds,
                "audio_seconds_per_compute_second": seconds / seconds_elapsed,
            }
            runs.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        status, _ = exchange(process, selector, {"command": "status", "request_id": "status"}, args.timeout)
        exchange(process, selector, {"command": "shutdown", "request_id": "shutdown"}, args.timeout)
        process.wait(timeout=10)
        warm = runs[1:] if len(runs) > 1 else runs
        report = {
            "backend": "flex", "build_profile": "release", "load_seconds": load_elapsed,
            "model_dir": str(args.model_dir.resolve()), "speaker": args.speaker,
            "text": args.text, "sample_rate": args.sample_rate,
            "runs": runs, "warm_median_rtf": statistics.median(r["rtf"] for r in warm),
            "warm_median_inference_seconds": statistics.median(r["inference_seconds"] for r in warm),
            "status": status, "worker_exit_code": process.returncode,
            "rayon_threads": args.threads, "exact_gelu": args.exact_gelu,
        }
        if args.reference_pt:
            report["pytorch"] = compare_pytorch(args, last_pcm)
        return report
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        selector.close()


def compare_pytorch(args, pcm):
    import math
    from array import array
    import torch
    from torch.package import PackageImporter

    torch.set_num_threads(args.threads or 4)
    model = PackageImporter(str(args.reference_pt)).load_pickle("tts_models", "model")
    times = []
    with torch.inference_mode():
        for _ in range(args.repeats):
            start = time.perf_counter()
            expected = model.apply_tts(
                text=args.text, speaker=args.speaker, sample_rate=args.sample_rate
            ).detach().float().cpu().reshape(-1).tolist()
            times.append(time.perf_counter() - start)
    actual = array("f")
    actual.frombytes(pcm)
    if len(actual) != len(expected):
        raise ValueError("Reference and optimized waveform lengths differ")
    errors = [(a - b) for a, b in zip(actual, expected)]
    max_abs = max(abs(x) for x in errors)
    rmse = math.sqrt(sum(x * x for x in errors) / len(errors))
    warm = times[1:] if len(times) > 1 else times
    return {"warm_median_seconds": statistics.median(warm), "times": times,
            "max_absolute_error": max_abs, "rmse": rmse, "sample_count": len(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", default="Привет, мир!")
    parser.add_argument("--speaker", default="aidar")
    parser.add_argument("--sample-rate", type=int, default=8000, choices=[8000, 24000, 48000])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--exact-gelu", action="store_true")
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--reference-pt", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be at least 1")
    report = run(args)
    print("SUMMARY " + json.dumps(report, ensure_ascii=False), flush=True)
    if args.output_json:
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
