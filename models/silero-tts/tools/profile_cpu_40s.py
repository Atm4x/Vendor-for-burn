"""Bounded, reproducible perf sampling of a persistent Silero CubeCL CPU worker."""

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = [
        {"command": "capabilities", "request_id": "cap"},
        {"command": "load", "request_id": "load", "model_dir": str(args.model_dir.resolve()), "backend": "cpu"},
        {
            "command": "infer", "request_id": "infer",
            "input": {"kind": "accented", "text": args.text},
            "speaker": args.speaker, "sample_rate": args.sample_rate,
            "output": {"kind": "wav_file", "path": str(out / "output.wav")},
        },
        {"command": "status", "request_id": "status"},
        {"command": "shutdown", "request_id": "shutdown"},
    ]
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode()
    command = [str(args.perf), "record", "-e", "cpu-clock:u", "-F", "99", "-o", str(out / "cpu.data"), "--", str(args.binary.resolve()), "--stdio"]
    env = os.environ.copy()
    env["DEBUGINFOD_URLS"] = ""
    if args.skip_cse:
        env["SILERO_SKIP_SIMPLE_CSE"] = "1"
    else:
        env.pop("SILERO_SKIP_SIMPLE_CSE", None)
    started = time.monotonic()
    with (out / "worker.jsonl").open("wb") as stdout, (out / "perf-stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        assert process.stdin is not None
        process.stdin.write(payload)
        process.stdin.close()
        timed_out = False
        try:
            process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=4)
    elapsed = time.monotonic() - started
    reports = [
        ("dso.txt", ["--sort", "dso", "--percent-limit", "0.1"]),
        ("symbols.txt", ["--sort", "symbol", "--percent-limit", "0.07", "--no-demangle"]),
    ]
    for filename, options in reports:
        try:
            result = subprocess.run(
                [str(args.perf), "report", "--stdio", "--no-children", "-i", str(out / "cpu.data"), *options],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, check=False,
            )
            (out / filename).write_bytes(result.stdout)
            (out / (filename + ".stderr")).write_bytes(result.stderr)
        except subprocess.TimeoutExpired:
            (out / (filename + ".stderr")).write_text("perf report exceeded 20 seconds\n")
    responses = []
    for line in (out / "worker.jsonl").read_text().splitlines():
        try:
            value = json.loads(line)
            responses.append({"request_id": value.get("request_id"), "ok": value.get("ok"), "error": value.get("error"), "sample_count": (value.get("data") or {}).get("sample_count")})
        except json.JSONDecodeError:
            responses.append({"malformed": True})
    perf_text = (out / "perf-stderr.log").read_text(errors="replace")
    match = re.search(r"\((\d+) samples\)", perf_text)
    result = {
        "duration_limit_seconds": args.seconds,
        "elapsed_wall_seconds": round(elapsed, 3),
        "terminated_by_limit": timed_out,
        "perf_returncode": process.returncode,
        "sample_count": int(match.group(1)) if match else None,
        "skip_cse": args.skip_cse,
        "responses": responses,
        "model_dir": str(args.model_dir.resolve()),
        "binary": str(args.binary.resolve()),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--perf", type=Path, default=Path("/usr/bin/perf"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--text", default="а")
    parser.add_argument("--speaker", default="aidar")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--skip-cse", action="store_true")
    run(parser.parse_args())
