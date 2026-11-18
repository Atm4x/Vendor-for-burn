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
                except ProcessLookupErro