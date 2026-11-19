"""Validate the opt-in RUAccent frontend against real Silero and RUAccent weights."""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--silero-dir", type=Path, required=True)
    parser.add_argument("--ruaccent-dir", type=Path, required=True)
    parser.add_argument("--onnx-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    report = {"checks": [], "errors": [], "passed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["ORT_DYLIB_PATH"] = str(args.onnx_library.resolve())
    environment["RAYON_NUM_THREADS"] = "4"
    started = time.monotonic()
    stderr_path = args.output.with_suffix(".stderr.log")
    with stderr_path.open("wb") as stderr, subprocess.Popen(
        [str(args.worker.resolve())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
        env=environment,
        bufsize=0,
    ) as worker:
        sequence = 0

        def exchange(command, **payload):
            nonlocal sequence
            sequence += 1
            message = {"command": command, "request_id": f"e2e-{sequence}", **payload}
            worker.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            worker.stdin.flush()
            line = worker.stdout.readline()
            if not line:
                raise AssertionError(f"Worker exited before replying to {command}; status={worker.poll()}")
            response = json.loads(line)
            if response.get("request_id") != message["request_id"]:
                raise AssertionError(f"Request-id mismatch: {response}")
            return response

        def success(command, **payload):
            reply = exchange(command, **payload)
            if not reply["ok"]:
                raise AssertionError(f"{command} rejected: {reply.get('error')}")
            return reply["data"]

        def infer(kind, text, speaker="aidar", rate=8000):
            response = success(
                "infer", input={"kind": kind, "text": text}, speaker=speaker, sample_rate=rate
            )
            raw = base64.b64decode(response["audio"]["data_base64"], validate=True)
            if len(raw) != response["sample_count"] * 4:
                raise AssertionError("Invalid f32 PCM buffer length")
            samples = memoryview(raw).cast("f")
            if not samples or not all(math.isfinite(x) for x in samples):
                raise AssertionError("Empty or non-finite PCM audio")
            return raw, {
                "kind": kind,
                "speaker": speaker,
                "sample_rate": rate,
                "sample_count": response["sample_count"],
                "audio_seconds": round(len(samples) / rate, 6),
                "inference_ms": response["elapsed_ms"],
                "pcm_sha256": hashlib.sha256(raw).hexdigest(),
            }

        def check(name, passed, **details):
            report["checks"].append({"name": name, "passed": passed, **details})
            if not passed:
                raise AssertionError(f"{name} failed: {details}")
            print(f"[PASS] {name}", flush=True)

        try:
            features = success("capabilities")
            check("opt-in capability advertised", "ruaccent" in features.get("optional_text_providers", []))
            success("load", backend="flex", model_dir=str(args.silero_dir.resolve()),
                    ruaccent_dir=str(args.ruaccent_dir.resolve()), ruaccent_omograph_model="tiny")

            for phrase, explicitly_stressed in [
                ("На двери висит замок.", "На дв+ери вис+ит зам+ок."),
                ("Он все понял и ушел.", "+Он всё п+онял и уш+ёл."),
            ]:
                a, info = infer("ruaccent_text", phrase)
                b, _ = infer("accented", explicitly_stressed.lower())
                check("RUAccent PCM equals explicit accents: " + phrase, a == b, **info)
                repeated, _ = infer("ruaccent_text", phrase)
                check("same-session deterministic PCM: " + phrase, a == repeated)

            plain_before, before_info = infer("text", "На двери замок.")
            for speaker, rate in [("baya", 24000), ("eugene", 48000)]:
                audio, info = infer("ruaccent_text", "На двери висит замок.", speaker, rate)
                check(f"real PCM at {rate}Hz", bool(audio), **info)

            long_text = ("На двери висит замок. Он все понял и ушел. " * 17).strip()
            audio, info = infer("ruaccent_text", long_text)
            check("single-request long-form RUAccent PCM", info["audio_seconds"] > 40, **info)

            success("unload")
            success("load", backend="flex", model_dir=str(args.silero_dir.resolve()))
            plain_after, after_info = infer("text", "На двери замок.")
            check("default text PCM unchanged when RUAccent is loaded", plain_before == plain_after,
                  before=before_info, after=after_info)

            declined = exchange("infer", input={"kind": "ruaccent_text", "text": "На двери замок."},
                                speaker="aidar", sample_rate=8000)
            check("RUAccent requires explicit loaded model", declined["ok"] is False
                  and "RUAccent not loaded" in declined.get("error", ""))
            recovered, _ = infer("text", "На двери замок.")
            check("worker survives missing-provider failure", recovered == plain_after)
            success("shutdown")
            worker.wait(timeout=20)
            check("clean worker exit", worker.returncode == 0)
            report["passed"] = True
        except Exception as error:
            report["errors"].append(str(error))
            worker.kill()
            worker.wait(timeout=20)
            raise
        finally:
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            report["worker_exit_code"] = worker.returncode
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"PASS {len(report['checks'])} real-model checks. Report: {args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
