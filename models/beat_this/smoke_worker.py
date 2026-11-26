import argparse
import hashlib
import json
import math
import struct
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path


def snapshot(path):
    data = path.read_bytes()
    return {
        "file_name": path.name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def tone(path, frequency):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", ""))
        output.writeframes(
            b"".join(
                struct.pack("<h", round(12000 * math.sin(i * frequency * math.tau / 16000)))
                for i in range(16000)
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="beat_this_small1", choices=("beat_this_small1", "beat_this_final1"))
    parser.add_argument("--flac", type=Path)
    options = parser.parse_args()
    exe = options.exe.resolve(strict=True)
    base = [
        str(exe),
        "--ipc-stdio",
        "--model-dir",
        str(options.model_dir.resolve(strict=True)),
        "--runtime-dir",
        str(options.runtime_dir.resolve(strict=True)),
    ]
    if options.model_id != "beat_this_small1":
        base += ["--model-id", options.model_id]
    base.append("--backend")
    unsupported = subprocess.run(base + ["cuda"], input="", text=True, capture_output=True, timeout=10)
    assert unsupported.returncode != 0 and not unsupported.stdout, unsupported
    with tempfile.TemporaryDirectory(prefix="beat-this-contract-") as directory:
        root = Path(directory)
        first = root / "first.wav"
        other = root / ("second.flac" if options.flac else "second.wav")
        tone(first, 330)
        if options.flac:
            other.write_bytes(options.flac.read_bytes())
        else:
            tone(other, 440)
        good = snapshot(first)
        second = snapshot(other)
        bad = {**good, "sha256": "0" * 64}
        escaped = {**good, "file_name": "../first.wav"}
        schedule = [
            ("capabilities", {}, True),
            ("infer", {"beats_version": 1, "kind": "beats", "input": good}, False),
            ("load", {"model_id": "wrong"}, False),
            ("load", {"model_id": options.model_id}, True),
            ("status", {}, True),
            ("warmup", {}, True),
            ("status", {}, True),
            ("infer", {"beats_version": 1, "kind": "beats", "input": bad}, False),
            ("infer", {"beats_version": 1, "kind": "beats", "input": escaped}, False),
            ("infer", {"beats_version": 1, "kind": "beats", "input": good}, True),
            ("infer", {"beats_version": 1, "kind": "beats", "input": second}, True),
            ("unload", {}, True),
            ("status", {}, True),
            ("shutdown", {}, True),
        ]
        requests = [
            {"version": 1, "id": str(uuid.uuid4()), "command": command, "payload": payload}
            for command, payload, _ in schedule
        ]
        run = subprocess.run(
            base + ["cpu", "--input-dir", str(root)],
            input="".join(json.dumps(request) + "\n" for request in requests),
            encoding="utf-8",
            capture_output=True,
            timeout=240,
        )
        assert run.returncode == 0, run.stderr
        replies = list(map(json.loads, run.stdout.splitlines()))
        assert len(replies) == len(requests), replies
        for request, response, (_, _, expected) in zip(requests, replies, schedule):
            assert response["version"] == 1 and request["id"] == response["id"], response
            assert response["ok"] is expected, (request["command"], response)
        assert replies[0]["result"]["backends"] == ["cpu"]
        assert replies[0]["result"]["beats_contract_version"] == 1
        assert replies[4]["result"]["phase"] == "loaded"
        assert replies[6]["result"]["phase"] == "ready"
        assert replies[6]["result"]["variant"] == "burn_flex_cpu_fp32"
        assert replies[12]["result"]["phase"] == "started"
        for reply in replies[9:11]:
            result = reply["result"]
            assert result["kind"] == "beats" and result["beats_version"] == 1
            assert result["duration_seconds"] > 0 and result["sample_rate_hz"] >= 8000
            for field in ("beats_seconds", "downbeats_seconds"):
                values = result[field]
                assert all(0 <= time <= result["duration_seconds"] for time in values), result
                assert values == sorted(set(values)), result
        print("Beat This! worker lifecycle, rejected inputs and two inferences passed")


if __name__ == "__main__":
    main()
