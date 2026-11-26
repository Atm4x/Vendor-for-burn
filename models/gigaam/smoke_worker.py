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


def make_tone(path):
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", ""))
        handle.writeframes(
            b"".join(
                struct.pack("<h", round(12000 * math.sin(i * 330 * math.tau / 16000)))
                for i in range(16000)
            )
        )


def snapshot(path):
    data = path.read_bytes()
    return {
        "file_name": path.name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    options = parser.parse_args()
    kind = (
        "embeddings"
        if options.model_id.endswith("_ssl")
        else "emotion" if options.model_id == "gigaam_emo" else "asr"
    )
    with tempfile.TemporaryDirectory(prefix="gigaam-worker-") as directory:
        root = Path(directory)
        audio = root / "tone.wav"
        make_tone(audio)
        good = snapshot(audio)
        bad = dict(good, sha256="0" * 64)
        escaping = dict(good, file_name="../tone.wav")
        schedule = [
            ("capabilities", {}, True),
            ("infer", {"gigaam_version": 1, "kind": kind, "input": good}, False),
            ("load", {"model_id": "gigaam_invalid"}, False),
            ("load", {"model_id": options.model_id}, True),
            ("status", {}, True),
            ("warmup", {}, True),
            ("status", {}, True),
            ("infer", {"gigaam_version": 1, "kind": kind, "input": bad}, False),
            ("infer", {"gigaam_version": 1, "kind": kind, "input": escaping}, False),
            ("infer", {"gigaam_version": 1, "kind": kind, "input": good}, True),
            ("unload", {}, True),
            ("status", {}, True),
            ("shutdown", {}, True),
        ]
        requests = [
            {
                "version": 1,
                "id": str(uuid.uuid4()),
                "command": command,
                "payload": payload,
            }
            for command, payload, _ in schedule
        ]
        command = [
            str(options.exe.resolve(strict=True)),
            "--ipc-stdio",
            "--backend", options.backend,
            "--model-id", options.model_id,
            "--model-dir", str(options.model_dir.resolve(strict=True)),
            "--input-dir", str(root),
            "--output-dir", str(root),
        ]
        result = subprocess.run(
            command,
            input="".join(json.dumps(request) + "\n" for request in requests),
            encoding="utf-8",
            capture_output=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stderr
        replies = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(replies) == len(requests), replies
        for request, reply, (_, _, expected) in zip(requests, replies, schedule):
            assert reply["version"] == 1 and reply["id"] == request["id"], reply
            assert reply["ok"] is expected, reply
        assert replies[4]["result"]["phase"] == "loaded"
        assert replies[6]["result"]["phase"] == "ready"
        assert replies[11]["result"]["phase"] == "started"
        assert replies[9]["result"]["kind"] == kind
        if kind == "embeddings":
            output = replies[9]["result"]["output"]
            payload = (root / output["file_name"]).read_bytes()
            assert len(payload) == output["size_bytes"]
            assert hashlib.sha256(payload).hexdigest() == output["sha256"]
    print("GigaAM worker Beat This! protocol v1 smoke: PASS")


if __name__ == "__main__":
    main()
