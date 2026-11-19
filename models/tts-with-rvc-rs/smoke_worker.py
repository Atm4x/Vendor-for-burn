"""Verify RVC worker stdio v1 lifecycle and optionally a private model/audio fixture."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import uuid
from pathlib import Path


def call(proc, command, payload=None):
    request_id = str(uuid.uuid4())
    request = {"version": 1, "id": request_id, "command": command, "payload": payload or {}}
    proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
    proc.stdin.flush()
    response = proc.stdout.readline()
    if not response:
        raise RuntimeError(f"worker exited while handling {command}; exit={proc.poll()}")
    result = json.loads(response)
    assert result["id"] == request_id and result["version"] == 1, result
    assert isinstance(result["ok"], bool), result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--model-id", default="rvc_v2_smoke")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--input-wav", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        input_dir, output_dir = root / "input", root / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        model_dir = args.model_dir or root / "model"
        if args.model_dir is None:
            model_dir.mkdir()
        argv = [str(args.exe.resolve()), "--ipc-stdio", "--backend", args.backend,
                "--model-id", args.model_id, "--model-dir", str(model_dir.resolve()),
                "--input-dir", str(input_dir.resolve()), "--output-dir", str(output_dir.resolve())]
        with (root / "stderr.log").open("w", encoding="utf-8") as log, subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, encoding="utf-8", bufsize=1) as proc:
            assert call(proc, "capabilities")["result"]["model_id"] == args.model_id
            assert call(proc, "status")["result"]["phase"] == "started"
            assert not call(proc, "infer", {})["ok"]
            assert not call(proc, "nonexistent", {})["ok"]
            if args.input_wav is not None:
                if args.model_dir is None:
                    raise ValueError("--input-wav needs --model-dir")
                assert call(proc, "load", {"model_id": args.model_id})["ok"]
                assert call(proc, "warmup")["ok"]
                data = args.input_wav.read_bytes()
                snapshot = input_dir / "sample.wav"
                snapshot.write_bytes(data)
                response = call(proc, "infer", {
                    "rvc_version": 1, "kind": "voice_conversion",
                    "input": {"file_name": snapshot.name, "size_bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()},
                })
                assert response["ok"], response
                info = response["result"]["output"]
                encoded = (output_dir / info["file_name"]).read_bytes()
                assert len(encoded) == info["size_bytes"]
                assert hashlib.sha256(encoded).hexdigest() == info["sha256"]
                assert encoded.startswith(b"RIFF")
                assert call(proc, "unload")["ok"]
            assert call(proc, "shutdown")["ok"]
            assert proc.wait(timeout=20) == 0
            assert proc.stdout.read() == "", "stdout contains non-protocol text"
        print("RVC worker IPC smoke PASS")


if __name__ == "__main__":
    main()
