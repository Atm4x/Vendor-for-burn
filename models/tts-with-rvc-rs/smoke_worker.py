"""Exercise the native RVC worker framing/lifecycle, optionally with real weights."""

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
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"worker exited during {command}; code={proc.poll()}")
    result = json.loads(line)
    assert result["version"] == 1 and result["id"] == request_id, result
    assert isinstance(result["ok"], bool), result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--backend-path", type=Path)
    parser.add_argument("--runtime-dll", type=Path, action="append", default=[])
    parser.add_argument("--gpu-id")
    parser.add_argument("--name", default="rvc_v2")
    parser.add_argument("--voice", type=Path)
    parser.add_argument("--hubert", type=Path)
    parser.add_argument("--rmvpe", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--input-wav", type=Path)
    args = parser.parse_args()
    if args.input_wav and not all((args.voice, args.hubert, args.rmvpe)):
        parser.error("--input-wav requires --voice, --hubert, --rmvpe")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        input_dir, output_dir = root / "input", root / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        fixtures = []
        for role, given in (("voice", args.voice), ("hubert", args.hubert), ("rmvpe", args.rmvpe)):
            if given is None:
                file = root / f"fixture-{role}.pt"
                file.write_bytes(b"smoke only; do not load")
            else:
                file = given.resolve(strict=True)
            fixtures.extend((f"--{role}", str(file)))
        argv = [str(args.exe.resolve()), "--ipc-stdio", "--backend", args.backend,
                "--name", args.name, "--input-dir", str(input_dir.resolve()),
                "--output-dir", str(output_dir.resolve()), *fixtures]
        if args.backend_path:
            argv.extend(("--backend-path", str(args.backend_path.resolve(strict=True))))
        for dll in args.runtime_dll:
            argv.extend(("--runtime-dll", str(dll.resolve(strict=True))))
        if args.gpu_id:
            argv.extend(("--gpu-id", args.gpu_id))
        if args.index:
            argv.extend(("--index", str(args.index.resolve(strict=True))))
        with (root / "stderr.log").open("w", encoding="utf-8") as log, subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, encoding="utf-8", bufsize=1) as proc:
            assert call(proc, "capabilities")["result"]["name"] == args.name
            assert call(proc, "status")["result"]["phase"] == "started"
            assert not call(proc, "infer", {})["ok"]
            assert not call(proc, "nonexistent", {})["ok"]
            assert not call(proc, "set_voice", {"voice": str(args.voice or "missing.pth")})["ok"]
            if args.input_wav:
                assert call(proc, "load")["ok"]
                assert call(proc, "warmup")["ok"]
                source = args.input_wav.read_bytes()
                snapshot = input_dir / "sample.wav"
                snapshot.write_bytes(source)
                response = call(proc, "infer", {
                    "input": {"file_name": snapshot.name, "size_bytes": len(source),
                              "sha256": hashlib.sha256(source).hexdigest()},
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
