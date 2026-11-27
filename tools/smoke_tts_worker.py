import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


exe = Path(sys.argv[1]).resolve(strict=True)
family = sys.argv[2]
with tempfile.TemporaryDirectory(prefix="tts-worker-") as directory:
    placeholder = Path(directory) / "invalid"
    placeholder.write_bytes(b"invalid weights")
    common = [str(exe), "--ipc-stdio", "--backend", "cuda",
              "--input-dir", directory, "--output-dir", directory]
    if family == "fish-speech-1.5":
        flags = ["--semantic-config", "--semantic-tokenizer", "--semantic-manifest",
                 "--semantic-weights", "--firefly-config", "--firefly-manifest", "--firefly-weights"]
        launches = [common + [value for flag in flags for value in (flag, str(placeholder))]]
    elif family == "f5-tts":
        launches = [
            common + ["--model-id", model, "--dit-weights", str(placeholder),
                      "--vocos-weights", str(placeholder), "--vocab", str(placeholder)] + extra
            for model, extra in (
                ("f5_tts_ru", []),
                ("f5_tts_cross_lingual", ["--speed-weights", str(placeholder)]),
            )
        ]
    else:
        raise ValueError(f"unsupported worker family: {family}")
    schedule = (("capabilities", True), ("infer", False), ("load", False),
                ("status", True), ("warmup", False), ("shutdown", True))
    requests = [dict(version=1, id=str(uuid.uuid4()), command=command, payload={})
                for command, _ in schedule]
    payload = "".join(json.dumps(request) + "\n" for request in requests)
    for launch in launches:
        result = subprocess.run(launch, input=payload, encoding="utf-8",
                                capture_output=True, timeout=45, check=True)
        replies = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(replies) == len(schedule), replies
        for request, reply, (_, success) in zip(requests, replies, schedule):
            assert reply["version"] == 1 and reply["id"] == request["id"], reply
            assert reply["ok"] is success, reply
        assert replies[3]["result"]["phase"] == "started", replies[3]
        assert replies[0]["result"]["service"] == "tts", replies[0]
print(f"{family} IPC smoke: PASS")
