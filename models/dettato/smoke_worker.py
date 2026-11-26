import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


with tempfile.TemporaryDirectory(prefix="dettato-worker-") as directory:
    root = Path(directory)
    placeholder = root / "placeholder"
    placeholder.write_bytes(b"invalid checkpoint")
    command = [
        str(Path(sys.argv[1]).resolve(strict=True)),
        "--ipc-stdio", "--backend", "cpu",
        "--input-dir", directory, "--output-dir", directory,
    ]
    for stage in ("transcriber", "evidence", "grouper", "drums"):
        for role in ("config", "weights"):
            command.extend((f"--{stage}-{role}", str(placeholder)))

    schedule = [
        ("capabilities", {}, True),
        ("status", {}, True),
        ("warmup", {}, False),
        ("load", {}, False),
        ("status", {}, True),
        ("unload", {}, True),
        ("shutdown", {}, True),
    ]
    requests = [
        {"version": 1, "id": str(uuid.uuid4()), "command": name, "payload": payload}
        for name, payload, _ in schedule
    ]
    run = subprocess.run(
        command,
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    replies = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(replies) == len(schedule), replies
    for request, reply, (_, _, success) in zip(requests, replies, schedule):
        assert reply["version"] == 1 and reply["id"] == request["id"], reply
        assert reply["ok"] is success, reply
    assert replies[0]["result"]["outputs"] == ["combined", "tracks"]
    assert replies[1]["result"]["phase"] == replies[4]["result"]["phase"] == "started"
print("Dettato IPC smoke: PASS")
