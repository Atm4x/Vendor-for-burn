import argparse
import hashlib
import json
import tarfile
import tomllib
from pathlib import Path


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def restore_vendor(root: Path, lockfile: Path, zerocopy_archive: Path):
    packages = tomllib.loads(lockfile.read_text(encoding="utf-8"))["package"]
    crate = next(p for p in packages if p["name"] == "zerocopy" and p["version"] == "0.8.63")
    archive_bytes = zerocopy_archive.read_bytes()
    if digest(archive_bytes) != crate["checksum"]:
        raise ValueError("zerocopy crate archive does not match Cargo.lock checksum")
    with tarfile.open(zerocopy_archive, "r:gz") as archive:
        member = archive.extractfile("zerocopy-0.8.63/win-cargo.bat")
        if member is None:
            raise ValueError("original win-cargo.bat not found in crate archive")
        original_batch_file = member.read()

    restored = 0
    checked = 0
    for manifest in sorted(root.glob("*/.cargo-checksum.json")):
        entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        for name, expected in entries.items():
            path = manifest.parent / name
            original = path.read_bytes()
            checked += 1
            if digest(original) == expected:
                continue
            lf = original.replace(b"\r\n", b"\n")
            crlf = lf.replace(b"\n", b"\r\n")
            if digest(lf) == expected:
                data = lf
            elif digest(crlf) == expected:
                data = crlf
            elif path.relative_to(root).as_posix() == "zerocopy-0.8.63/win-cargo.bat":
                data = original_batch_file
            else:
                raise ValueError(f"unrecoverable vendor checksum mismatch: {path}")
            if digest(data) != expected:
                raise ValueError(f"recovered bytes do not match upstream checksum: {path}")
            path.write_bytes(data)
            restored += 1
    print(f"Verified {checked} vendored source files; restored {restored} original byte sequences")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, default=Path("vendor"))
    parser.add_argument("--lockfile", type=Path, default=Path("Cargo.lock"))
    parser.add_argument("--zerocopy-archive", type=Path, required=True)
    args = parser.parse_args()
    restore_vendor(args.vendor, args.lockfile, args.zerocopy_archive)


if __name__ == "__main__":
    main()
