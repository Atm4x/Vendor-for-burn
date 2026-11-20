"""Create a temporary uncompressed PCM WAV prefix; never include it in reports."""

import argparse
import wave
from pathlib import Path


def trim(source, destination, seconds):
    with wave.open(str(source), "rb") as src:
        if src.getcomptype() != "NONE":
            raise ValueError("Only uncompressed PCM WAV is supported; use -ProfileSeconds 0 for other formats")
        count = min(src.getnframes(), round(src.getframerate() * seconds))
        with wave.open(str(destination), "wb") as dst:
            dst.setparams(src.getparams())
            dst.writeframes(src.readframes(count))
    if count == 0:
        raise ValueError("Input WAV is empty")
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, required=True)
    a = p.parse_args()
    if not (0.5 <= a.seconds <= 90):
        p.error("seconds must be between 0.5 and 90")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Temporary PCM WAV frames: {trim(a.source, a.output, a.seconds)}", flush=True)


if __name__ == "__main__":
    main()
