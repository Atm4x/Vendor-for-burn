"""Require the complete 85–95 second WAV for full-length RVC profiling."""
import json
import sys
import wave
from pathlib import Path


def inspect(path):
    with wave.open(str(path), "rb") as wav:
        samples = wav.getnframes()
        rate = wav.getframerate()
        seconds = samples / rate
        result = dict(path=str(path), sample_rate=rate, samples=samples, seconds=seconds,
                      channels=wav.getnchannels(), sample_width=wav.getsampwidth())
        if not 85 <= seconds <= 95 or wav.getnchannels() != 1:
            raise ValueError(f"Expected 85-95s mono WAV, got {result}")
        return result


if __name__ == "__main__":
    try:
        print(json.dumps(inspect(Path(sys.argv[1]))))
    except (OSError, ValueError, IndexError, wave.Error) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
