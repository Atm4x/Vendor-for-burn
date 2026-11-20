"""Compute exact PCM16 comparison and error statistics for two mono WAVs."""

import array
import hashlib
import json
import math
import sys
import wave
from pathlib import Path


def read_pcm(path: Path):
    with wave.open(str(path), 'rb') as wav:
        metadata = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes())
        if metadata[1:3] != (1, 2):
            raise ValueError(f'expected mono PCM16 for {path}, got {metadata}')
        raw = wav.readframes(wav.getnframes())
    samples = array.array('h')
    samples.frombytes(raw)
    if sys.byteorder != 'little':
        samples.byteswap()
    return metadata, samples, raw


def compare(original: Path, candidate: Path):
    m1, ref, a = read_pcm(original)
    m2, got, b = read_pcm(candidate)
    if m1 != m2:
        return dict(compatible=False, baseline_format=m1, candidate_format=m2)
    different = 0
    max_lsb = 0
    error_energy = 0
    signal_energy = 0
    for x, y in zip(ref, got):
        delta = int(x) - int(y)
        different += delta != 0
        max_lsb = max(max_lsb, abs(delta))
        error_energy += delta * delta
        signal_energy += int(x) * int(x)
    snr = 10 * math.log10(signal_energy / error_energy) if error_energy and signal_energy else (999.0 if error_energy == 0 else -999.0)
    return dict(compatible=True, bit_exact=a == b, sample_count=len(ref), different_samples=different,
                max_pcm_lsb=max_lsb, rmse_pcm_lsb=math.sqrt(error_energy / max(1, len(ref))),
                snr_db=snr, baseline_pcm_sha256=hashlib.sha256(a).hexdigest(),
                candidate_pcm_sha256=hashlib.sha256(b).hexdigest())


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: Compare-Stage130-PCM.py BASELINE.wav CANDIDATE.wav')
    print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), ensure_ascii=False))
