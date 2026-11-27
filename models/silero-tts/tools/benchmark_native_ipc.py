#!/usr/bin/env python3
"""Measure cold load, warmup and steady IPC inference versus an optional PyTorch oracle."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import time
import uuid
import wave
from pathlib import Path


def benchmark(binary, checkpoint, text, speaker, rate, repeats, threads):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder).resolve()
        env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
        start = time.perf_counter()
        with subprocess.Popen([str(binary.resolve()), '--ipc-stdio', '--backend', 'flex',
                               '--checkpoint', str(checkpoint.resolve()), '--input-dir', str(root),
                               '--output-dir', str(root)], cwd=root, env=env, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, encoding='utf-8') as worker:
            def ask(command, payload=None):
                request = {'version': 1, 'id': str(uuid.uuid4()), 'command': command,
                           'payload': {} if payload is None else payload}
                started = time.perf_counter()
                worker.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
                worker.stdin.flush()
                response = json.loads(worker.stdout.readline())
                elapsed = time.perf_counter() - started
                if response['id'] != request['id'] or not response['ok']:
                    raise RuntimeError(response)
                return response['result'], elapsed

            try:
                _, startup = ask('capabilities')
                startup = time.perf_counter() - start
                model, load = ask('load')
                _, warmup = ask('warmup')
                payload = {'silero_version': 1, 'speaker': speaker, 'sample_rate': rate,
                           'input': {'kind': 'text', 'text': text}}
                runs = []
                last = None
                for _ in range(repeats + 1):
                    result, elapsed = ask('infer', payload)
                    desc = result['audio']
                    path = root / desc['file_name']
                    data = path.read_bytes()
                    assert len(data) == desc['size_bytes'] and hashlib.sha256(data).hexdigest() == desc['sha256']
                    with wave.open(str(path), 'rb') as wav:
                        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == rate
                        assert wav.getnframes() == result['sample_count']
                        last = wav.readframes(wav.getnframes())
                    path.unlink()
                    runs.append({'wall_seconds': elapsed, 'compute_seconds': result['compute_ms'] / 1000,
                                 'sample_count': result['sample_count']})
                ask('shutdown')
                assert worker.wait(timeout=10) == 0
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()
    seconds = runs[-1]['sample_count'] / rate
    median = statistics.median(x['compute_seconds'] for x in runs[1:])
    return {'model_variant': model['model_variant'], 'speaker': speaker, 'text': text,
            'sample_rate': rate, 'threads': threads, 'startup_seconds': startup, 'load_seconds': load,
            'warmup_seconds': warmup, 'audio_seconds': seconds, 'runs': runs,
            'median_compute_seconds': median, 'rtf': median / seconds}, last


def oracle(checkpoint, text, speaker, rate, repeats, threads, pcm):
    import math
    import struct
    import torch
    from torch.package import PackageImporter

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    model = PackageImporter(str(checkpoint)).load_pickle('tts_models', 'model')
    load = time.perf_counter() - start
    times = []
    with torch.inference_mode():
        for _ in range(repeats + 1):
            start = time.perf_counter()
            result = model.apply_tts(text=text, speaker=speaker, sample_rate=rate).detach().float().cpu().reshape(-1)
            times.append(time.perf_counter() - start)
    expected = result.tolist()
    actual = [x[0] / 32767 for x in struct.iter_unpack('<h', pcm)]
    if len(actual) != len(expected):
        raise ValueError(f'waveform length differs: {len(actual)} vs {len(expected)}')
    error = [a - b for a, b in zip(actual, expected)]
    noise = sum(x*x for x in error)
    signal = sum(x*x for x in expected)
    median = statistics.median(times[1:])
    return {'load_seconds': load, 'times': times, 'median_seconds': median,
            'rtf': median / (len(expected) / rate), 'max_absolute_error_pcm16': max(map(abs, error)),
            'snr_db_pcm16': 10*math.log10(signal/noise) if noise else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--text', required=True)
    parser.add_argument('--speaker', required=True)
    parser.add_argument('--rate', type=int, default=24000)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--pytorch-oracle', action='store_true')
    args = parser.parse_args()
    report, pcm = benchmark(args.binary, args.checkpoint, args.text, args.speaker, args.rate, args.repeats, args.threads)
    if args.pytorch_oracle:
        report['pytorch'] = oracle(args.checkpoint, args.text, args.speaker, args.rate, args.repeats, args.threads, pcm)
        report['rust_over_pytorch'] = report['median_compute_seconds'] / report['pytorch']['median_seconds']
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
