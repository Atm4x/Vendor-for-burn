#!/usr/bin/env python3
"""Alternate two warmed native IPC processes; compare latency and PCM16."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import tempfile
import uuid
import wave


class Worker:
    def __init__(self, stack, binary, checkpoint, threads):
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.process = stack.enter_context(subprocess.Popen([
            str(binary.resolve()), '--ipc-stdio', '--backend', 'flex',
            '--checkpoint', str(checkpoint.resolve()), '--input-dir', str(self.root),
            '--output-dir', str(self.root),
        ], env=dict(os.environ, RAYON_NUM_THREADS=str(threads)), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, encoding='utf-8'))
        # ExitStack runs this before Popen.__exit__, even on an assertion failure.
        stack.callback(self.close)
        self.ask('load')
        self.ask('warmup')

    def ask(self, command, payload=None):
        request = {'version': 1, 'id': str(uuid.uuid4()), 'command': command,
                   'payload': {} if payload is None else payload}
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if response['id'] != request['id'] or not response['ok']:
            raise RuntimeError(response)
        return response['result']

    def infer(self, payload):
        result = self.ask('infer', payload)
        desc = result['audio']
        path = self.root / desc['file_name']
        data = path.read_bytes()
        assert len(data) == desc['size_bytes'] and hashlib.sha256(data).hexdigest() == desc['sha256']
        with wave.open(str(path), 'rb') as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
            assert wav.getframerate() == payload['sample_rate']
            assert wav.getnframes() == result['sample_count']
            pcm = wav.readframes(wav.getnframes())
        path.unlink()
        return result['compute_ms'], pcm

    def close(self):
        if self.process.poll() is None:
            try:
                self.ask('shutdown')
                self.process.wait(timeout=10)
            finally:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()


def compare(baseline, candidate):
    a = [x[0] for x in struct.iter_unpack('<h', baseline)]
    b = [x[0] for x in struct.iter_unpack('<h', candidate)]
    assert len(a) == len(b) and a
    delta = [x-y for x, y in zip(a, b)]
    noise = sum(x*x for x in delta)
    signal = sum(x*x for x in a)
    assert signal > 0
    return {'samples': len(a), 'max_error_lsb': max(map(abs, delta)),
            'snr_db': 10*math.log10(signal/noise) if noise else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--text', required=True)
    parser.add_argument('--speaker', required=True)
    parser.add_argument('--rate', type=int, default=24000)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--max-error-lsb', type=int, default=8)
    args = parser.parse_args()
    assert args.repeats > 0 and args.threads > 0
    payload = {'silero_version': 1, 'speaker': args.speaker, 'sample_rate': args.rate,
               'input': {'kind': 'text', 'text': args.text}}
    times = [[], []]
    audio = [None, None]
    with ExitStack() as stack:
        workers = [Worker(stack, path, args.checkpoint, args.threads)
                   for path in (args.baseline, args.candidate)]
        for worker in workers:
            worker.infer(payload)
        for i in range(args.repeats):
            for j in ([0, 1] if i % 2 == 0 else [1, 0]):
                ms, pcm = workers[j].infer(payload)
                if audio[j] is not None:
                    assert audio[j] == pcm, 'non-deterministic output'
                audio[j] = pcm
                times[j].append(ms)
    parity = compare(*audio)
    medians = [statistics.median(x) for x in times]
    print(json.dumps({'rate': args.rate, 'threads': args.threads, 'repeats': args.repeats,
                      'baseline_ms': times[0], 'candidate_ms': times[1],
                      'median_baseline_ms': medians[0], 'median_candidate_ms': medians[1],
                      'speedup': medians[0] / medians[1], 'parity': parity}, indent=2))
    if parity['max_error_lsb'] > args.max_error_lsb:
        raise SystemExit('PCM16 parity threshold exceeded')


if __name__ == '__main__':
    main()
