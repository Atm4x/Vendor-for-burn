#!/usr/bin/env python3
"""Test the original CUDA semantic sampler via single-thread C++ emulation against independent sorted probabilities."""
from __future__ import annotations

import ctypes
import math
import pathlib
import random
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "cuda_semantic_sampling.cu"
MASK = (1 << 64) - 1


def f32(x: float) -> float:
    return ctypes.c_float(x).value


def mix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def reference(logits: list[float], history: list[int], top_p: float, temp: float, penalty: float, seed: int, step: int) -> int:
    adjusted = [f32(f32(v * penalty) if v < 0 else f32(v / penalty)) if h else f32(v) for v, h in zip(logits, history)]
    adjusted = [0.0 if x == 0 else x for x in adjusted]
    indices = sorted(range(len(logits)), key=lambda i: (-adjusted[i], i))
    highest = adjusted[indices[0]]
    weights = [math.exp(f32(adjusted[i] - highest)) for i in indices]
    total = sum(weights)
    accumulated = 0.0
    best = (-1.0, -1)
    for rank, (i, mass) in enumerate(zip(indices, weights)):
        accumulated += mass / total
        if rank > 0 and accumulated > f32(top_p):
            break
        random_bits = mix64(seed ^ ((0xD1B54A32D192ED03 * (step + 1)) & MASK) ^ ((0x94D049BB133111EB * (i + 1)) & MASK))
        uniform = ((random_bits >> 11) + 0.5) / 9007199254740992.0
        value = math.exp(f32(f32(adjusted[i] - highest) / f32(temp))) / -math.log(uniform)
        if value > best[0] or (value == best[0] and i < best[1]):
            best = (value, i)
    return best[1]


def shim() -> str:
    prefix = """#include <cstdio>
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
#define __syncthreads() ((void)0)
struct FakeDim { int x; };
static FakeDim threadIdx{0};
static FakeDim blockDim{1};
static int atomicExch(int* p, int v) { int old = *p; *p = v; return old; }
"""
    suffix = """
int main() {
    int n, eos;
    float top_p, temp, penalty;
    unsigned long long seed, step;
    while (scanf("%d %f %f %f %llu %llu %d", &n, &top_p, &temp, &penalty, &seed, &step, &eos) == 7) {
        static float logits[262144];
        static int history[262144];
        for (int i=0;i<n;++i) scanf("%f", &logits[i]);
        for (int i=0;i<n;++i) scanf("%d", &history[i]);
        int result[2]{-1,-1};
        int column[2]{-1,-1};
        fish_semantic_sample(logits, history, n, top_p, temp, penalty, seed, step, eos, result, column, 1);
        printf("%d %d %d\\n", result[0], result[1], column[0]);
        fflush(stdout);
    }
}
"""
    return prefix + SOURCE.read_text() + suffix


def main() -> None:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise SystemExit("C++ compiler required")
    rng = random.Random(0xF13028)
    cases = []
    for n in [1, 2, 7, 19, 63, 64, 65, 511, 1024, 4097, 16384, 65536, 155000]:
        for profile in range(5):
            values = [f32(rng.uniform(-3, 3)) for _ in range(n)]
            if profile == 0:
                values = [0.0] * n
            elif profile == 1:
                values = [f32((i % 11) * 0.25) for i in range(n)]
            history = [int(rng.random() < 0.19) for _ in range(n)]
            top_p = f32([0.01, 0.2, 0.7, 0.97, 1.0][profile])
            temp = f32([0.4, 0.7, 0.99, 1.2, 0.3][profile])
            penalty = f32([1.0, 1.1, 1.5, 1.7, 0.9][profile])
            seed = rng.randrange(1, 1 << 64)
            step = profile
            eos = rng.randrange(n)
            cases.append((values, history, top_p, temp, penalty, seed, step, eos))
    with tempfile.TemporaryDirectory() as d:
        source, exe = pathlib.Path(d) / "semantic.cpp", pathlib.Path(d) / "semantic"
        source.write_text(shim())
        subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(exe)], check=True)
        payload = "".join(
            f"{len(xs)} {p} {t} {penalty} {seed} {step} {eos}\n"
            + " ".join(str(v) for v in xs) + "\n"
            + " ".join(str(v) for v in history) + "\n"
            for xs, history, p, t, penalty, seed, step, eos in cases
        )
        lines = subprocess.run([str(exe)], input=payload, text=True, capture_output=True, check=True, timeout=100).stdout.splitlines()
    assert len(lines) == len(cases), (len(lines), len(cases))
    for index, (line, case) in enumerate(zip(lines, cases)):
        xs, history, p, t, penalty, seed, step, eos = case
        token, flagged, column_token = map(int, line.split())
        expected = reference(xs, history, p, t, penalty, seed, step)
        assert token == expected, (index, len(xs), token, expected, p)
        assert flagged == int(token == eos), (index, token, flagged, eos)
        assert column_token == token, (index, column_token, token)
    print(f"PASS {len(cases)} CUDA semantic host-emulation/reference parity cases")


if __name__ == "__main__":
    main()
