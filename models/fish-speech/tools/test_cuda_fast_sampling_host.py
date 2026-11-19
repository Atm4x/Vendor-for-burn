#!/usr/bin/env python3
"""Compile the CUDA sampling source with a single-thread CPU shim and compare to the source sampler mathematics."""
from __future__ import annotations

import ctypes
import math
import pathlib
import random
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "cuda_fast_sampling.cu"
MASK64 = (1 << 64) - 1


def f32(value: float) -> float:
    return ctypes.c_float(value).value


def mix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def expected_sample(logits: list[float], history: list[int], top_p: float, temp: float, penalty: float, seed: int, codebook: int) -> int:
    adjusted = [f32(x) for x in logits]
    for i, value in enumerate(adjusted):
        if history[i]:
            adjusted[i] = f32(value * penalty if value < 0 else value / penalty)
    ordered = sorted(range(len(logits)), key=lambda i: (-adjusted[i], i))
    maximum = adjusted[ordered[0]]
    denominator = sum(math.exp(f32(adjusted[i] - maximum)) for i in ordered)
    cumulative, best, selected = 0.0, -1.0, -1
    for rank, i in enumerate(ordered):
        value = f32(adjusted[i] - maximum)
        likelihood = math.exp(value)
        cumulative += likelihood / denominator
        if rank != 0 and cumulative > f32(top_p):
            break
        x = seed ^ ((0xD1B54A32D192ED03 * (codebook + 1)) & MASK64) ^ ((0x94D049BB133111EB * (i + 1)) & MASK64)
        u = ((mix64(x) >> 11) + 0.5) / float(1 << 53)
        weight = math.exp(f32(value / f32(temp)))
        race = weight / -math.log(u)
        if race > best:
            best, selected = race, i
    return selected


def prepare_source() -> str:
    source = SOURCE.read_text()
    source = source.replace("extern __shared__ float score[];", "alignas(8) static float score[8192];")
    source = source.replace("__shared__ int invalid;", "static int invalid;")
    prefix = """#include <cstdio>
#define __device__
#define __global__
#define __forceinline__ inline
#define __syncthreads() ((void)0)
struct FakeDim { int x; };
static FakeDim threadIdx{0};
static FakeDim blockDim{1};
static int atomicExch(int* p, int v) { int old = *p; *p = v; return old; }
"""
    suffix = """
int main() {
    int size, codebook;
    float top_p, temperature, penalty;
    unsigned long long seed;
    while (scanf("%d %f %f %f %llu %d", &size, &top_p, &temperature, &penalty, &seed, &codebook) == 6) {
        float logits[4096]{};
        int history[4096]{};
        for (int i=0; i<size; ++i) scanf("%f", &logits[i]);
        for (int i=0; i<size; ++i) scanf("%d", &history[i]);
        int padded=1;
        while (padded < size) padded *= 2;
        int result=-1;
        int column[16]{};
        fish_fast_sample(logits, history, size, padded, top_p, temperature, penalty, seed, codebook, &result, column);
        printf("%d\\n", column[codebook]);
        fflush(stdout);
    }
}
"""
    return prefix + source + suffix


def main() -> None:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise SystemExit("C++ compiler required for host emulation")
    rng = random.Random(0xF15113)
    cases = []
    for n in [1, 2, 3, 7, 16, 63, 64, 65, 127, 128, 511, 512, 513, 1024, 1025, 2048, 4096]:
        for profile in range(6):
            logits = [f32(rng.uniform(-7, 7)) for _ in range(n)]
            if profile == 0:
                logits = [0.0] * n
            if profile == 1:
                logits = [f32((i % 7) * 0.25) for i in range(n)]
            history = [int(rng.random() < 0.17) for _ in range(n)]
            top_p = [0.1, 0.3, 0.7, 0.9, 1.0, 0.999][profile]
            temperature = f32([0.3, 0.7, 1.0, 1.4, 0.99, 0.55][profile])
            penalty = f32([1.0, 1.1, 1.5, 1.8, 0.9, 1.2][profile])
            seed = rng.randrange(1, 1 << 64)
            codebook = profile + 1
            cases.append((logits, history, top_p, temperature, penalty, seed, codebook))
    with tempfile.TemporaryDirectory() as temp:
        src, exe = pathlib.Path(temp) / "sampler.cpp", pathlib.Path(temp) / "sampler"
        src.write_text(prepare_source())
        subprocess.run([compiler, "-std=c++17", "-O2", str(src), "-o", str(exe)], check=True)
        payload = "".join(
            f"{len(xs)} {p} {t} {pen} {seed} {codebook}\n"
            + " ".join(str(x) for x in xs) + "\n"
            + " ".join(str(x) for x in hist) + "\n"
            for xs, hist, p, t, pen, seed, codebook in cases
        )
        output = subprocess.run([str(exe)], input=payload, text=True, capture_output=True, check=True).stdout.splitlines()
    assert len(output) == len(cases), (len(output), len(cases))
    for index, (actual, (xs, hist, p, t, pen, seed, codebook)) in enumerate(zip(output, cases)):
        expected = expected_sample(xs, hist, p, t, pen, seed, codebook)
        assert int(actual) == expected, f"case {index} n={len(xs)} actual={actual} expected={expected}"
    print(f"PASS: {len(cases)} C++-host CUDA-kernel cases match independent CPU sampling reference")


if __name__ == "__main__":
    main()
