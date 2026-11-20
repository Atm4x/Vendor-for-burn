# Stage 39 — measured TemporalQuad result and interior-path R&D

Date: 2026-11-10. Branch: `experiment/conv-kernel-rnd-stage33-20261110` (PR #2).

## Provenance

Real-device results were supplied by the operator in `RVC_GPU_BENCH_RESULTS.zip`, produced on Windows x64 with an NVIDIA GeForce RTX 4060 (`torch 2.11.0+cu128`, cuDNN 9.19, `tf32_allowed=false`). The archive contains full FP32 tensors for PyTorch, CUDA CubeCL and WGPU CubeCL, plus reports, stderr logs and environment diagnostics. The operator used `-Iterations 20 -Warmup 5`. The measurements include launch/allocation and synchronized CPU readback. All inputs are **deterministic synthetic** tensors with NSF model geometries, not private CrazyMita activations.

## Operator GPU results

All times in milliseconds (lower is better). `Packed` is the two-time-position baseline; `Quad` is the four-time-position kernel. The acceleration is `Packed / Quad` (>1 is faster Quad).

| Layer (Cin→Cout, L, K, dilation) | CUDA Packed | CUDA Quad | CUDA Packed/Quad | WGPU Packed | WGPU Quad | WGPU Packed/Quad | PyTorch strict FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K3 (256→256, 240, K3, D1) | 0.2077 | 0.2044 | 1.016x | 0.2959 | 0.3959 | 0.747x | 0.1435 |
| K11/D5 (128→128, 2400, K11, D5) | 0.8498 | 0.7275 | **1.168x** | 1.1492 | 1.1796 | 0.974x | 0.4927 |
| K11 (32→32, 9600, K11, D1) | 0.5121 | 0.4753 | **1.077x** | 0.5901 | 0.6474 | 0.911x | 0.3987 |

The CUDA Quad remains 1.424x, 1.477x and 1.192x slower in measured latency than PyTorch strict FP32 respectively. These ratios are **not kernel-only throughput comparisons**, and PyTorch/cuDNN kernel selection was not profiled. Stage 37 PyTorch times were different, so no attribution to a code change is justified across separate runs.

## Full-tensor numerical checks

The uploaded archive's reports indicate `pass` for all 9 Direct/Packed/Quad variants on each backend. An independent examination of `.f32le` contents confirmed CUDA Quad and WGPU Quad are bit-for-bit identical over **61,440**, **307,200**, and **307,200** elements respectively. All checked values are finite.

| Layer | Maximum absolute deviation against PyTorch | RMSE |
|---|---:|---:|
| K3 | 2.9802322e-6 | 5.4006816e-7 |
| K11/D5 | 1.0490417e-5 | 1.3410244e-6 |
| K11 | 1.9073486e-6 | 3.1129690e-7 |

These are **full-tensor synthetic parity** measurements, not actual RVC checkpoint/audio parity.

## R&D hypothesis: interior boundary specialization

Stage 38 always evaluated per-kernel spatial bounds and up to four `valid*` predicates before issuing loads. For fully valid four-position tiles, these checks are redundant. Stage 39 introduces a **separate experimental variant `QuadInterior`** in the existing shared CubeCL source:

1. Compute a whole-tile interior predicate once: the earliest `t0 * stride - padding` is nonnegative and the latest `t3 * stride + (kernel - 1) * dilation - padding` lies inside the input, with `t3 < output_length`.
2. For an interior tile, compute four input positions without repeated padding comparisons. Reuse the same packed weight vector for all four accumulators.
3. For all boundary tiles, retain Stage 38's checked path, including incomplete last tiles, arbitrary valid stride/dilation and grouping.
4. Preserve `Quad` as an independent baseline and add `QuadInterior` to the Windows A/B harness and full-tensor comparator. Neither production model dispatch nor existing packed-weight format changes.

In the three measured NSF geometries, **58/60 (96.7%)**, **586/600 (97.7%)**, and **2396/2400 (99.8%)** of temporal tiles are fully interior. These percentages count logical tiles, not hardware thread execution. The hypothesis is that moving spatial checks outside the accumulation loops may reduce GPU instruction cost; the additional branch and code size could also hurt compilation, registers, cache behavior, or WGPU performance. There is no Stage 39 GPU speedup claim until a new hardware benchmark.

## Local verification and next gates

- Rust arithmetic reference: 3888/3888 configurations passed with identical full outputs (batch 1/2; groups 1/2/4; per-group channels; lengths 7/15/23; kernels 1/3/5/11; strides 1/2; dilation 1/2/5). This is a CPU reference **model**, not actual CubeCL GPU execution.
- Rust 1.98.1, offline locked cargo checks: **WGPU PASS** and **CUDA PASS** (using supplied Tracel LLVM 23.1.0-4 offline; the initial CUDA check attempted network installation before the LLVM archive was installed). Compilation cannot establish runtime parity or speedup.
- Python tensor comparator: 12/12 synthetic tensor files PASS using baseline Quad contents for the new candidate as a smoke fixture; deliberate non-finite corruption correctly fails. This does not test the new kernel on hardware.
- The Windows runner exports all four variants, enables full FP32 tensor comparison to PyTorch strict FP32, and preserves the existing Stage 38 benchmark modes.
- Required next experiment: run new CI-built Windows GPU benchmark on RTX 4060 (`-Warmup 5 -Iterations 20`, preferably repeated runs). Report `QuadInterior / Quad`, plus WGPU regression or improvement and full parity.
- Subsequent profiling: CUDA kernel-only events, registers/spills, generated PTX/SASS or shader analysis; select temporal tile by measured backend/shape, not by universal hard-coding. Compare weight-pack amortization and full NSF resident inference separately.

No private weight, index or audio file is stored in the repository. This is research code only.
