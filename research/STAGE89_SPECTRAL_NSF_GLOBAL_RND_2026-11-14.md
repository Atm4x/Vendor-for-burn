# Stage 89 — Global spectral NSF alternative: overlap-save FFT + batched GEMM

**Base:** Stage 88 local `2052eb219cc5a43305243100217da4859d4c5125` (published PR #2 commit `e534ebf120c05a16566f2f958987638eec74e198`). **Toolchain:** Rust 1.100.0. **Scope:** source-only R&D; **no changes to production or to legacy/vendor code.** All reference computations are local CPU calculations, **not CUDA benchmarks**.

## Research question

The Stage 87 real-checkpoint cost atlas estimates roughly 341 GFLOPs of ResBlock convolution per full-sized local Rust chunk (384 latent frames) and GPU diagnostics attribute a large fraction of full-wave latency to NSF residual branches. Earlier direct fused kernels reduce intermediate traffic but do not reduce the main channel×channel×kernel dot products; SVD at 99% weight-energy preservation offered weak FLOP reductions. Investigate a genuinely different, algebraically equivalent computation.

For fixed Conv1d weights with small-to-moderate temporal kernels, overlap-save real FFT replaces direct temporal cross-correlation by spectral multiplication. **All source frames, dilation, zero padding, weight normalization, residual nonlinearities and cross-correlation orientation are retained.** The FFT changes floating-point reduction order; bitwise FP32 identity is **not** expected.

The 1D periodicity is **inside** each zero-padded FFT tile, not around the original audio. The overlap-save discard region prevents circular-convolution wraparound from entering the result. The source is padded by exactly `(kernel - 1) * dilation / 2` on each side, matching PyTorch/Rust Conv1d.

## Implemented research code

- `research/stage89-fft-nsf/spectral_conv.py`: NumPy CPU implementation with weight FFT precomputation, arbitrary dilation, same padding, two spectral-matmul formulations (complex or three real GEMMs), batching of adjacent temporal windows into GEMM columns, and optional bias.
- `research/stage89-fft-nsf/spectral_planner.rs`: pure-Rust 1.100 (std-only) geometrical and cost model plus global constrained choice between direct and spectral operators. No CUDA execution. It uses stable standard library allocation; a custom allocator is not justified by the GPU bottleneck.
- `research/stage89-fft-nsf/atlas.py`: same multiple-choice knapsack planner using **real CrazyMita kernel dimensions**, with a checkpoint-wide memory budget. It keeps direct conv as an option for layers where FFT loses.
- `research/stage89-fft-nsf/test_spectral_conv.py`, `test_atlas.py`, `validate_resblocks.py`: numerical and plan-selection correctness checks. The actual voice checkpoint is read locally and is not included or published.
- `research/stage89-fft-nsf/bench_cpu.py`: operator-only CPU timing of Torch Conv1d and NumPy FFT against the same normalized weights and random inputs; no full RVC or RTX claim.

The vendored `cubek-fft-0.3.1` in this repository **already contains** GPU `rfft` and `irfft` entry points, including support for large windows. This is an implementation opportunity, not evidence that a complete efficient FFT convolution GPU kernel exists. A real GPU implementation additionally needs window packing, spectrum reuse, batched three-GEMM, and inverse overlap-save extraction.

## Correctness evidence

`test_spectral_conv.py`: 64 geometry cases × 2 multiplication formulations × 3 window batch sizes = **384 numerical comparisons PASS**, with both synthetic padding boundaries and non-power-of-two output lengths. The test uses 2e-5 absolute and relative tolerances. **2 Python unittest methods PASS**.

Real CrazyMita checkpoint: 36 Conv1d layers (kernels 7/11; channels 64/128/256) compared against PyTorch FP32 with the original `weight_g / ||weight_v||` normalization. Maximum absolute output difference **4.1723e-7** on 70-sample synthetic feature activations. This verifies operators, not full audio or timbre.

End-to-end computation through 3 Conv1d pairs + LeakyReLU + residual addition inside one real `ResBlock1` of each stage (256/128/64/32 channels), using 127 positions and synthetic input activations:

| Channels | Max absolute FP32 deviation | Relative L2 |
|---:|---:|---:|
| 256 | 8.34e-7 | 2.19e-7 |
| 128 | 1.13e-6 | 3.41e-7 |
| 64 | 1.19e-6 | 3.07e-7 |
| 32 | 2.38e-7 | 2.99e-7 |

`test_atlas.py`: **2 PASS**, including exact exhaustive comparison on small global knapsack instances. Standalone Rust 1.100 `rustc --test`: **4 PASS / 0 FAIL**, including inverse memory-safety bound and matching global optimization against exhaustive search.

The chosen FFT length depends on kernel+dilation and a global spectral-weight memory budget. For 384 latent frames and the current four NSF stages, a model of **72 Conv1d instances** yields:

| GPU spectral-weight budget | FFT layers chosen | Stored spectra | Idealized direct/spectral FLOP ratio |
|---:|---:|---:|---:|
| 64 MiB | 21 | 36.0 MiB | 1.49× |
| 128 MiB | 36 | 75.0 MiB | 2.01× |
| 256 MiB | 49 | 184.0 MiB | 2.48× |
| 512 MiB | 65 | 408.4 MiB | 3.13× |
| 1024 MiB | 72 | 901.7 MiB | 3.36× |

**Interpretation:** These numbers are a theoretical count of real floating-point operations, **not GPU performance or guaranteed memory allocations**. The model counts three real GEMMs per complex product, frequency recomposition and approximate forward/inverse FFT operations. It omits data packing, GPU occupancy and allocation, additional temporary buffers, launch overhead, inter-stage activations and the memory allocator. Weight FFT preprocessing is amortized only when a model remains warm. The 384 latent-frame geometry is taken from the previous Stage 87 cost atlas; it is not the only possible chunk size.

## Negative CPU results (recorded, no extrapolation to CUDA)

All cases use exactly the same real checkpoint-normalized weights, 3840 output samples, four Torch CPU threads, four BLAS threads, three timed iterations after a warmup, and NumPy `three_real_gemm` with 16 temporal windows packed per spectral matrix multiplication. Machine and CPU libraries differ, so these are diagnostic rather than normalized kernel microbenchmarks.

| Real NSF Conv1d | Torch CPU median | NumPy FFT CPU median | FFT / Torch latency |
|---|---:|---:|---:|
| 256ch, k7, d1 | 28.903 ms | 84.120 ms | 2.91× slower |
| 256ch, k11, d5 | 18.362 ms | 243.129 ms | 13.24× slower |
| 128ch, k7, d1 | 3.727 ms | 17.172 ms | 4.61× slower |
| 128ch, k11, d5 | 5.669 ms | 54.795 ms | 9.67× slower |
| 64ch, k11, d5 | 1.495 ms | 9.482 ms | 6.34× slower |

Maximum absolute deviation on those timed real-weight layers was below 3.3e-7. Optimized Torch CPU and NumPy/BLAS use different backend runtimes. **FFT cannot be proposed for production based on FLOPs alone.** The result disproves immediate CPU speed advantage; CUDA remains unmeasured.

## Critical architectural insight

**Per-window complex matrix-vector multiplication defeats Tensor Core utilization.** Batching *many* overlap-save windows is mandatory: for each frequency bin, multiply the fixed `[out_channels,in_channels]` spectral weight matrix by an `[in_channels,windows]` activation matrix. Reuse the weight FFT across windows and, ideally, across 32 original Rust segments. The CPU reference proves numerical parity for packed window batches.

For a 256-channel k11/d5 layer with 3840 output samples, the analytic choice is 1024-point FFT with only four output windows. Four GEMM columns are too few for efficient Tensor Core occupancy; the same operation on much longer chunks or across jointly scheduled segments would offer more columns. A practical CUDA path must resolve this **global batching** problem rather than only porting a NumPy FFT implementation to CubeCL.

A 512 MiB spectral-weight budget picks 65 of 72 layers, despite not taking transient buffers into account. Rigidly storing all converted weights wastes memory. The planner must measure actual VRAM and track all processing buffers, especially for mixed FP16 and other simultaneous model services.

## Decision, next gates

**Do not enable spectral Conv1d in production yet.** Stage 89 is a mathematically validated alternative and a real-checkpoint cost study, not a measured speed breakthrough. The CPU implementation loses substantially. A future GPU proof of concept should, in order:

1. Implement GPU packing/FFT/complex GEMM/IRFFT for **one isolated 128-channel k11/d5 layer** using the vendored `cubek-fft` building blocks and the current CUDA matmul infrastructure; do not alter all 72 layers at once.
2. Compare 3-real-GEMM versus existing FP16 Conv1d with *identical* shapes and activation dtype, measuring FLOPs, JIT time, device memory, GPU kernel time and max/relative error. Require larger GEMM column widths from batched temporal windows.
3. Verify the CUDA precision budget on representative *real model activations* (not just synthetic features), then run the full WAV. Reject if resource usage or quality degrades significantly.
4. If spectral is still slower than ImplicitGemm on RTX 4060, abandon it; redirect to the global 3-second chunk geometry / original Python parity and whole-stage scheduler, not more FFT tuning.

Rust 1.100 custom allocators do not affect device workspace allocations. No user PC was required; no Windows CUDA result was created. Stage 88 Windows CI independently succeeded, but the Stage 88 *user GPU test* is still outstanding.

## Reproduction

From the repository root, with Python + NumPy + PyTorch CPU and a local (private) CrazyMita checkpoint:

```bash
python research/stage89-fft-nsf/test_spectral_conv.py --checkpoint /private/CrazyMita.pth --report stage89_probe.json --real-limit 36
python research/stage89-fft-nsf/validate_resblocks.py --checkpoint /private/CrazyMita.pth --report stage89_resblocks.json
python research/stage89-fft-nsf/atlas.py --checkpoint /private/CrazyMita.pth --report stage89_atlas.json
python research/stage89-fft-nsf/test_atlas.py
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 python research/stage89-fft-nsf/bench_cpu.py --checkpoint /private/CrazyMita.pth --report stage89_cpu_benchmark.json
rustc --edition=2024 --test research/stage89-fft-nsf/spectral_planner.rs -o stage89_planner_tests
./stage89_planner_tests
```
