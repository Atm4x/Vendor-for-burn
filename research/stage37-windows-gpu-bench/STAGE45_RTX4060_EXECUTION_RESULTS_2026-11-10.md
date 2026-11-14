# Stage 45 — RTX 4060 NSF Sequential / Parallel CUDA A/B

Date: 2026-11-10. Source: user-supplied `NSF_CUDA_EXECUTION_RESULTS.zip`, containing `nsf_cuda_execution.json`, `nsf_cuda_execution.stderr.log`, `nsf_cuda_device.txt`. Experimental PR #2. Benchmark commit `acc4573253e3fe67c77fe7fcb94b871c110e377e`.

## Critical scope

**CUDA's batching/scheduling was NOT changed to match Stage 44 CPU `CUBECL_CPU_MAX_TASKS=32`.** Stage 45 changes only the residual branch construction strategy: `ResidualExecution::Parallel` (Rayon) versus `ResidualExecution::Sequential`, with the same default CUDA runtime and no extra `flush()` / `sync()` calls. Thus this experiment cannot validate portability of the complete 6–8x CPU result.

The CPU 6–8x used **two factors simultaneously**: Sequential residual branches AND a larger CPU scheduler batch threshold (8 → 32). The CUDA execution backend does not use that CPU scheduler implementation. It uses `CudaServer` and event-based `MultiStream`; blindly setting `max_tasks=32` in CUDA would not be an equivalent experiment.

## Hardware and measurement

GPU: NVIDIA GeForce RTX 4060, 8188 MiB VRAM, driver 610.88. Synthetic NSF checkpoint, 64/256/1024 input frames, 20 timed iterations after 5 warmups, both variants alternated within a run. Each invocation measures host wall-clock submit (Burn forward/CubeCL dispatch) plus blocking output readback. Timings are **not pure GPU kernel latency**.

## End-to-end measured totals (ms)

| Frames | Parallel | Sequential | Quotient of reported medians | Median of within-round Parallel/Sequential ratios |
|---|---:|---:|---:|---:|
| 64 | 3.3159 | 2.5998 | 1.275x | **1.761x** |
| 256 | 3.2840 | 1.9899 | 1.650x | **1.768x** |
| 1024 | 2.3366 | 1.3031 | 1.793x | **1.731x** |

Within-round ratio IQRs: 64 frames 1.708–1.787x; 256 frames 1.669–1.861x; 1024 frames 1.649–1.814x. The ratio of the two overall medians differs from the paired median because the absolute times move between performance regimes during the same run. The profiler uses a nearest-rank quantile index for its published median; the paired ratio summaries were computed post hoc from the same raw 20 rounds. Report descriptive data rather than an assumed stable speedup.

### Host timing components (independently reported medians, ms)

| Frames | Parallel submit | Sequential submit | Parallel readback | Sequential readback |
|---|---:|---:|---:|---:|
| 64 | 3.0319 | 2.1387 | 0.3589 | 0.1886 |
| 256 | 2.9848 | 1.8634 | 0.3491 | 0.1969 |
| 1024 | 2.0353 | 1.1107 | 0.3013 | 0.1871 |

The majority of measured latency sits in the host-observed `forward` call. This includes dispatch and any scheduling waits in that call; it is **not evidence** that CUDA arithmetic itself has accelerated. Component medians do not necessarily sum to the separately reported total median.

### Time-series behavior

At 64 frames, Parallel drops from a median ~4.567 ms in the first ten timed rounds to ~2.092 ms in the last ten, while Sequential moves from ~2.609 ms to ~1.195 ms. Both variants experience large common-mode shifts, but the ratio of medians within each half remains ~1.75x. 256-frame timings shift in the opposite direction during the run. This warrants independent A/B repetitions with GPU clock/utilization telemetry and more controlled benchmarking.

## Correctness

- Full FP32 output parity across Sequential/Parallel: **max_abs_error=0** for all three frame sizes.
- Intra-mode repeated full output parity: **max_abs_error=0**.
- Matches the earlier Stage 45 approach of alternating the execution mode per round, not separate warmup-heavy jobs.

## Stage 45 work delivered

1. Added `models/tts-with-rvc-rs/src/bin/profile_nsf_execution.rs` implementing interleaved Parallel/Sequential A/B and complete FP32 parity, with separate submit and readback timing.
2. Added `research/stage37-windows-gpu-bench/windows/Run-NSF-Execution.ps1` to run CUDA A/B and produce a ZIP of JSON, logs and diagnostics.
3. Updated the Windows CI workflow `.github/workflows/rvc-nsf-cuda-scheduling.yml` to package the experimental profiler.
4. Confirmed Rust 1.98.1 local CPU and CUDA `cargo check`; local CPU execution A/B confirmed full parity.
5. Published the code and test harness in commit `acc4573253e3fe67c77fe7fcb94b871c110e377e`.
6. User ran RTX 4060 Windows test and returned the raw ZIP; this report records analysis of those results.

No CUDA scheduler implementation was modified; no production NSF switch, CUDA Graph implementation or kernel-performance win was claimed.

## Next R&D — correct CUDA analogue of the CPU bottleneck

1. Instrument host graph construction, CubeCL client dispatch, event/stream dependency resolution, actual CUDA kernel execution and transfer separately. A host-side `submit` stopwatch is insufficient to attribute causes.
2. Test **CUDA Graph Capture/Replay** as a potential analogue to amortizing repeated small kernel launches. It is already exposed by Burn's `burn::tensor::graph::capture` API and CubeCL CUDA graph implementation. Require `Graph::is_hardware()==true`, stable live input/weight buffers, correct refresh/write ordering and full FP32 output parity. Treat graph capture setup as a separate cost from warm replay.
3. Profile simultaneous Rayon + CubeCL CUDA stream scheduling and compare structured sequential graph generation versus concurrent graph construction without forcibly synchronizing kernels.
4. Repeat on multiple NSF shapes, both synthetic and actual RVC data when available. Do not extrapolate the synthetic checkpoint results to production.

This is a research measurement, not a production recommendation.
