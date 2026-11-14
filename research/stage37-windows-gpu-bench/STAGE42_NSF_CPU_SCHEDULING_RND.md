# Stage 42 — NSF execution scheduling on CubeCL CPU JIT

Date: 2026-11-10. Branch: `experiment/conv-kernel-rnd-stage33-20261110` (PR #2).
Status: **CPU R&D verified on synthetic NSF decoder; GPU, real-checkpoint and end-to-end RVC speedups not established.**

## Hypothesis and experiment

Rather than further optimizing Conv1d arithmetic, instrument the **call path and queued operations** of the real Burn NSF decoder. A two-stage NSF decoder (loaded from the repository's `tests/fixtures/synthetic_nsf_decoder.pth`) runs with `burn` / `cubecl-cpu` JIT. It exercises `conv_pre`, upsample, harmonic source convolution, residual branches and `conv_post` with real checkpoint deserialization, but **not CrazyMita's original weights**. All benchmark cases perform an entire forward followed by blocking tensor readback; reported milliseconds are warmed host-observed wall-clock times, not kernel-only times.

The benchmark compares baseline `ZeroSeeded + Default`, `FirstBranch + Default`, and six explicit scheduling placements: `FlushPerBranch`, `FlushPerStage`, `SyncPerBranch`, `FlushPerPair`, `FlushPerConv`, `SyncPerPair`. The public original `forward` remains on its previous default path, and `ResidualExecution::Parallel` is not switched to a new policy. An instrumented `profile_cpu` additionally forces blocking readback at intermediate stage/pair boundaries; its timing cannot be treated as a drop-in optimization because it changes the execution schedule.

All conditions used pinned single-CPU execution on a virtualized AMD EPYC, Rust 1.98.1, LLVM 23.1.0; observed absolute milliseconds are specific to this environment. The benchmark rotates variant order between rounds; CPU core and virtualization can affect variance.

## Repeated CPU results

Primary repetition: `nsf_scheduling_cpu_pinned_v3.json`, median of 12 iterations, 5 warmup rounds. Units are milliseconds, lower is better.

| Input frames | Default | FirstBranch | Flush/branch | Flush/stage | Sync/branch | Flush/pair | Flush/conv | Sync/pair | Default / Flush-pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 46.492 | 43.901 | 24.412 | 35.930 | 24.052 | **9.932** | 15.594 | 14.960 | **4.68x** |
| 256 | 46.293 | 44.012 | 24.271 | 35.884 | 23.996 | **13.126** | 12.702 | 15.981 | **3.53x** |
| 1024 | 48.536 | 47.891 | 25.230 | 40.172 | 32.098 | 18.423 | **16.461** | 16.462 | **2.63x** |

At 1024 frames, `FlushPerConv` was slightly faster than `FlushPerPair`, indicating that an optimal boundary is **shape/backend-dependent**. An earlier repeated run `nsf_scheduling_cpu_pinned_v2.json` used 7 iterations, 3 warmup rounds:

| Input frames | Default | Flush/pair | Default / Flush-pair |
|---|---:|---:|---:|
| 4 | 42.764 | 8.682 | 4.93x |
| 64 | 43.114 | 9.125 | 4.73x |
| 256 | 46.306 | 13.134 | 3.53x |

The original two aggregation strategies with no flush differed minimally in baseline warm runs and were not the primary improvement. The instrumented intermediate-readback profile took 7.032 ms in the v3 record but **should not be compared as an equivalent unsynchronized inference path**; it intentionally changes scheduling and inserts reads.

## Correctness and verification

- `max_abs_error = 0.0` against the unmodified baseline output for all tested scheduling variants and all measured frame lengths.
- `profiled_vs_baseline_max_abs_error = 0.0` in the detailed CPU profile.
- `cargo test --release --locked --offline --no-default-features --features flex --test parity nsf_` confirmed 7/7 relevant NSF tests PASS, including a new bitwise-equivalence test for `FirstBranch` aggregation on the Flex backend.
- CPU benchmark `profile_nsf_stage42` compiled in release and ran the full synthetic decoder.
- **CUDA and WGPU compilation/performance for Stage 42 are not claimed**, nor are full RVC PCM parity and real-checkpoint timing.

## Interpretation and cautions

This is strong **CPU-local evidence of scheduling overhead** in the Burn/CubeCL execution of long residual chains, not evidence that Conv1d FLOPs were reduced. The nearly flat 44 ms baseline over short lengths and large savings from nonblocking `Device::flush` are consistent with queue/fusion/planner overhead, but the precise compiler/runtime cause has **not** been isolated. `Device::flush()` is not equivalent to `Device::sync()`; forcing it on CUDA could worsen execution due to extra launches and lost fusion. Do not promote a global flush policy from CPU timings alone.

The complete real RVC execution path also includes HuBERT, RMVPE, feature preparation, retrieval, flow and synthesis, which have not been profiled together in this stage. No production dispatch was changed.

## Next gates

1. Compare per-operator CPU timestamps and graph/fusion group statistics from CubeCL to distinguish graph construction/compilation costs, repeated fusion and dispatch overhead.
2. Re-run pinned CPU A/B across longer input lengths and on actual NSF checkpoint data; separate resident-buffer and blocking-readback timings.
3. Add opt-in scheduling flag to Windows GPU benchmark and test `Default`, `FlushPerStage`, `FlushPerPair`, `FlushPerConv`, with strict full-output CUDA/WGPU parity checks.
4. Test backend-specific scheduling selection only when verified by stable A/B on each backend; keep production default otherwise.

## Evidence

All raw JSON records and selected console/test logs are committed in `evidence/stage42-cpu-nsf/`. Reproduction tool: `models/tts-with-rvc-rs/src/bin/profile_nsf_stage42.rs`; synthetic checkpoint is already in the repository. Do not commit private model weights/audio.
