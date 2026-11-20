# Stage 45 — NSF residual execution topology: CPU sanity and CUDA A/B

Date: 2026-11-10. Repository: `Atm4x/Vendor-for-burn`, experiment PR #2. This stage adds a **standalone, opt-in benchmark**; it does not change production inference or NSF kernels.

## Research question

Stage 44 found 6–8× CPU acceleration when changing two independent factors at once: CubeCL CPU scheduler batch threshold `8 → 32`, and NSF residual construction `Parallel (Rayon) → Sequential`. Stage 42 found that inserting `flush()` is helpful for the CPU backend, but the user's RTX 4060 tests showed that it increases CUDA latency. To determine what transfers across backends, we must isolate the backend-agnostic factor: **residual construction topology**.

## New experiment

`profile_nsf_execution` compares the same synthetic-checkpoint NSF forward under:

- `Parallel`: the original Rayon residual branch execution.
- `Sequential`: serial branch construction with identical residual arithmetic and input tensors.

The comparison deliberately avoids explicit `flush()` and `sync()` inside the decoder. It does not change the CubeCL CPU batch threshold. Modes are executed in **alternating order each round** after common warmup in a single process. `--mode both` checks the complete FP32 output tensors; `--mode sequential` and `--mode parallel` can be run separately.

Timings are divided into:

1. `submit_ms`: host wall time spent in `decoder.forward_with_execution`, including any synchronous work inside the Burn/CubeCL runtime.
2. `readback_ms`: time to obtain the complete output tensor on CPU, including outstanding GPU computation and transfer if needed.
3. `total_ms`: end-to-end forward + blocking output readback.

These are **host-timed phases**, not pure GPU kernel execution times. Per-iteration samples, medians, minima, and p90 are written to JSON. Output parity is checked across the entire FP32 tensor, and repeated outputs within each mode are checked separately.

## Local CPU sanity (new benchmark)

Virtualized AMD EPYC, 5 available CPUs; Rust 1.98.1, CubeCL CPU JIT, `max_tasks=8` (default); 3 warmups + 8 measurements. This is an isolated CPU tool validation on the available local source, not an extrapolation to RTX 4060.

| NSF frames | Parallel full (ms) | Sequential full (ms) | Parallel submit (ms) | Sequential submit (ms) | Cross-mode max_abs_error |
|---|---:|---:|---:|---:|---:|
| 64 | 107.908 | 96.042 | 107.454 | 95.593 | 0 |
| 256 | 120.294 | 92.004 | 119.690 | 91.598 | 0 |

The CPU host-side forward call accounts for almost all latency in this test. This is consistent with the current CPU scheduling/flush behavior. For CUDA this decomposition must be measured independently; it may look different.

## Prior RTX 4060 CUDA results (user-provided, Stage 42 profiler)

12 measurements after 4 warmups, full synthetic NSF forward (ms):

| NSF frames | Default (sequential) | FlushPerPair | FlushPerConv | FirstBranch | max_abs_error |
|---|---:|---:|---:|---:|---:|
| 64 | 2.0040 | 2.2707 | 2.6653 | 2.1117 | 0 |
| 256 | 2.1164 | 2.4261 | 2.6192 | 2.0289 | 0 |
| 1024 | 2.0974 | 2.4249 | 2.6334 | 1.4760 | 0 |

**Critical distinction:** in the previous GPU profiler, `Default` was `ResidualExecution::Sequential`. These numbers **do not** provide a Parallel GPU baseline. Thus the Stage 44 6–8× CPU speedup has neither been reproduced nor disproved on CUDA. Further GPU performance claims require the new A/B.

## How to run on RTX 4060

Download the `rvc-nsf-cuda-execution-windows-x64` artifact from the updated `RVC NSF CUDA scheduling and execution A/B package` workflow. Extract it, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-NSF-Execution.ps1 -Iterations 20 -Warmup 5 -Frames "64,256,1024" -Mode Both
```

For independent runs, change `-Mode Both` to `-Mode Parallel` or `-Mode Sequential`. These independent mode runs intentionally **do not claim cross-mode parity**; use `Both` to validate numerical equivalence.

Send back `NSF_CUDA_EXECUTION_RESULTS.zip`. The archive contains JSON measurements, stderr and `nvidia-smi` diagnostics.

## Next R&D decision tree

- If Parallel > Sequential on CUDA and both have similar GPU readback time, optimize host-side Rayon graph construction, unnecessary interleaved launches, per-branch allocations, and graph dispatch.
- If CUDA differences emerge in readback rather than submit, investigate GPU stream scheduling and concurrent kernels, not CPU worker pools.
- If the two CUDA modes are equivalent, the ~6–8× CPU gain is primarily a CPU-specific oversubscription/batch-scheduling issue. Do not alter GPU production defaults.
- Investigate actual decoder layouts, weight prepacking lifetime, and residual aggregation only with full end-to-end NSF benchmarks and strict output parity; isolated kernel speedups are not sufficient.

This stage does not claim new CUDA gains or replace the CPU scheduling design. It provides an experiment to isolate which layer is responsible.