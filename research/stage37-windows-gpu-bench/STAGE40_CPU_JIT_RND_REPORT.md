# Stage 40 — CubeCL CPU JIT benchmark and alternative indexing/tiling experiments

Date: 2026-11-10. Target PR: #2 (`experiment/conv-kernel-rnd-stage33-20261110`). Source baseline: Stage 39, 2ff4628f (refreshed Git Bridge export); experimental changes in this commit only. **No production dispatch changes.**

## Goal and provenance

The RTX 4060 Stage 38 measurements demonstrated that Quad was faster than Packed on CUDA for synthetic NSF shapes (+1.6%, +16.8%, +7.7%) but slower on WGPU. Stage 39 introduced QuadInterior, which had no measured hardware performance at the time. Stage 40 adds **real CubeCL CPU JIT execution**, not merely arithmetic reference or CPU CI, to rank candidate kernels before testing them on GPU. Reported timings are from this run's AMD EPYC Linux host, not the user's CPU or RTX 4060.

Machine: virtualized AMD EPYC 9V74, 5 logical CPUs exposed, AVX2/AVX-512 visible, Linux x86_64. Rust 1.98.1, CubeCL 0.11.3, Tracel LLVM 23.1.0-4 from the supplied offline archive. Vendored Cargo dependencies, `--release --offline --locked --no-default-features --features cpu`. Inputs and weights are deterministic synthetic FP32 with NSF geometries, batch=1, groups=1, stride=1, symmetric padding. Model checkpoint, audio and FAISS index are **not** used for the timings.

Commands after compilation:

```bash
taskset -c 1 target/release/rvc-cubecl-gpu-bench --backend cpu --sync-only --warmup 8 --iterations 30 --json result.json
taskset -c 1 target/release/rvc-cubecl-gpu-bench --backend cpu --warmup 8 --iterations 30 --json result_with_readback.json
```

The second mode retains the Stage 38/39 definition: launch + fresh output allocation + synchronous readback. The new `--sync-only` mode keeps output tensors resident across iterations and times launch + synchronization only; it still includes host dispatch and synchronization overhead, **not** isolated machine-code kernel execution. Packing is done once before repeated inference and excluded from steady-state times. Both modes read/compare complete outputs independently. Each single-core measurement is a per-case median of 30 iterations after 8 warmups; table entries below are the median of two independent process runs. The hypervisor and core affinity mean the absolute numbers are not a consumer CPU performance claim.

## Confirmed CPU measurements, one pinned logical CPU (ms; lower is better)

**Resident output, launch + sync** (`pair-single-1/2.json`):

| Layer | Direct | Packed/Pair | Quad | QuadInterior | QuadLinear | PairInterior |
|---|---:|---:|---:|---:|---:|---:|
| K3 256→256 L240 | 2.210 | 1.751 | 1.551 | **1.286** | 1.378 | 1.801 |
| K11/D5 128→128 L2400 | 19.001 | 15.973 | 14.595 | **11.447** | 12.566 | 16.177 |
| K11 32→32 L9600 | 5.515 | 3.857 | 3.416 | **2.634** | 2.851 | 4.053 |

QuadInterior / Packed speedup by latency: **1.362×, 1.395×, 1.464×**. Direct / QuadInterior: **1.719×, 1.660×, 2.094×**. All cases returned `status=pass`; entire output tensors were checked against Direct. Maximum absolute errors by shape were at most about `3e-6`, `8e-6`, `1e-6` (the JSON retains exact comparisons).

**Fresh output allocation + readback** (`linear-single-1/2.json`, before the PairInterior candidate):

| Layer | Direct | Packed | Quad | QuadInterior | QuadLinear |
|---|---:|---:|---:|---:|---:|
| K3 | 2.294 | 1.747 | 1.580 | **1.275** | 1.373 |
| K11/D5 | 19.617 | 16.267 | 14.782 | **12.029** | 12.566 |
| K11 | 5.723 | 4.006 | 3.525 | **2.690** | 2.873 |

An additional pair of sync-only runs without PairInterior (`sync-single-1/2.json`) reproduced QuadInterior as the leading implementation. Five-logical-CPU results (`pair-multi.json`, 30 iterations, 6 warmups) also ranked QuadInterior first or within a small margin of QuadLinear; because timing varied substantially between prior unrestricted runs, treat multi-CPU effects as less stable.

## New R&D candidate 1: QuadLinear — reject for CPU

QuadLinear shares the same four-accumulator interior branch as QuadInterior, but uses incrementing input/packed-weight cursors rather than recalculating `ci * stride` offsets within the channel loop. The candidate compiles and yields numerical parity, but single-core measurements consistently showed a **~2–10% latency regression** compared with QuadInterior. The 5-core sample was mixed. No claimed CUDA benefit and no production promotion.

This suggests LLVM JIT already optimizes the indexed loop effectively on the tested machine, or that the modified induction variables interfere with its code generation. This is a hypothesis, not proven by assembly inspection.

## New R&D candidate 2: PairInterior — reject for CPU

PairInterior retains the two temporal accumulators of Packed, and applies the Stage 39 one-time full-interior bounds predicate, preserving the original checked path at edges. CPU parity passes, but it was **slower than Packed** for each NSF shape on both pinned runs, and much slower than QuadInterior. Simply reducing bounds checks is not universally beneficial without enough temporal reuse. CUDA/WGPU behavior remains unmeasured; do not assume the CPU rejection applies to WGPU.

## What to optimize next

1. **QuadInterior is the CPU-winning candidate** across all tested NSF shapes. Keep Direct, Packed, Quad and QuadInterior independently selectable; do not force GPU dispatch based on CPU alone.
2. Measure QuadInterior and PairInterior separately on RTX 4060 CUDA and WGPU with full-tensor parity, warmup, multiple runs, plus generated kernel/code profiling where available. CPU speedup does **not** imply proportional CUDA speedup.
3. Study repeated scalar input loads across independent output-channel vectors and the output-stationary work-item mapping. Optimizing input reuse/coalescing or output-channel tiling is a stronger next architectural hypothesis than incrementing address counters alone.
4. Add kernel-device timing (CUDA events or equivalent) independent of host allocation/readback; distinguish launch + sync from kernel-only execution. Profile registers/spills, instruction count, cache hit rate and occupancy before increasing temporal tile beyond four.
5. Compare full resident NSF inference and weight-pack amortization on checkpoint data only after isolated operator speed and parity are reliable.

## Reproducibility and scope

The repository includes the CPU feature/CLI path, both timing modes, complete synthetic output parity checks for all six algorithms, and seven raw JSON reports in `evidence/stage40_cpu`. No private checkpoint, HuBERT/RMVPE weights, index or user audio is committed. All six variants are **experimental**; the existing production RVC kernel selection remains untouched.
