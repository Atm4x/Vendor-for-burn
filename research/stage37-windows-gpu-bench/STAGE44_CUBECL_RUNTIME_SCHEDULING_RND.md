# Stage 44 — CubeCL runtime batch scheduling, NSF execution topology

Date: 2026-11-10. Status: experimental CPU R&D; production defaults unchanged.

## Question

The Stage 42 NSF CPU JIT experiment obtained large wins by inserting \`flush()\` inside the decoder. Stage 43 fixed a real worker queue accounting race but did **not** reproduce that speedup. We tested whether the bottleneck exists above the CPU worker backend, in generic CubeCL scheduling and its interaction with NSF's Rayon-based parallel residual graph.

## Verified execution path

\`\`\`text
NSF DecoderStage::forward_with_execution
  ├─ Rayon Parallel OR Sequential residual branches
  └─ Burn tensor / conv ops
       └─ CubeCL runtime Client::launch (async submission)
            └─ cubecl-server SchedulerMultiStream::register
                 └─ max_tasks threshold (CPU default 8)
                      └─ execute_schedules_interleave
                           └─ SchedulerStreamBackend::flush
                                └─ CPU: CpuStream::submit (BLOCKING)
                                     └─ worker queues, dependency counters
\`\`\`

\`max_tasks\` is enforced by the **shared** \`cubecl-server\` scheduler, while the value \`8\` and the blocking \`flush\` implementation are supplied by \`cubecl-cpu\`. Thus this is a cross-layer performance problem, not solely a defective worker queue or Conv1d arithmetic.

CUDA has different \`flush\` semantics: it flushes command submission, not a CPU-style wait for all jobs to finish. CUDA performance cannot be inferred from these CPU timings.

## Experimental changes

1. \`CUBECL_CPU_MAX_TASKS\` (1..=256) overrides the default batch threshold of 8 at CPU server creation. Unset, invalid and out-of-range values retain 8. This is **not** automatically enabled for users.
2. \`CUBECL_CPU_INLINE_SINGLE_UNIT\` runs eligible single-unit kernels inline after draining earlier work. Disabled by default; this serves only as an independent dispatch-cost hypothesis.
3. \`bench_stage43_cpu\` now supports \`--sequential\` to disable Rayon residual-branch parallel construction. It also supports \`--dump-prefix\` to save complete little-endian FP32 outputs after timing, without counting file I/O in the measurement.
4. Full NSF math, checkpoint format, public inference API, and normal scheduler defaults remain unchanged.

## Test method

Native Rust 1.98.1 release binaries, CubeCL CPU JIT / LLVM 23.1.0-4, virtualized AMD EPYC 9V74. Affinity: one logical CPU (0), or five logical CPUs (0–4). Same public synthetic NSF checkpoint and inputs at 4/64/256 frames; warmup 4, then 8–10 iterations. Timings are warmed per-forward medians including the CPU output readback, *not* kernel-only latency.

Reproduction:

\`\`\`bash
cargo build --offline --locked --release -p tts-with-rvc-rs --no-default-features --features cpu --bin bench_stage43_cpu
CUBECL_CPU_MAX_TASKS=8 taskset -c 0-4 target/release/bench_stage43_cpu 10 --dump-prefix /tmp/nsf-baseline
CUBECL_CPU_MAX_TASKS=32 taskset -c 0-4 target/release/bench_stage43_cpu 10 --sequential --dump-prefix /tmp/nsf-optimized
cmp /tmp/nsf-baseline-64.f32le /tmp/nsf-optimized-64.f32le
\`\`\`

## Key A/B: five logical CPUs

| Configuration | 4 frames | 64 frames | 256 frames |
|---|---:|---:|---:|
| Original: Parallel, batch 8 | 119.970 ms | 107.942 ms | 112.130 ms |
| Sequential, batch 8 | 96.302 ms | 95.979 ms | 96.155 ms |
| Parallel, batch 32 | 39.524 ms | 37.422 ms | 28.205 ms |
| **Sequential, batch 32** | **16.133 ms** | **17.770 ms** | **19.310 ms** |

Replications on the same machine:

| Configuration | 64 frames, independent per-run medians |
|---|---|
| Parallel / batch 8 | 111.950, 107.942, 126.224 ms |
| Sequential / batch 32 | 17.770, 17.148, 16.055, 16.055 ms |
| Parallel / batch 32 | 37.422, 24.033, 22.997 ms |
| Sequential / batch 8 | 95.979, 99.967 ms |

The improvement is a roughly **6–8x reduction in full-forward latency**, *for this host, model and tested execution mode*. Do not extrapolate to a full production RVC voice conversion pipeline or to GPU.

One CPU: batch 8 / Parallel ~44.4 ms, batch 32 / Sequential ~36.1 ms. This smaller benefit shows that the major anomaly involves CPU concurrency and scheduling, not solely the numerical workload.

## Nonmonotonic batch effects

When keeping Rayon Parallel on five CPUs, the batch-8 threshold produced ~112 ms at 64 frames, batch-16 ~72 ms and batch-32 ~44 ms in one sweep. Batch-64 produced times between ~16 and ~40 ms across separate trials; batch-128 and batch-256 were not uniformly superior. Too-small batches (4 tasks) were dramatically slower (~198 ms at 64 frames on five CPUs).

These results are **not** a monotonic tuning law. A fixed large number can impair memory use, inter-stream fairness or responsiveness, and may amplify worker starvation in other graphs.

The inline single-unit route reduced one-CPU forward from ~44.5 to ~36.5 ms at 64 frames but did not improve the five-CPU case (~107.84 ms both ways). This isolates some dispatch overhead but is not the primary multi-CPU solution.

## Numerical correctness

The benchmark exported **all** FP32 output elements after the timed runs for baseline Parallel/batch-8 and optimized Sequential/batch-32. \`cmp\` reports identical bytes for all three lengths:

| Frames | SHA-256 of full FP32 buffer in both modes |
|---|---|
| 4 | \`62484aa879281b76da3b31ec5f409200b57c9e629c0f0ab58f47bd22e0061ea8\` |
| 64 | \`892aec83a6b4b7a0935487f3e183db7d47dada12a3277405d658d12c7b85358d\` |
| 256 | \`a9ae3996a9659ccd5b7bc99b6865fa37adb54dd6d12cb1476d5104d72db85f19\` |

This establishes exact numerical equivalence of the compared outputs on these fixtures; it does not guarantee parity for every shape, checkpoint or backend.

Raw numeric records: \`evidence/stage44-cpu/nsf_scheduler_experiments.json\`.

## Interpretation and next direction

A worker counter race alone was not responsible for Stage 42's dramatic flush benefit. The observed performance comes from the interaction of:
- CPU-specific blocking drain at each generic scheduler batch boundary;
- batch size and dependency-chain length;
- CPU worker queues checking readiness;
- Rayon residual-branch parallel construction above CubeCL.

A clean long-term design should **decouple enqueue/dispatch from completion waiting**, model dependency readiness explicitly and wake runnable work rather than repeatedly polling deferred tasks. The scheduler must retain buffer lifetime, write ordering, multi-stream hazards, cross-stream isolation and error propagation. Such a redesign requires its own tests; this stage adds reversible experimental controls, not a production scheduler rewrite.

## CUDA validation

A separate Windows CUDA NSF scheduling profiler build is provided by \`.github/workflows/rvc-nsf-cuda-scheduling.yml\`, with \`Run-NSF-Scheduling.ps1\`. It runs eight default/flush/sync variants and full-tensor FP32 comparisons using the public synthetic checkpoint. The RTX 4060 CUDA timing is **not available** in this report, and CPU gains must not be presented as CUDA gains.

No private CrazyMita weights, HuBERT / RMVPE checkpoints, indexes, or audio have been committed.
