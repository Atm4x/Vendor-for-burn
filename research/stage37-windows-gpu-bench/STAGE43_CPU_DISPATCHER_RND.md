# Stage 43 — CubeCL CPU dispatcher accounting and NSF A/B

Date: 2026-11-10. Branch: `experiment/conv-kernel-rnd-stage33-20261110`. Status: correctness fix verified; performance hypothesis not supported.

## Motivation

Stage 42 showed large speedups when inserting explicit `flush()` boundaries into the NSF decoder (up to 4.68x in the prior CPU experiment). This stage tests whether a race in `cubecl-cpu 0.11.3` dispatcher queue-depth accounting explains that difference.

## Source-level bug and change

In `DispatcherScheduler::send`, the original implementation sent a task to an `mpsc` channel and only then incremented the `lens[target]` counter. A worker could receive and finish the task before the increment and execute `fetch_sub(1)` on zero. This transiently underflows the unsigned counter to `usize::MAX`, corrupting the heuristic used to choose the least-loaded worker.

The patch reserves the queue slot **before** publishing the task to the worker. On channel disconnection it rolls back the increment and reports an error instead of silently leaving a phantom queued task. Two `#[cfg(test)]` unit tests cover early worker receipt and disconnected channel rollback.

This is a correctness fix, not a replacement for dependency-aware scheduling.

## Reproducibility

Hardware: virtualized AMD EPYC 9V74, 5 visible logical CPUs (affinity CPU 0, or CPUs 0–4). Toolchain: Rust 1.98.1, tracel LLVM 23.1.0-4, release build with offline vendored dependencies.

Build command:

```bash
cargo build --offline --locked --release -p tts-with-rvc-rs --no-default-features --features cpu --bin bench_stage43_cpu
```

The benchmark runs the full synthetic-checkpoint NSF decoder forward, reads the entire output on CPU, and reports the median from 12 iterations after 4 warmup iterations for 4, 64 and 256 frames. It does not isolate kernel-only time. The two variants use independently linked executables; the vendored `cubecl-cpu` library was explicitly invalidated between builds.

The local benchmark tree was restored from the Stage 40 `2ff4628f` source with Stage 40/41 experiment overlays; the Stage 42 scheduling instrumentation was not present in this local checkout. The default forward graph was held constant within this A/B. The published Stage 42 results should not be treated as a direct same-binary baseline for the new numbers.

## A/B results

Medians of independent per-run medians; milliseconds, lower is better. Full nine-trial data: `evidence/stage43-cpu/nsf_dispatcher_ab.json`.

| Shape | 1 CPU baseline | 1 CPU fixed | 5 CPU baseline | 5 CPU fixed |
|---|---:|---:|---:|---:|
| NSF 4 frames | 44.477 | 44.338 | 108.389 | 113.966 |
| NSF 64 frames | 46.204 | 44.309 | 104.032 | 105.449 |
| NSF 256 frames | 46.141 | 44.009 | 111.911 | 109.955 |

Sample counts: 1 CPU baseline 2, 1 CPU fixed 2, 5 CPU baseline 3, 5 CPU fixed 2. One earlier aggregate A/B harness was interrupted before its eighth trial; independent follow-up runs completed. These are not statistically controlled confidence intervals.

Across all successfully completed trials the output sums and sample counts were identical for each tested input length. This is not a full cross-process bitwise-tensor comparison. Two isolated atomic-accounting regression tests passed (2/2); the release NSF CPU JIT build passed.

## Interpretation

There is **no reproducible material NSF speedup** from the counter fix. At 5 CPUs, medians still cluster around 100–114 ms and remain markedly slower than single-core execution. Results fluctuate on the virtualized host. The Stage 42 flush benefit cannot be attributed to this race alone.

The next R&D should focus on actual execution topology: deferred tasks with unmet dependencies in each worker's `aside` queue, wake-up/backoff behavior, bounded worker queues, and routing of runnable work. Profile task readiness, queue age, worker idle time and dependency-chain critical paths separately. Do not add compulsory flush or change production decoder dispatch based on this test.

No private checkpoints, voice/audio files, generated PCM or proprietary indexes were added. The mathematical NSF operations and public APIs remain unchanged.
