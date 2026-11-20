# Stage 68 — CPU profiler triage and instrumented full-WAV CUDA diagnostic

Parent Stage67: `015dcd0becaec24f2a49d0b31dd409aff4b3c3ba`. Branch: `experiment/conv-kernel-rnd-stage33-20261110`, PR #2. Date: 2026-11-12.

## 89.94-second full WAV Stage67 repeat (two Unity processes closed)

| Mode | Before closing Unity | After closing Unity |
|---|---:|---:|
| Baseline warm (Direct + stepwise GRU) | 55.722s | 37.991s |
| New combined warm (GEMM + persistent BiGRU) | 12.705s | 11.818s |
| Warm accelerated RTF | 0.14126 | 0.13140 |
| Baseline / accelerated | 4.386× | 3.214× |

The NVIDIA FB memory Used snapshot changed 6970 MiB to **2705 MiB** (RTX4060 8188MiB). First-pass accelerated time fell from 28.626s to 17.754s. Baseline audio hash is byte-identical between archives; accelerated audio hash is byte-identical between archives. Both modes retain 3,597,570 40 kHz PCM samples and 32 chunks. Baseline-versus-accelerated PCM16 RMSE 1.55193 units, max absolute 86; this is numerical agreement only, not a listening test.

## Actual use of rust-huge-profiler.zip from Library

The user-provided archive is a **Debian 13 Linux amd64** bundle containing perf, Heaptrack, Valgrind, strace, sysstat and Tracy CLI. Verified **44/44 SHA256 manifest entries**; selectively extracted Debian binaries with `dpkg-deb -x`, without system installation or modifying libc.

Performed host-side profiling on the **original repository `chunked.rs`** compiled directly with Rust 1.98.1, optimized, using 89.939s of synthetic 16kHz PCM, core=300, B=2, and a deliberately trivial 40kHz mock decoder (no ML operations):
- warm median CPU-only chunking/reflect/overlap-add ~29.65 ms (repeat ~29.28 ms); first run 70.10/42.80 ms;
- `perf stat` across seven mock passes: 226.98 ms aggregate cpu-clock, ~106,022 page faults; no CUDA;
- `heaptrack`: 862 allocations and 36 temporary allocations across seven mock passes; call stacks include Rust Vec construction and chunked CPU path.

This **does not profile real RVC inference**, FAISS, model copying or CUDA GPU kernels. It only makes an isolated CPU chunking bottleneck unlikely at the scale of 11.82 seconds. The Linux profiler tools cannot measure individual CUDA kernel durations inside the user's Windows EXE; for true device event timelines use NVIDIA Nsight or CUDA events. Tracy binaries in the ZIP do not include automatic Rust-side instrumentation.

## New Stage68 instrumented real inference (awaiting Windows RTX4060 trace)

Existing `infer-wav-full-diagnostic` with `RVC_DIAGNOSTIC_TRACE=1` already collects CUDA-synchronized HuBERT, RMVPE, FAISS, Encoder, Flow, NSF and nested stages. Added diagnostic-only spans:
- `nsf.stage_upsample`, `nsf.stage_noise_conv`, `nsf.stage_residuals_aggregate`, tagged with NSF stage index. Outer `nsf.upsample_resblocks` remains.
- `rmvpe.gru_fused_input_projection` and `rmvpe.gru_persistent_recurrent` for the new fused BiGRU; `rmvpe.gru_stepwise_forward` and `rmvpe.gru_stepwise_reverse` for baseline.

The production algorithm, weights, output, memory policy, default flags, Sequential scheduling and source chunking are unchanged. Diagnostic calls sync only when `RVC_DIAGNOSTIC_TRACE=1`.

New `research/stage37-windows-gpu-bench/windows/Run-RVC-Full-Profile-Stage68.ps1` profiles four full-WAV modes: baseline, NSF GEMM only, persistent GRU only, and both. Each mode performs a representative shape warmup and **one full 89.94s synchronous diagnostic pass**. Writes:
- mode-specific `stages.jsonl`, `stage-time-summary.csv`, `nsf-by-stage.csv`;
- complete stdout/stderr, output WAV and diagnostic JSON;
- NVIDIA GPU utilization, memory, power, SM clocks, temperature and process CPU/RSS samples;
- config/input SHA256 and consolidated `summary.json` in `RVC_STAGE68_PROFILE_RESULTS.zip`.

**Critical:** these nested diagnostic durations include extra explicit synchronizations. **Do not add nested parent/child spans**, and **do not compare diagnostic RTF to uninstrumented Stage67 ~11.82 seconds**. Group only `phase=measured`; use per-mode comparison to locate relative bottlenecks.

## Windows CMD from unpacked Stage68 CI artifact

```cmd
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Profile-Stage68.ps1 -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -CoreFrames 300 -TensorBatchSize 2 -Modes all
```

Choose `-Modes baseline-and-both` for two modes; `-Modes all` isolates the contribution from both acceleration algorithms. Output: `RVC_STAGE68_PROFILE_RESULTS.zip`.

Local validation: Rustfmt syntax parsing of modified files and `git diff --check` passed; full offline Cargo check exceeded local command timeout while recompiling dependencies and is **not claimed successful**. Windows CI must validate build and PowerShell runner. No Windows GPU profiler execution yet in this session.
