# Silero Flex CPU — scratch reuse and GEMM scheduling R&D

Date: 2026-11-19. Base branch: `codex/silero-tts-cubecl-port-20261118`, commit `c37406612d478c8d138f38cfad644794b7440bda`.

## Environment

- Rust 1.100.0, Burn Flex 0.22.1, `--release --no-default-features --features reference-flex`; the provided offline `v5_ru.pt` and its exported SafeTensors weights.
- AMD EPYC 9V74 shared Linux VM, 4-core cgroup CPU quota, `RAYON_NUM_THREADS=4`.
- Persistent JSONL worker. The worker timer includes waveform Base64 conversion and excludes model loading. All series medians below discard the first inference in a new worker process.
- These experiments modify only the Silero model's local opt-in experimental paths. No source files in `vendor/` were changed.

## A. Reuse of ConvNeXt scratch buffers — rejected

The opt-in whole-host ConvNeXt pipeline already avoids intermediate Burn Tensor conversions but still allocates four host `Vec<f32>` buffers per block: depthwise+LayerNorm output, first GEMM output, second GEMM output and transposed channel-major result.

A local prototype added a single inference-scoped scratch object with three working arrays (`normalized`, `hidden`, `temporal`) and one ping-pong channel-major output. It reused those buffers across eight blocks through in-place GEMM destinations, without sharing mutable state between worker requests. This removes approximately 28 repeated output allocations per 8-block chain, without changing the mathematical operations.

**Correctness:** 48/48 Rust release library tests PASS; the new native-into-buffer path was compared bitwise with the old host pipeline on multiple shapes and repeated scratch reuse. Six direct real-weight PyTorch PCM tests (aidar, kseniya, xenia, baya, eugene; 8/24/48 kHz) passed max absolute PCM error `< 0.003`, and the scratch and control workers emitted **identical PCM hashes on all six**. The maximum PyTorch PCM deviation remained `0.000313759`.

**Performance:** two independent alternating series experiments, with `SILERO_FLEX_WHOLE_CONVNEXT=1` in both arms and `SILERO_FLEX_REUSE_SCRATCH=1` only in the prototype:

| Benchmark | Control whole-host | Scratch prototype |
|---|---:|---:|
| 16 alternating series (8 each), median of 6 warmed calls | **139.75 ms** | **139.75 ms** |
| Separate 2×2 factorial, 4 series per arm | **150.75 ms** | 155.50 ms |

The 2×2 factorial additionally measured `SILERO_FLEX_GEMM_DIRECT_WEIGHTS=1` without scratch: `154.75 ms`, and with scratch: `149.50 ms`. With only four sessions per cell and shared-host variation, this is not a reliable interaction effect. **Conclusion: no reproducible end-to-end benefit.** The scratch prototype is retained in the evidence ZIP, but is deliberately **not incorporated into production source**.

## B. GEMM thread scheduling — rejected

On top of the original `c374066` sources, a separate ephemeral experiment controlled `gemm::Parallelism` for every pointwise projection via `SILERO_FLEX_GEMM_THREADS`: `1` = serial, `2` = Rayon(2), `4` = Rayon(4), `0` = Rayon(0), and unset = existing automatic threshold (`192³` operations). This was only a local control hook, **not committed**. Each candidate was run in three separate persistent-worker sessions for each audio input, seven calls per session with the first discarded (15 sessions per input).

| GEMM mode | Short text, median | Long text, median |
|---|---:|---:|
| Current auto | **42.0 ms** | **136.0 ms** |
| Serial (1) | 53.0 ms | 194.5 ms |
| Rayon(2) | 47.0 ms | 191.5 ms |
| Rayon(4) | 45.5 ms | 138.5 ms |
| Rayon(0) | 42.5 ms | 139.0 ms |

Short input was `Привет, мир!` and long input was the same phrase repeated six times, both `aidar`, 8 kHz. The durations and output counts follow the same real-checkpoint worker contract as earlier cycles. Host noise was considerable (for example, some auto short medians varied between 39.5 and 61 ms), so these numbers establish only that the current automatic mode was not convincingly beaten. They **do** show that forcing serial or only two Rayon threads significantly worsens the long case on this host.

## Decision and next R&D

Do **not** merge the scratch or GEMM scheduling code: both add execution modes without demonstrated throughput gain. The committed change is this evidence report only. The evidence ZIP includes exact prototype sources, baseline snapshots, benchmark scripts, JSON series, test logs and 6-case PCM hashes; no model weights, generated audio, binary executables or vendored sources.

Further work should directly measure GEMM microkernel utilization, synchronization and pack costs or explore a fused pipeline that eliminates *computation or substantial memory traffic*, not just allocator calls. A more controlled dedicated CPU host would reduce timing noise. Do not treat these VM medians as portable performance guarantees.
