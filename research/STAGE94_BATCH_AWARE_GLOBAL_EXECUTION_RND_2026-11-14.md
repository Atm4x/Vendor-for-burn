# Stage 94 — shape-gather batching and batch-aware global RVC execution

Date: 2026-11-14. Rust 1.100.0. Parent: Stage 93 `3b10103f` / published PR #2 `fc27a4a5`.

## Finding

The Stage 93 global cost model was blind to the *actual execution grouping*. The runtime batches
only adjacent chunks of exactly equal PCM length. Silence-aware and cost-aware segmentation
introduces distinct chunk lengths, sometimes reducing effective `TensorBatchSize` to 1.

Stage 94 implements two independent opt-in changes:

1. `RVC_BATCH_SCHEDULE=shape-gather`: deterministically gather identical-length chunks from a
   bounded forward window (default 64 chunks), even when nonadjacent. Preserve original indices,
   F0/phase/noise coordinates and the **chronological order of output mixing and progress**.
   Retain all per-chunk waveform data until it is safe to drain it in order. No audio padding.
2. `RVC_CHUNK_STRATEGY=batch-aware`: restricted global search over constant-core-length lattices
   (10-frame length grid; first-cut offset up to 40 frames), evaluated against the existing
   unrestricted Stage93 cost-aware shortest-path candidate. It charges the dispatch proxy per
   **actual shape-gather batch** while retaining per-chunk linear and quadratic model work and
   the original normalized boundary-energy term. This mode automatically chooses shape-gather
   unless `RVC_BATCH_SCHEDULE` is explicitly overridden.

Neither change is enabled by default for `fixed`, `quiet-splice`, or `cost-aware`.

## Real-WAV geometry (MitaKind.wav)

Source 89.94s, filtered 16kHz PCM = 1,439,028 samples, 8,994 model frames. Core limit 900,
40-frame left/right context, 4 trailing frames; all observations are geometry/model-proxy numbers,
**not GPU timings**. See `evidence/stage94/real_wav_batch_plans.jsonl`.

| Strategy, tensor batch 2 | Chunks | Consecutive calls | Shape-gather calls | Σ model s | Σ attention area |
|---|---:|---:|---:|---:|---:|
| quiet+1 | 11 | 11 | 10 | 99.18 | 8,944,880 |
| cost-default | 10 | 7 | 6 | 98.34 | 9,670,778 |
| batch-aware default | 10 | 6 | 5 | 98.34 | 9,670,770 |
| cost-attention, attention_milli=100 | 26 | 22 | 15 | 111.78 | 4,806,104 |
| batch-aware, attention_milli=100 | 36 | 19 | 18 | 120.18 | 4,012,026 |

With batch size 4 and the attention-heavy proxy, batch-aware picks 41 chunks and 11 batch calls
(versus Stage93 cost-aware 26 chunks / 13 gather calls), with 3,773,586 quadratic-area units.
This illustrates a real structural tradeoff: reducing estimated quadratic work may increase total
context and even the number of batch calls. The score reflects a deliberately **uncalibrated**
latency proxy and should never be cited as an expected speedup.

A crucial seam risk: batch-aware first-cut offset shifts a periodic lattice, improving equal-sized
batches but sometimes picking higher-energy cut points. For example, with batch size 2 and the
default cost coefficients, mean input boundary energy rises from ~0.000382 (cost-aware) to
~0.001173 (batch-aware). **Neither waveforms nor perceived discontinuities have been validated**.

## Correctness and limitations

- Standalone Rust 1.100.0 unit harness: **28 PASS / 0 FAIL**, including bitwise mock-waveform
  assembly equality for nonadjacent batches, overlapping boundaries, original chunk IDs and
  chronological progress; Stage90/Stage93 regressions also pass.
- Whole crate `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features
  --features cpu,cuda,wgpu --lib --bin tts-with-rvc-rs`: PASS with `DOCS_RS=1` in a Linux
  offline environment (no CUDA JIT execution or GPU benchmark).
- `RVC_BATCH_SCHEDULE` absent means legacy contiguous batching except when `batch-aware`
  strategy is selected. New metadata fields `planned_tensor_batches` and `batch_schedule` are
  emitted from the *actual batch-specific plan*, not an unrelated scalar plan.
- The optimizer searches a **restricted lattice family plus the Stage93 optimum**; it is not
  a globally exact batch-constrained shortest path and is not a trained latency predictor.
- Per-sample independence of the model and bitwise GPU equivalence across batch shapes must
  still be verified. A mock decoder matching bits is not the same as exact neural-output parity.
- Batch-gather's per-chunk output buffer has a bounded lookahead window; it trades some CPU RAM
  for fewer serial batch calls and preserved chronological output assembly.

## Next R&D gate

Calibrate stage-specific, shape-and-batch latency on the target RTX (HuBERT frontend/attention,
TextEncoder, NSF), then compare Stage93 cost-aware sequential, identical plan with gather, and
Stage94 batch-aware+gather in A/B/C/C/B/A order. Measure elapsed time, seam energy on synthesized
WAV, cross-mode PCM RMSE and within-mode SHA stability. Keep Stage94 experimental until all pass.
