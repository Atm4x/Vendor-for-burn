# Stage 32 — NSF parallel ResBlock A/B on CrazyMita.pth

Date: 2026-11-09. Local branch `experiment/nsf-universal-polyphase-20261109`.

## Scope

The production NSF decoder computes three independent ResBlock branches at each upsampling stage. Stage 32 parallelizes these branches using Rayon and gathers them in a deterministic original order. The public default `forward` and `forward_from_f0` behavior is unchanged. A `ResidualExecution::{Sequential,Parallel}` option was added to measure both modes within the same binary, model, and exact input. The sequential control preserves the original streaming accumulation loop; no extra `Vec` allocation is introduced into the baseline.

No changes were made to the mathematical Conv1d/ConvTranspose1d kernels. A standalone polyphase ConvTranspose1d kernel and prepacked Conv1d GEMM were investigated earlier in this stage, but were not promoted due to weak or negative performance. The real CrazyMita upsampling strides are `[10,10,2,2]`.

## Controlled CPU release A/B

Checkpoint: private `CrazyMita.pth`. FP32. Same real checkpoint, same prepared encoder/flow latent tensors and harmonic source per run. Medians of warm runs, alternating sequential and parallel execution order. Wall time for NSF forward including output extraction, excluding checkpoint load, TextEncoder and Flow.

| Frames | Rayon threads | Sequential ms | Parallel ms | Speedup | Samples |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 2 | 270.324 | 196.350 | 1.377x | 9,600 |
| 24 | 4 | 251.195 | 156.668 | 1.603x | 9,600 |
| 128 | 2 | 1,253.369 | 1,269.061 | 0.988x | 51,200 |
| 128 | 4 | 988.951 | 766.709 | 1.290x | 51,200 |

For 24 frames each median uses 5 warm pairs; for 128 frames 3 warm pairs. `max_abs` between sequential and parallel f32 outputs was exactly 0 in all runs. One earlier A/B had a serial baseline that also collected its branches into a `Vec`, and those preliminary timings must **not** supersede this final controlled run. Results are sensitive to shared CPU load and insufficient to establish platform-independent speedup.

## Regression validation

- Release unit: 15/15 PASS.
- Release Torch parity: 31/31 PASS, including the new `nsf_residual_execution_modes_are_bitwise_equal` synthetic fixture regression.
- Actual short RVC: `CrazyMita.pth`, `hubert_base.pt`, `rmvpe.pt`, `CrazyMita.index`, 10.1s input, B=2: 4/4 chunks, 404,000 int16 PCM samples at 40kHz, wall-clock 26.13s (includes model loading), max resident set 1,396,544 KiB.
- Compared with preserved Stage31 WAV: **exactly equal** 404,000/404,000 PCM samples, max difference 0, zero clipping.

The full 89.94s WAV was **not rerun** in Stage32; its acceptance gate is still open. No measured end-to-end PyTorch CPU comparison is available. No CUDA parity test was run.

## Remaining work

1. Profile nested Rayon / GEMM contention: 2-thread 128-frame speedup was not observed.
2. Benchmark NSF on an uncontended CPU at 24, 128 and 512 frames with more repetitions.
3. Check scheduling for batch sizes 1, 2, 4 and different thread counts; avoid relying on a fixed parallel scheduling threshold without data.
4. Repeat entire RVC WAV→WAV with checkpointing/resumption for long audio; compare steady-state latency with PyTorch under identical runtime conditions.
5. Test standalone native Conv1d/ConvTranspose1d algorithmic improvements only after f16, NaN, empty-channel and grouped parity tests pass; never silently substitute a slower algorithm.

Private weights, PCM, FAISS and model outputs are kept outside Git.
