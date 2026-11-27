# Silero Flex CPU — fused GEMM epilogues

Date: 2026-11-19. Base commit: aa873b8025744bb0400e31f331920c239624cc55. Rust 1.100.0 / Burn Flex 0.22.1 release; PyTorch 2.10 CPU reference with original Silero v5_ru.pt weights. Host: shared virtualized AMD EPYC 9V74 with four-core cgroup quota.

## Implementation

The existing two 512→1536→512 native pointwise GEMMs are unchanged. The first GEMM output is processed once for bias addition plus fast GELU; the second output is processed once for bias addition, gamma multiplication and residual addition. The floating-point operation order is preserved and all host buffers remain f32. Rayon row parallelism retains the earlier size gating.

The optimized Flex epilogue is chosen at model load by default. Set SILERO_FLEX_STAGED_EPILOGUE=1 *before loading* to restore the previous sequentially staged operations. The existing SILERO_FLEX_TENSOR_POINTWISE=1 remains a separate full-tensor fallback. CubeCL and CUDA code paths were not changed.

## Full TTS release latency

The long test text is six repetitions of "Привет, мир!" (77 characters), aidar speaker, 8 kHz, 43,800 output samples = 5.475 s. Each A/B series used warmed persistent-worker calls; the first inference call of each run was discarded.

| Run | Staged epilogue | Fused epilogue |
|---:|---:|---:|
| 1 | 156 ms | — |
| 2 | — | 139 ms |
| 3 | — | 146 ms |
| 4 | 151 ms | — |
| 5 | 156 ms | — |
| 6 | — | 174 ms |
| 7 | — | 151 ms |
| 8 | 150 ms | — |
| **Median of four run medians** | **153.5 ms** | **148.5 ms** |

Observed median speedup: 1.034× (approximately 3.3% latency reduction). Due to shared-host variability, this is not a robust claim about general-purpose production throughput. A separately conducted 14/16 partial factorial run had four staged medians of 161, 154.5, 160, 152.5 ms and four fused medians of 144, 146.5, 170, 153 ms; the entire factorial run did not complete and cannot establish a stronger result.

Short text "Привет, мир!" (aidar, 8 kHz): staged median 44.5 ms, fused median 44.0 ms in eight alternating runs (four per mode). This difference is beneath host noise.

## PCM parity and tests

The staged and fused implementations produced byte-identical PCM on each of five real-checkpoint worker comparisons. Both matched original PyTorch output, with the same maximum absolute error:

| Speaker | Rate | Text | Max PCM error |
|---|---:|---|---:|
| aidar | 8 kHz | "Привет, мир!" | 0.00004656 |
| kseniya | 8 kHz | "На двери замок." | 0.00006343 |
| xenia | 8 kHz | "Привет, мир!" × 6 | 0.00004850 |
| baya | 24 kHz | "Привет, мир!" × 6 | 0.00002825 |
| eugene | 48 kHz | "Привет, мир!" × 6 | 0.00031400 |

All pass max absolute PCM error below 0.003, with exact output lengths. A new unit test compares fused and separately staged epilogues bit-for-bit on 67×1536 and 67×512 f32 matrices, including Rayon execution. Release library results: 43 passed, 0 failed. Tests requiring SILERO_PARITY_DIR remain fixture-gated; the five direct worker/PyTorch checks use real model weights independently.

## Rejected experiment

A separate opt-in native GEMM path for the final vocoder spectral head gave inconsistent full-TTS latency across alternating runs, with only about 1.6 ms median gain in direct head-stage timing. Despite passing PCM comparisons, this implementation is not included in the accepted commit. The previous RHS-packing-threshold experiment also failed to show repeatable gains.

## Scope and follow-up

No new weights, binaries, WAV, vendored crates or checksum normalization are committed. The restricted Linux environment required 432 local vendored checksum corrections to compile, but these are build-only. The next worthwhile work is a genuinely reusable weight-packing scheme or fused depthwise/normalization path with controlled CPU measurements and the same PCM parity gate.
