# Silero v5 RU — opt-in Flex depthwise + LayerNorm fusion

Date: 2026-11-19. Base commit: `74fdc17f666b906b46779d41ecf9aae5fea1108b`.

## Goal and implementation

After the native Flex depthwise CPU optimization, the model still executes eight separate LayerNorm operators, each requiring an intermediate channel-major Burn tensor and conversion to time-major layout. This experiment introduces an explicit `ConvNeXtPreprocess` boundary that owns both operators and provides two implementations:

- `Separate`: the existing `DepthwiseEngine` (native Flex or Burn grouped convolution), followed by the original Burn `layer_norm` with stored Tensor weights.
- `Flex(FlexDepthwiseNorm)`: one host-side pass over each `[batch, time, channels]` row computes channelwise padded depthwise convolution and population-variance LayerNorm with f32 affine parameters. The output goes directly to the pointwise GEMM in time-major layout. Rayon parallelizes independent time rows for output sizes >= 32,768 elements.

The fused path is **opt-in**, enabled by `SILERO_FLEX_FUSED_DEPTHWISE_NORM=1` before model load, only for `reference-flex` on Flex CPU. `SILERO_FLEX_TENSOR_DEPTHWISE=1` takes precedence and explicitly selects the standard tensor depthwise path. The original default behavior and CubeCL implementations are unchanged. The fused model owns one copy of each weight set, not both native and tensor forms.

The native kernel preserves convolution accumulation order and f32 arithmetic, but the LayerNorm reduction order differs from Burn Flex SIMD, so bitwise identity is not required. A direct unit test compares fused output with native depthwise followed by Burn LayerNorm on five shapes, including 512-channel and single-time-step edge cases; threshold: max absolute deviation `2e-5`.

## Actual Rust/PyTorch correctness

Rust 1.100.0, Burn 0.22.1, `--release --no-default-features --features reference-flex`. Real offline `v5_ru.pt` with the supplied SafeTensors and original `torch.package` implementation, PyTorch CPU 2.10.

**46/46 release Rust library tests passed.** Six paired persistent-worker inference scenarios passed max absolute PCM error threshold `0.003` with identical sample counts. The existing and fused Rust routes do **not** produce bitwise-identical PCM; both stay within the same strict reference threshold.

| Speaker | Sample rate | Max abs PCM error, separate | Max abs PCM error, fused |
|---|---:|---:|---:|
| aidar (short) | 8 kHz | 0.00004656 | 0.00004676 |
| kseniya (short) | 8 kHz | 0.00006343 | 0.00006480 |
| xenia (long) | 8 kHz | 0.00004850 | 0.00005008 |
| baya (long) | 24 kHz | 0.00002825 | 0.00003000 |
| eugene (long) | 48 kHz | 0.00031400 | 0.00031376 |
| aidar (long) | 8 kHz | 0.00009276 | 0.00007932 |

## A/B: CPU performance

Environment: shared AMD EPYC 9V74 Linux VM with a quota of four cores, `RAYON_NUM_THREADS=4`; original model; `aidar`, long repeated `Привет, мир!` text, 8 kHz, 43,800 samples (5.475 seconds audio). Both paths were measured in the exact same release binary, switched by environment before model load. Eight alternating persistent-worker series, seven calls each, first call excluded. All worker durations include Base64 output serialization and exclude model load.

| Series | Separate, warm median ms | Fused, warm median ms |
|---:|---:|---:|
| 1 | 178.5 | — |
| 2 | — | 176.5 |
| 3 | — | 184.5 |
| 4 | 193.0 | — |
| 5 | — | 183.5 |
| 6 | 165.0 | — |
| 7 | 138.5 | — |
| 8 | — | 124.0 |
| **Median of four series** | **171.75** | **180.00** |

Instrumentation within those runs: sum of eight depthwise+LayerNorm spans, warm series medians **13.254 ms separate** vs **11.260 ms fused**, or approximately **1.18× faster for that stage**. A previous prototype, before the clean `ConvNeXtPreprocess` refactor, reported 12.271 vs 10.616 ms for the stage and full-TTS medians of 150.25 vs 144.25 ms. These conflicting end-to-end outcomes are compatible with shared-VM scheduling noise; there is **no established whole-inference speedup**.

A short sample (four alternating series) showed 50.75 ms vs 42.75 ms full TTS and 4.848 vs 2.898 ms preprocessing, but the sample size and 40–61 ms baseline spread prohibit a stable short-input throughput claim.

## Integration with upstream GEMM weight layout experiment

During the same cycle, the target branch advanced to `ba801c17cfbf58326893195027b58cd0ef4b9bcc`, adding opt-in `SILERO_FLEX_GEMM_DIRECT_WEIGHTS=1`. Its `flex_pointwise.rs` was reproduced locally byte-identically (Git blob `89868022dea03ba0c5df9486fc90c30e19b67222`) and used for integration verification. **47/47 Rust release library tests passed** with both experimental implementations present in the source tree.

Eight alternating release worker sessions (two per configuration), seven inference calls each, first excluded; flags are `SILERO_FLEX_FUSED_DEPTHWISE_NORM` / `SILERO_FLEX_GEMM_DIRECT_WEIGHTS`:

| Fusion | Direct GEMM weights | Full-TTS median of two runs | Depthwise+norm median | Pointwise GEMM median |
|---|---|---:|---:|---:|
| off | off | 151.8 ms | 11.840 ms | 59.335 ms |
| on | off | 145.2 ms | 10.345 ms | 58.135 ms |
| off | on | 142.5 ms | 11.177 ms | 57.801 ms |
| on | on | 140.5 ms | 10.224 ms | 56.686 ms |

The two opt-in modes can be combined and showed no obvious negative interaction. However, **two trials per configuration are not enough** to establish production-level latency improvement, and shared CPU scheduling varies widely. The 47/47 test suite and six actual PyTorch parity comparisons were rerun on the integrated source tree with `SILERO_FLEX_GEMM_DIRECT_WEIGHTS=1`; both fused and separate preprocessing remained within max PCM error 0.003 across 8/24/48 kHz and five speakers. The maximum observed PyTorch deviation remained approximately 0.000314.

The GEMM source file belongs to the prior commit and is not modified again by this patch. The accompanying evidence includes the factorial CSV/JSON, strict PCM gate and integrated tests.

## Decision

Retain an **opt-in experimental** implementation and the unmodified default. Publish the source with its tests and negative end-to-end evidence so future CPU-specific measurements can compare or remove it. Do not make the opt-in default without tighter CPU pinning, benchmark stability and broader parity inputs. The next useful optimization is eliminating another Burn tensor roundtrip between native preprocessing and native pointwise GEMM, ideally by passing a host slice within a single owned Flex ConvNeXt block.

## Source / vendor scope

This change touches only `models/silero-tts/src/vocoder.rs`, `models/silero-tts/src/flex_depthwise.rs` and this report. `vendor/burn-flex` and `vendor/cubecl` operator source code is unchanged. Local `.cargo-checksum.json` repairs used to enable offline builds are not an algorithmic patch and must not be committed as a surrogate for source changes.
