# Silero V5 RU — native Flex depthwise Conv1D

Date: 2026-11-19. Baseline GitHub HEAD: `43672bcbdf900f01f79b15855a53235156106830`.
Environment: Rust 1.100.0, Burn 0.22.1, release `reference-flex`, four Rayon threads,
AMD EPYC 9V74 virtual machine (four-core cgroup quota). Original offline `v5_ru.pt`.

## Finding

After fusing the spectral head, profiling the eight ConvNeXt blocks on the currently
available virtual CPU revealed about 37–42 ms per long inference inside their
`depthwise_ms` spans. The generic grouped `conv1d` dispatch is a better immediate
target than caching weights for the already-specialized native GEMM. The previous
GEMM RHS packing investigation did not establish a packing bottleneck.

## Accepted implementation

`flex_depthwise.rs` contains a Flex-only native depthwise cross-correlation for
`[B, C, T]` float32 input, `[C, 1, K]` weights, odd kernel size, stride 1,
dilation 1 and symmetric `(K-1)/2` padding. It uses a preloaded host weight
buffer and parallelizes independent channel rows with Rayon above 32,768 output
elements. Border conditions are explicitly clipped, and output remains `[B,C,T]`.

`Vocoder::ConvNeXtBlock` now owns `DepthwiseEngine`: `Tensor { weight, bias }`
or `Flex(FlexDepthwise)`, rather than holding redundant Tensor and native copies
of the weights. At model load, the Flex implementation is selected by default
only for the Flex device; `SILERO_FLEX_TENSOR_DEPTHWISE=1` selects the previous
Burn grouped-convolution path. CubeCL backends continue to use the Tensor
variant. The pointwise, spectral, ISTFT and frontend contracts are unchanged.

## CPU performance

Text: `Привет, мир!` repeated six times with spaces, `aidar`, 8 kHz,
43,800 samples, 5.475 seconds of generated audio. Worker inference timings
include inline Base64 encoding. Eight alternating persistent-worker release
series (6 inferences per series; first call excluded), in this order:
Tensor, Native, Native, Tensor, Tensor, Native, Native, Tensor.

| Series | Tensor median (ms) | Native median (ms) |
|---:|---:|---:|
| 1 | 195 | — |
| 2 | — | 217 |
| 3 | — | 185 |
| 4 | 223 | — |
| 5 | 197 | — |
| 6 | — | 172 |
| 7 | — | 179 |
| 8 | 208 | — |
| Median of series medians | **202.5** | **182.0** |

Observed end-to-end speedup = **1.113x** (10.1% latency reduction);
RTF **0.0370 → 0.0332**. The shared VM exhibited substantial run-to-run
fluctuation, so this is bounded local evidence, not a stable cross-machine speed
guarantee.

Stage instrumentation from independent 4-call sessions (first excluded) measured
the sum of eight `depthwise_ms` spans:

| Path | Warm totals (ms) | Warm median (ms) |
|---|---|---:|
| Burn grouped conv1d | 40.946, 37.240, 41.737 | **40.946** |
| Native channel-wise | 8.146, 12.374, 8.057 | **8.146** |

The stage is **5.03x** faster on median, saving **32.80 ms** within the
measured spans. End-to-end savings are smaller because other stages and shared
CPU scheduling vary.

## Correctness

Release library: **45 passed, zero failed**. New test
`native_depthwise_matches_burn_for_batches_and_temporal_boundaries` compares
native convolution with Burn grouped conv1d on five shapes, including
512-channel 97-frame, two-batch, and one-frame cases; maximum allowed test
deviation `2e-5`.

The original PyTorch `PackageImporter` model was run independently on six real
scenarios, for both Tensor fallback and native depthwise. Sample counts matched,
maximum absolute PCM errors remained below `0.003`, and native/Tensor output
buffers had **identical SHA-256 hashes in all six cases**:

| Scenario | Rate | Samples | Max error against PyTorch |
|---|---:|---:|---:|
| aidar, short | 8 kHz | 7,900 | 0.00004656 |
| kseniya, short | 8 kHz | 9,600 | 0.00006343 |
| xenia, long | 8 kHz | 54,300 | 0.00004850 |
| baya, long | 24 kHz | 147,900 | 0.00002825 |
| eugene, long | 48 kHz | 246,600 | 0.00031400 |
| aidar, long | 8 kHz | 43,800 | 0.00009276 |

The real-checkpoint tests were performed outside the fixture-gated Rust parity
tests. A final post-refactor release suite again passed 45/45, and a second
end-to-end PCM round confirmed bitwise equality across all six cases.

## Reproducibility

Build inside `models/silero-tts`:

```bash
cargo test --offline --release --no-default-features --features reference-flex --lib
cargo build --offline --release --no-default-features --features reference-flex --bin silero-tts
```

Use `tools/benchmark_release_flex.py` with `--binary`, `--model-dir`,
`--reference-pt`, `--threads 4`, `--repeats 6`, and `--max-pcm-error 0.003`.
Compare default native execution with `SILERO_FLEX_TENSOR_DEPTHWISE=1`;
set `SILERO_VOCODER_TIMING=1` for per-block stage timings.

## Scope and next steps

No model weights, generated audio, executables, vendored checksum changes,
AVX-512 experiments, or other model implementations are part of this change.

Further performance work should avoid speculative GEMM RHS packing and instead
profile GEMM microkernel utilization, LHS stride layout, and a possible
depthwise-plus-layernorm fusion as a separate, gated experiment. Retain direct
PyTorch parity and on/off release A/B testing for every candidate.
