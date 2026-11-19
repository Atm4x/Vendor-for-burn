# Silero V5 RU — CPU R&D: depthwise+LayerNorm fusion and GEMM weight strides

Date: 2026-11-19. Base: `74fdc17f666b906b46779d41ecf9aae5fea1108b`. Rust 1.100.0, Burn 0.22.1, real `v5_ru.pt` model, release Flex CPU, four Rayon threads. Benchmark host: shared/virtualized AMD EPYC 9V74; CPU cgroup quota four cores.

## 1. Two rejected depthwise + LayerNorm fusions

The standalone native depthwise convolution currently costs about 8 ms for all eight ConvNeXt blocks on a long phrase, and Burn Flex LayerNorm about 4–5 ms. Two experimental fusion designs were built locally:

1. A direct row-wise native kernel producing `[batch,time,channels]` while applying same-padded depthwise followed by biased-variance normalization.
2. A two-pass design: native channel-major convolution into a host buffer, then contiguous output with normalization, bypassing intermediate Burn tensor construction.

Both passed 46/46 release tests (including input sizes from one frame up to 512 channels), and all six actual-checkpoint PyTorch parity scenarios passed max-abs PCM error 0.003. The direct design produced bitwise identical PCM to the staged baseline in all six cases; the two-pass variant produced small PCM differences but met the threshold.

However, neither improved the warmed end-to-end latency, measured over eight alternating worker sessions (four per variant):

| Candidate | Staged median of run medians | Fusion median | Outcome |
|---|---:|---:|---|
| Direct row-wise convolution+norm | 144 ms | 153 ms | 6.3% slower |
| Two-pass channel-major convolution+norm | 134.5 ms | 143 ms | 6.3% slower |

The shared CPU exhibited large run-to-run variance. Both proposals were **rejected and omitted from the commit**. The original native depthwise and Burn layer-normalization architecture is retained.

## 2. GEMM physical weight layout experiment

The existing Flex pointwise implementation stores each projection's weights as output-major contiguous `[out,in]` and passes the matrix to `gemm::gemm` with `rhs_cs=in`, `rhs_rs=1`. The GEMM implementation transposes row-major destination GEMM internally. This makes the stored weights the internal LHS operand with `lhs_rs=in`, which can require re-packing during inference.

An alternative stores the original `[in,out]` float32 data (output channel contiguous), and passes `rhs_cs=1`, `rhs_rs=out`. After GEMM's internal orientation adjustment, the static weights have `lhs_rs=1`. This can avoid re-packing in eligible microkernel shapes and preserves the exact matrix product.

The alternative is **opt-in** via `SILERO_FLEX_GEMM_DIRECT_WEIGHTS=1`, selected when the model loads. Default layout and non-Flex paths remain unchanged. The code owns exactly one chosen contiguous representation per projection.

### Tests and PCM parity

The new unit test compares both layouts on five matrices: `(m,k,n)=(1,9,17),(7,16,32),(97,512,1536),(173,1536,512),(257,512,1536)`. All 46 release tests passed, zero failed.

Six independent end-to-end tests against the original PyTorch model passed for speakers aidar, kseniya, xenia, baya, eugene, at 8/24/48 kHz and both short and long Russian text. The new layout produced **bitwise identical PCM to the previous GEMM layout for every tested case**. Max absolute deviations from PyTorch stayed in the range `2.825e-5`–`3.140e-4`.

### End-to-end CPU measurements

Eight alternating 8-inference release sessions (four per layout, first call excluded):

| Layout | Run medians (ms) | Median of run medians |
|---|---|---:|
| Original | 152, 188, 171, 196 | **179.5** |
| New output-contiguous weights | 136, 164, 162, 190 | **163.0** |

The observed end-to-end difference is about **9.2% lower latency** on this series, but the significant shared-host drift means it is **not** yet a robust production-speedup claim.

More localized instrumentation measuring all 16 pointwise projection spans per inference:

| Series | Original (warm median) | New layout (warm median) |
|---|---:|---:|
| First pair | 56.002 ms | 55.013 ms |
| Second pair | 57.396 ms | 53.550 ms |

This supports a small repeatable saving directly in GEMM, approximately **1–4 ms** per long phrase on the tested host. It does not support attributing the entire observed 16.5 ms end-to-end difference to GEMM.

## 3. Decision and next step

Keep the old weight layout as default until the new layout is tested on a controlled physical CPU. Publish the new opt-in layout and parity regression test only. Do not commit the rejected depthwise+LayerNorm fusion or any vendor changes.

Next CPU work: profiler-guided investigation of internal GEMM packing flags, microkernel stride behavior, and weight reuse under different frame counts. Benchmark full TTS with fixed CPU affinity/host load and preserve the strict PyTorch PCM gate.

To reproduce a test: `cargo test --offline --release --no-default-features --features reference-flex --lib`.
For the opt-in performance run set `RAYON_NUM_THREADS=4 SILERO_FLEX_GEMM_DIRECT_WEIGHTS=1`, and compare to the same binary without that variable. For layer timings set `SILERO_VOCODER_TIMING=1`. Run `tools/benchmark_release_flex.py` using the real model.
