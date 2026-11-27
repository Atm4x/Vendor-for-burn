# Silero V5 RU — whole ConvNeXt host pipeline (Flex CPU)

Date: 2026-11-19. Baseline commit: `9fd798d6d2168e1926c9bc4e10442c95d9c481ba` on `codex/silero-tts-cubecl-port-20261118`.

## Scope and design

The existing Flex implementation already has native depthwise+LayerNorm (`FlexDepthwiseNorm`) and native GEMM+GELU+residual (`FlexPointwise`), but when composed by the ordinary `ConvNeXtBlock`, the two stages convert back to Burn tensors. The vocoder then repeats those conversions for eight blocks.

The new `FlexConvNeXt` composes the two native stages over host `f32` slices. A `ConvNeXtBlock` owns **either** the standard Tensor/native-component pair **or** the whole native block, never duplicate copies of the model weights. When all eight blocks use `FlexConvNeXt`, `Vocoder::forward` converts the initial embedding output to host memory once, passes channel-major `Vec<f32>` outputs between the eight blocks, and reconstructs one Burn tensor after the last block. The intermediate depthwise+normalization output remains in time-major host layout for the first pointwise GEMM.

The native branch is **opt-in**: `SILERO_FLEX_WHOLE_CONVNEXT=1`, set before model load. It is available only with the `reference-flex` feature and the Flex device. `SILERO_FLEX_TENSOR_DEPTHWISE=1` and `SILERO_FLEX_TENSOR_POINTWISE=1` take precedence and prevent whole-block activation. Existing standard, tensor fallback, non-Flex backend and CubeCL paths remain unchanged. Native weight layout selection (`SILERO_FLEX_GEMM_DIRECT_WEIGHTS=1`) and staged GEMM epilogue selection remain supported. The whole-block path uses the same fused depthwise+LayerNorm math as the previous opt-in `SILERO_FLEX_FUSED_DEPTHWISE_NORM=1` mode; it is not expected to match the unfused LayerNorm bit-for-bit.

No Burn/CubeCL vendored source files were changed.

## Release testing

- Rust 1.100.0, Burn 0.22.1, offline release `reference-flex`, `-C target-cpu=native`.
- CPU: shared virtualized AMD EPYC Linux instance, quota of four cores, four Rayon threads.
- Library: **48/48 PASS** after the final source changes. The new `host_chain_matches_tensor_boundary_bitwise` test covers `[1,8,13]`, `[2,16,35]`, and `[1,32,75]` shapes and verifies whole-block versus staged-native bitwise equality.
- Six independent real-checkpoint PyTorch PCM parity scenarios, covering aidar/kseniya/xenia/baya/eugene and 8/24/48 kHz, passed the maximum absolute PCM threshold `0.003`. The whole native pipeline produced **exactly the same PCM bytes** as the separate native fused-depthwise path on all six cases. This was repeated with `SILERO_FLEX_GEMM_DIRECT_WEIGHTS=1`: again six PyTorch PASS and six pairwise bitwise matches.
- The maximum observed PyTorch PCM error was `0.000313759` (eugene, 48 kHz); other cases ranged from `0.000029996` to `0.000079319` in the long samples.

## A/B — complete synthesis

Test phrase: `Привет, мир!` repeated six times, `aidar`, 8 kHz, 43,800 output samples (5.475 s). Twelve alternating persistent-worker release sessions (six per mode), seven calls per session with the first call discarded. Measurements include Base64 encoding and exclude model load. The **control** enables `SILERO_FLEX_FUSED_DEPTHWISE_NORM=1` and uses the previous separate-native path, while the candidate additionally enables `SILERO_FLEX_WHOLE_CONVNEXT=1`.

| Candidate | Control median of session medians | Candidate median of session medians | Observed latency difference |
|---|---:|---:|---:|
| Whole ConvNeXt host pipeline | 149.0 ms | 145.0 ms | -4.0 ms (~2.7%) |

This is a small local effect against considerable VM timing variation, **not a proven general whole-TTS speedup**. The architecture removes Tensor conversions even if the shared CPU does not provide a stable corresponding latency win.

### Rejected: fused final transpose/residual write

An additional trial attempted to combine the second GEMM epilogue and time-major→channel-major transpose into one strided pass. This preserved all 48 library tests and six exact PCM comparisons, but **regressed** in another 12-series A/B: control median **150.5 ms**, candidate median **161.75 ms**. Its source was reverted and is not published. The accepted host pipeline retains the existing two-pass epilogue followed by a layout conversion.

## Reproduction

```bash
cargo test --offline --release --no-default-features --features reference-flex --lib
cargo build --offline --release --no-default-features --features reference-flex --bin silero-tts
```

Use `tools/benchmark_release_flex.py` with `--binary`, `--model-dir`, `--threads 4`, `--text`, `--repeats 7`, and optionally `--reference-pt`/`--max-pcm-error 0.003`. Compare two worker processes: control `SILERO_FLEX_FUSED_DEPTHWISE_NORM=1` and candidate with both `SILERO_FLEX_FUSED_DEPTHWISE_NORM=1 SILERO_FLEX_WHOLE_CONVNEXT=1`. Include `SILERO_VOCODER_TIMING=1` when investigating stage timings.

Actual reproducibility scripts and captured JSON/PCM results are in the evidence ZIP (without model weights, binaries, generated audio or vendor checksum changes).

## Next R&D

The resulting native chain still allocates fresh normalized, hidden and output buffers for each block. Next work should investigate stable reusable scratch buffers and GEMM output layout without strided writes before declaring further improvements. GPU/CubeCL correctness and speed remain separate backend verification tasks.
