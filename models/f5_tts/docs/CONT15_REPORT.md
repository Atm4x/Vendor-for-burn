# F5-TTS Rust CONT15 — TensorRT-RTX Vocos and Cross-Lingual SpeedTransformer

Date: 2026-11-20. Starting point: published GitHub CONT14 commit `e6819bea92b8ec35eed3e413c54446ecaed12de5`.

## Scope

Only `models/f5_tts` changed. No ONNX, Python runtime dependency, MaryML reuse, CubeCL fallback, Fish Speech/RVC/Legacy edits. TensorRT-RTX code uses the already pinned `trtx` crate; the independent CubeCL path remains unchanged.

## Functional implementation

1. `src/backend/trtx_vocos.rs` adds a fixed-shape TRT Vocos spectral decoder. It loads the exact `backbone.embed`, normalization, 8 ConvNeXt blocks, per-block GRN and final `head.out` weights directly from safetensors; the shape can also be set to a small two-block configuration for parity fixtures. Graph operations: model-specific group Conv1d, LayerNorm, GELU-erf, GRN, residual connections, spectral head.
2. `build_serialized_vocos` builds a TensorRT-RTX network accepting mel `[1,T,100]` and emitting `spectral_head` `[1,T,1026]`. `enqueue_vocos` binds explicit device pointers and stream, validating sizes and non-overlap. Its caller owns allocations, stream and warmed TRT context.
3. `decode_vocos_spectrum` implements the non-CubeCL post-head pipeline: log-amplitude exponential, phase sin/cos, independent native Rust IStft, mono 24 kHz PCM float waveform. This requires one GPU-to-host transfer of the head; an all-GPU cuFFT ISTFT is a possible optimization later. No implicit Burn/CubeCL step.
4. `src/backend/trtx_speed.rs` adds the full Cross-Lingual SpeedTransformer graph, directly from safetensors: mel projection, two grouped Conv1d/Mish layers, RoPE masked attention, layer normalization, tanh-GELU feed-forward, masked temporal softmax pooling, classifier normalization and 32-class head. Input `mel` `[B,T,100]` and `valid_frames` `[B,T]` are fixed-shape. Invalid key positions use a -FLT_MAX penalty, and invalid query outputs are masked as in Python. A 0/1 mask is generated from sequence lengths by `TrtSpeedInputs::from_lengths`.
5. `decode_predicted_speeds` applies the original first-index tie-breaking argmax and 0.25-step speed mapping. `enqueue_speed` binds validated device buffers and external stream.
6. Shared TRT helper functions for group Conv1d, LayerNorm epsilon, Mish and RoPE are reused instead of duplicating model math. Public model-specific interfaces remain independent.

## Verification in this cycle

- `cargo check --offline --locked` without features: PASS.
- `cargo check --offline --locked --features tensorrt-api`: PASS.
- `cargo check --offline --locked --features tensorrt-mock`: PASS.
- `cargo test --offline --locked --lib -- --test-threads=1`: 51/51 PASS.
- `cargo test --offline --locked --features tensorrt-mock --lib -- --test-threads=1`: **63/63 PASS, 2 ignored**. Six new tests cover Vocos weight shapes, independent PyTorch spectral-head-to-waveform numeric parity (threshold 2e-4), Vocos buffer overflow/aliasing, Cross-Lingual SpeedTransformer fixture weights, valid/padded masks, 32-class output decoding, invalid shapes/device buffers.
- Source files formatted by `cargo fmt --all`.
- No TRT engine creation was executed: the bundled `trtx` mock does not construct a real `INetworkDefinition`. The graph math is implemented but its successful TensorRT-RTX build **and its TRT GPU numerical equivalence are not proven**. The existing PyTorch comparison tests prove native postprocessing and host contracts; CubeCL parity from CONT11 is independent.
- CubeCL GPU hardware/Windows and pretrained inference: not run, intentionally secondary.

## Remaining functional work

1. Concrete warmed TRT runtime owner (device memory, contexts, stream, multistep CUDA-resident CFM loop, two ping-pong state buffers, input staging, Vocos staging, SpeedTransformer inference and lifecycle). Current TRT graph builders and enqueue functions are primitives; no self-contained TRT inference session exists yet.
2. TensorRT engine build and TRT/PyTorch numerical parity on actual NVIDIA runtime, which will expose any unsupported operator/shape or subtle FP math deviations. Avoid counting mock compilation as TRT inference.
3. Optional GPU ISTFT with cuFFT, avoiding the current spectrogram transfer and native CPU ISTFT. Not necessary for the correctness-first stage, but required for fully device-resident audio decoding.
4. Wider natural-language / pinyin edge cases.

## Progress

- **Functionality 92%**, previous 89%; engineering estimate for CubeCL plus specialized TensorRT-RTX model-specific feature coverage, including pending TRT session, not proof of end-to-end production readiness.
- **Workability 43% (3/7)**, unchanged; GPU runtime, pretrained checkpoints and Windows not verified.

## Reproduction

Use Rust 1.100.0 and the archived CONT12 union vendor at `/mnt/data/f5-cont12/vendor-union` (or mount the same vendor tree in any local path). The repo's locked dependencies are unchanged.

`cargo --config 'source.crates-io.replace-with="vendored-sources"' --config 'source.vendored-sources.directory="/path/to/vendor-union"' test --offline --locked --features tensorrt-mock --lib -- --test-threads=1`
