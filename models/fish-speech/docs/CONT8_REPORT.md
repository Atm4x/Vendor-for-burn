# Fish Speech 1.5 — CONT8 Rust / TensorRT-RTX handoff

**Date:** 2026-11-17. **Scope:** source-only local changes on top of attached CONT7 source snapshot, Rust 1.100.0, offline Cargo vendor extracted from the verified `codex/fish-speech` Git Bridge artifact. No GitHub write, Windows CI, real NVIDIA TensorRT-RTX runtime, checkpoint or GPU inference was used.

## What was implemented

1. `core/src/firefly_trtx_pointwise.rs`: typed NCHW pointwise plans for native TensorRT ops (`SiLU`, `GELU_ERF`, `Tanh`, residual add, parallel mean, learned gamma scale, channel-only LayerNorm epsilon 1e-6). Shape, arity, dtype and checkpoint channel vector validation.
2. `src/trtx_firefly_pointwise.rs`: actual compile-checked `trtx` 1.7 API lowering with direct SafeTensors gamma/beta binding. `kGELU_ERF` matches Python `nn.GELU()` semantics rather than TANH approximation. Both Python `LayerNorm` layouts reduce across the channel axis in NCHW.
3. `core/src/firefly_trtx_mask.rs` and `src/trtx_firefly_mask.rs`: original Python `sequence_mask` profiles. Uniform lengths within batch only; masked elements set to 0; all-valid masks become an identity. Decoder uses `valid_code_frames * 4` for mel and `valid_code_frames * 4 * 512` for PCM. Mask is a static FP32 constant when nontrivial, not a dynamic per-example input.
4. `core/src/firefly_trtx_fsq.rs` and `src/trtx_firefly_fsq.rs`: Fish-specific GroupedResidualFSQ **decoder** as a precomputed affine lookup table, one 1001-row-per-group LUT including the original `-1` sentinel semantics. Eight groups, 1000 legal nonnegative codes each, 64 output features per group, one quantizer. Table is `(8 * 1001 * 64 * 4) = 2,050,048 bytes` FP32. TRT graph uses offset + gather + shuffle, validates all indices lie in `[-1,999]` with a native assertion and binds original checked `project_out` checkpoint arrays. This is structurally compiled API code, **not benchmarked on RTX**.
5. `core/src/firefly_trtx_segment.rs` and `src/trtx_firefly_segment.rs`: strict dependency-closure graph segmentation. A caller supplies explicit external boundaries (e.g. `CodeInput` for decoder or log-mel for encoder); missing unsupported ops fail before modifying the TRT network. Topologically traverses the graph and links conv/pointwise/mask/FSQ-decode TRT layers using checked checkpoint weights. Cannot silently bypass STFT, FSQ-encode or mask semantics.
6. `core/tests/trtx_pointwise_original_pytorch.rs` + generated fixtures: eight PyTorch-equation cases for pointwise ops and both LayerNorm formulations. Rust CPU results within 1e-5. Existing CPU `gelu`/`silu` made public to reuse existing math, not duplicated.
7. `core/src/main.rs --firefly-trt-coverage`: more precise per-family unsupported-operation report.

## Actual structural coverage

| Graph | Conv / Linear | Pointwise / Norm | Mask | FSQ decode | Lowerable op nodes | Remaining graph nodes |
|---|---:|---:|---:|---:|---:|---|
| Reference encoder | 66 | 85 | 2 | 0 | **153/156** | `AudioInput` (external), `LogMel`, `FsqEncode` |
| Waveform decoder | 105 | 155 | 2 | 1 | **263/264** | `CodeInput` (external) |
| **Total** | **171** | **240** | **4** | **1** | **416/420 (99.0%)** | 2 external inputs + 2 internal ops |

The entire decoder graph has a valid **static topological dependency plan** when its `CodeInput` is supplied. This **does not** prove TRT engine creation, that all native API operations are supported together, numerical GPU parity, correct waveform, real-model decoding or performance. The reference encoder is still missing native `LogMel` integration and `FsqEncode` lowering. Whole TTS includes a large Slow/Fast AR Transformer and runtime still incomplete.

## Verified local results

- `cargo fmt --all --check`: PASS.
- `cargo test --locked --offline --manifest-path core/Cargo.toml`: **155 passed, 0 failed** (131 unit + 12 Firefly reference + 4 FSQ encode + 2 FSQ decode + 4 spectrogram + 1 convolution parity + 1 pointwise parity).
- `cargo test --locked --offline`: 3 passed, 0 failed (outer crate).
- `cargo test --locked --offline --features tensorrt-mock`: 7 passed, 0 failed (3 outer + 3 blueprint + 1 mock network).
- `cargo check --locked --offline --features tensorrt-api`: PASS, trtx 0.9.1+rtx1.7 Rust/C++ FFI typechecking, **no SDK-backed build or execution**.
- `python -m unittest discover -s tools -p 'test_*.py'`: 5 passed, 0 failed.
- FSQ table: all 1000 nonnegative code values in all 8 groups, plus `-1` sentinel, exactly match the existing Rust reference with synthetic FP32 projection weights. This is **CPU arithmetic** parity; real checkpoint/Torch GPU parity remains blocked.
- `core/tests/trtx_pointwise_original_pytorch.rs`: 8 original-PyTorch-equation cases within `1e-5`, CPU only.

## Global engineering readiness (subjective approximation, not GPU benchmark)

**~39% complete / ~61% remaining** of full Windows native Fish Speech 1.5 TTS.

- Core/tokenizer/prompt/sampling CPU contracts: ~68%, still lacks real tokenizer and end-to-end original-model golden checks.
- SafeTensors/exporter: ~70%, lacks actual pretrained semantic and Firefly checkpoint verification.
- Slow/Fast AR inference/backend: ~25%, substantial graph lowering, cache, sampling and runtime incomplete.
- Firefly model graphs/ops: ~70%, decoder static lowering connected, reference LogMel/FSQ encode missing, no actual GPU validation.
- TensorRT-RTX backend: ~30%, native API FFI compiled, but builder→serialized engine→GPU launch not tested.
- End-to-end TTS orchestration: ~10%, no audio generated.
- Windows GPU numerical parity/performance: 0% verified.
- Windows distribution/package: 0% verified.

## Hard blockers / risks to resolve next

1. Reference encoder LogMel and grouped FSQ encode; preserve PyTorch STFT window/reflect padding and two-stage residual scalar `bound()` semantics. Source-of-truth is user-supplied Fish 1.5, not upstream S2.
2. Actual pretrained `config.json`, `tokenizer.tiktoken`, semantic and Firefly checkpoints (proper license) and exact-weight CPU/PyTorch goldens. DO NOT assume production weights were tested.
3. Slow prefilling/decoding and Fast AR Transformer network lowering, active-prefix KV, sampling, GPU EOS without per-token CPU sync, lifecycle/cancellation.
4. Real TensorRT-RTX 1.7 SDK + Windows NVIDIA RTX GPU gate: builder, conv/deconv, normalization, gather/shuffle/assertion acceptance, serialized engine and output tensors. `add_assertion` overhead, LUT versus project_out GEMM latency, static batch masks and dynamic shape profiles need profiling.
5. Compare native warmed latency, peak VRAM and RTF against **the exact Python Fish 1.5** warmed `torch.compile(mode="reduce-overhead")`; Nsight Systems/Compute; DLL dependency packaging and engine cache invalidation.

### Critical caveats

- LUT precomputation may change FP16/BF16 rounding and numeric order compared with GPU `project_out`; first compare FP32 weighted output, then optimize.
- If code indices are outside `[-1,999]`, native assertion is intended to stop execution rather than return a different group's row. GPU support of this assertion is *NOT_TESTED_GPU*.
- Static masks apply one valid length to the entire batch; per-row variable-length masking needs runtime bindings or length-aware kernels.
- `trtx-rs` is experimental. API compilation is not proof of runtime support. No mock fallback is production inference.
- Original `CONT7` source snapshot was the actual local source baseline; Git Bridge commit `e89871a0` is older than the previously recorded remote HEAD `218f920a`. This CONT8 does not claim to include GitHub changes after CONT7.

## Reproduce

From `models/fish-speech` with Rust 1.100.0, offline vendor configured and `LIBCLANG_PATH` for trtx:

```bash
cargo fmt --all --check
cargo test --locked --offline --manifest-path core/Cargo.toml
cargo test --locked --offline
cargo test --locked --offline --features tensorrt-mock
cargo check --locked --offline --features tensorrt-api
python -m unittest discover -s tools -p 'test_*.py' -v
cargo run --locked --offline --manifest-path core/Cargo.toml --bin fish-proto -- --firefly-trt-coverage
```

Do not run `dotnet` for a Unity project. Do not edit old unused scripts.
