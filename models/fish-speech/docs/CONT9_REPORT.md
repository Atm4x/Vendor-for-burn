# Fish Speech 1.5 — CONT9 report

Date: 2026-11-17. Basis: user-supplied CONT8 source-only ZIP, original `Fish-speech-pipeline-main(1).zip`, source-based TensorRT-RTX 1.7 bindings in the previously downloaded Git Bridge artifact, Rust 1.100.0. All development was performed locally. No GitHub writes, CUDA SDK, GPU, original model checkpoints or synthetic speech were used.

## Implemented

1. `core/src/firefly_trtx_fsq.rs`: exact Fish 1.5 `FsqEncodePlan` for 8 groups, 1 residual quantizer, levels `[8,5,5,5]`. It validates checkpoint keys/shapes and packs eight `project_in` matrices/bias vectors into one grouped 1×1 convolution (32 channels). Generates FP32 constants for two sequential `bound()` operations and mixed-radix bases `[1,8,40,200]`.
2. `src/trtx_firefly_fsq.rs`: native TensorRT-RTX API lowering for reference FSQ encode: grouped convolution → double `tanh` bound → round-to-nearest-even → mixed-radix weighting → reshape/reduce → INT32 acoustic codes. Strict preflight refuses incorrect shape or checkpoint dtype. This only type-checks on Linux; real GPU numerical parity is **NOT_TESTED_GPU**.
3. `core/src/firefly_trtx_segment.rs`, `src/trtx_firefly_segment.rs`: `FsqEncode` participates in dependency closure, strict parameter preflight and actual native FFI lowering. Complete encoder from external log-mel boundary to acoustic codes is structurally connected.
4. `core/src/reference_audio.rs`: CPU preprocessing from mono FP32 PCM at exactly 44.1 kHz into batched NCT log-mel tensors. Uses the existing source-verified `FishLogMel`, derives frame profiles from source, checks graph boundary and rejects unequal batch lengths/sample rates. This is not an audio decoder/resampler nor a GPU STFT.
5. `src/trtx_firefly_network.rs`: a single typed FFI entry point to define full Firefly encoder/decoder TRT networks from their external boundaries, validate and mark outputs. Does **not** invoke TensorRT builder/runtime.
6. `core/src/trtx_blueprint.rs`, `src/trtx_lowering.rs`: Slow/Fast AR `RmsNorm` and static `LastToken` Slice have native blueprint lowering, removing two deferred operator families. RMSNorm is FP32 reduce(mean of squares)+epsilon+sqrt+divide+gamma; CPU result checked against an independent PyTorch reference fixture. Other Transformer stages remain deferred.
7. `core/src/main.rs`: CLI structural coverage report updated to include `FsqEncode`; previously hard-coded CONT8 check would panic at 154/156.
8. Mock/preflight assertions adjusted for the now-supported `RmsNorm` and `LastToken` operations. New Core integration tests verify FSQ packing, double-bound, F32 finite projection validation, full encoder closure and strict batch input semantics.

## Firefly structural coverage

| Graph | Lowerable internal ops | External or host-side boundaries | Counts |
|---|---|---|---|
| Reference encoder | Conv/Linear 66, pointwise 85, mask 2, FSQ encode 1 | AudioInput; LogMel (CPU host) | **154/156** |
| Waveform decoder | Conv/Linear 105, pointwise 155, mask 2, FSQ decode 1 | CodeInput (external) | **263/264** |
| Combined | 417 | 3 boundary nodes | **417/420 = 99.3%** |

These counts are only for graph-node *structural lowering*. They are not the amount of Fish Speech completed and are not a real TensorRT engine/GPU parity result. Multiple TensorRT layers may represent one Firefly node.

## Verified gates

- `cargo fmt --all --check`: PASS.
- `cargo test --locked --offline --manifest-path core/Cargo.toml`: **161 PASS** (137 Core unit tests plus 24 integration tests, doc tests 0).
- `cargo test --locked --offline`: **3 PASS** main crate.
- `cargo test --locked --offline --features tensorrt-mock`: **7 PASS** (3 main + 3 blueprint mock + 1 mock network).
- `cargo check --locked --offline --features tensorrt-api`: **PASS** on Linux with `LIBCLANG_PATH=/usr/local/swift/usr/lib`. This checks FFI compilation only; there is no TensorRT SDK/GPU launch.
- `python -m unittest discover -s tools -p 'test_*.py'`: **5 PASS**.
- `cargo run --locked --offline --manifest-path core/Cargo.toml --bin fish-proto -- --firefly-trt-coverage`: **PASS**, prints 154/156 reference, 263/264 decoder and all remaining boundary nodes.
- Independent PyTorch CPU RMSNorm fixture: PASS (max allowed absolute difference 3e-6). FSQ encode independent golden in pre-existing tests remains passing; new TRT graph numeric validation is still blocked on GPU.
- Warnings about six not-yet-consumed dependencies in outer Cargo.toml are pre-existing; no dependency versions were changed.

## Global completion estimate

**~42% full native Fish Speech TTS ready, ~58% remaining**, qualitative engineering weighting, not an automatic statistic.

| Domain | Engineering completion | Remaining hard work |
|---|---:|---|
| Core, tokenizer, prompts, CPU contracts | ~69% | Golden tokenizer parity on true checkpoint, long-text prompts |
| SafeTensors/checkpoint export | ~70% | Real pretrained weights and exact tensor check |
| Slow/Fast AR | ~28% | Embeddings, split QKV, RoPE, active-prefix KV and GQA execution, sampling |
| Firefly graph/host preprocessing | ~75% | Real TRT engine build, GPU quantizer parity, host WAV/mono/resample path |
| TensorRT-RTX backend | ~33% | GPU builder/runtime/buffers/stream, engine cache, unsupported ops on actual SDK |
| End-to-end native TTS | ~10% | Model lifecycle, full synthesis/reference voice cloning |
| Windows GPU parity/performance | 0% verified | Windows RTX SDK and pretrained model; warmed TorchInductor comparison |
| Distribution/package | 0% verified | Windows dependencies, minimal runtime, license acceptance, cache |

## Required next stages

1. Finish AR embedding and QKV/RoPE/GQA/KV-cache operations against the original 1.5 model, including true active-prefix and separate reset on every Fast step; do not replace with a generic Llama implementation.
2. Implement native TensorRT-RTX build/execute contract (Windows, SDK 1.7, RTX GPU) and verify Firefly encode and decode with pre-trained weights, before claiming real end-to-end inference.
3. Include host audio decode/mono conversion/high-fidelity resampling; reject unsupported formats explicitly; support varying batch lengths with runtime masks rather than static uniform-length masks.
4. Connect prompt tokenizer, Slow/Fast sampling, reference encoder, Firefly decoder and cancellation into a single warmed session.
5. Golden numerical reference from the supplied Fish Python pipeline using exact checkpoint; profile against warmed `torch.compile(..., mode="reduce-overhead")` for matched workloads and GPU precision.
6. Validate Windows packaging size/ABI/runtime/cache/licenses. No mock path as a production fallback.

**Important:** No pretrained `model.pth`, Firefly checkpoint, real model `config.json`, `tokenizer.tiktoken`, actual NVIDIA TensorRT-RTX SDK, Windows runner or GPU are present in this environment. Consequently `GPU_ENGINE_BUILD`, `GPU_NUMERIC_PARITY`, `END_TO_END_TTS`, `GPU_BENCH`, `WINDOWS_PACKAGE` remain `NOT_TESTED_GPU` / `BLOCKED_ON_ARTIFACT`, not PASS.

### Reproduction

Use the user-supplied Rust 1.100.0 and offline Cargo vendor from Git Bridge. `LIBCLANG_PATH` must point to an installed libclang below version 22 (Clang 17 worked here), and Cargo config must point to the local offline vendor.

```
cargo fmt --all --check
cargo test --locked --offline --manifest-path core/Cargo.toml
cargo test --locked --offline
cargo test --locked --offline --features tensorrt-mock
LIBCLANG_PATH=/usr/local/swift/usr/lib cargo check --locked --offline --features tensorrt-api
python -m unittest discover -s tools -p 'test_*.py'
cargo run --locked --offline --manifest-path core/Cargo.toml --bin fish-proto -- --firefly-trt-coverage
```
