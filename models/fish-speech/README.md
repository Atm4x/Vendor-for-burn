# Fish Speech offline dependencies

Run these commands from the repository root on Linux. Cargo uses the shared `vendor/` directory via `.cargo/config.toml` and resolves the exact versions in `models/fish-speech/Cargo.lock`.

```sh
cargo check --manifest-path models/fish-speech/Cargo.toml --locked --offline
cargo test --manifest-path models/fish-speech/Cargo.toml --locked --offline --features tensorrt-mock
```

The first command checks the CPU dependencies. The second also links the TensorRT-RTX 1.7 bindings and creates a mock builder and network. It requires a C++ compiler, Clang and its resource headers, and libclang below version 22. The mock path runs without an NVIDIA GPU, CUDA Toolkit, or TensorRT runtime and does not perform inference. The commands were verified on Linux x86_64 with Rust 1.98.1, GCC 13 and libclang 18.

The CPU dependencies cover serialization, hashing, SafeTensors and FP16/BF16, audio decoding and resampling, FFT, and ChaCha RNG. `tiktoken-rs` is available for checkpoints with the tiktoken vocabulary format, such as [Fish Speech v1.5.0](https://github.com/fishaudio/fish-speech/blob/v1.5.0/fish_speech/tokenizer.py). Use the tokenizer assets and special token IDs from the target checkpoint and verify token ID parity against its Python reference. The target checkpoint and native system toolchain are supplied separately.

## Prototype core

`core/` contains dependency-free Rust model contracts and CPU reference math.
Tests: `cargo test --manifest-path models/fish-speech/core/Cargo.toml --offline`.
Mock probes: `cargo test --manifest-path models/fish-speech/Cargo.toml --offline --features tensorrt-mock`.

Numerical parity is currently limited to synthetic/fixture inputs. No production TensorRT engine, weights or full speech inference is implemented. See `docs/CONT4_REPORT.md`.

## Features and verification boundaries

- `tensorrt-mock`: Compiles `trtx` bindings and tests the typed `TrtBlueprint` preflight. The upstream mock builder returns a null `INetworkDefinition` pointer: do not access `network.nb_inputs`, add layers or claim GPU execution.
- `tensorrt-api`: Compile-checks the Rust `NetworkDefinition` translator without requiring a local TensorRT-RTX SDK. Execution requires the real SDK and is not tested here.
- `core/`: 132 CPU reference tests for synthetic transformer/Firefly graph math and original Python numerical layer fixtures (no pretrained model weights).

The translator currently supports typed input bindings, identity, sigmoid, elementwise sum/product and matrix multiplication with a transposed RHS; `LastToken`, normalization, RoPE, attention, dynamic KV and other unsupported model operations are deliberately rejected before mutating a network.

`ConstantWeight` is presently modeled as an unbound runtime input in the API prototype; production static weight packing and engine compilation are NOT implemented.

## CONT6 update (2026-11-17)

The previous test figures describe the CONT4/CONT5 baseline. Current CONT6 includes a Rust SafeTensors weight loader/manifest, offline checkpoint exporter, CPU grouped residual FSQ encode, typed Firefly source-matched DAG and shape audit. Core now has **139 passing tests**; the Firefly architecture key/shape audit matches **450/450** parameter entries. See `docs/CONT6_REPORT.md` for exact evidence, offline commands and unresolved GPU gates. The architecture and CUDA bindings still cannot execute a complete Fish Speech model without the original checkpoint, NVIDIA RTX hardware and TensorRT-RTX SDK.

## CONT7 update (2026-11-17)

See [`docs/CONT7_REPORT.md`](docs/CONT7_REPORT.md) for the current test ledger. This cycle adds TensorRT-RTX NCHW Conv/Deconv/1x1 Linear lowering with verified Fish padding/cropping geometry, SafeTensors FP32 weight binding, and source-PyTorch convolution fixtures. **143 Core tests** and `cargo check --features tensorrt-api` pass on Rust 1.100.0 offline. Static Firefly Conv/Linear coverage: **171/420 DAG nodes**, which is not an executable full graph or a GPU benchmark. The old 132/139 Core-test numbers above describe previous CONTs, not the latest run. Real RTX SDK execution, original checkpoints, complete Slow/Fast AR and full TTS remain outstanding.

## Functional parity and operability progress

See `docs/CONT13_FUNCTIONALITY_MATRIX.md` for source-derived behavior inventory; Windows/CUDA verification is tracked independently. `core::synthesis::synthesize()` composes backend interfaces but does not itself perform weighted model inference.

## CONT14: reference audio, checkpoint configs and streaming events

- `src/reference_loader.rs` decodes supported reference-audio bytes or files with Symphonia,
  mixes all channels to mono, and performs a torchaudio-compatible `sinc_interp_hann`
  resample to 44,100 Hz. `ReferencePcm::prepare_mel()` passes this directly to
  the existing Firefly LogMel CPU preprocessor.
- `src/reference_cache.rs` caches reference acoustic tokens by SHA-256(audio) and
  by reference ID, including `.lab` captions and `CachePolicy::On/Off`.
  For safety, reference IDs are restricted to one path component.
  Unlike an apparent bug in the Python `load_by_hash`, cached audio tokens do not
  overwrite the current request's text.
- `src/model_config.rs` parses the original Fish Speech 1.5 `config.json`
  defaults and DualAR/Naive model types into `FishModelConfig`.
- `core/src/synthesis.rs::synthesize_with_events()` emits header, segment,
  final and error notifications through a callback; the existing `synthesize()`
  entrypoint remains compatible.
- `fish-reference-inspect <audio-path> [--mel]` is a local CPU smoke harness.

The provided offline vendor contains Symphonia WAV/FLAC/OGG codecs, **not** the
`symphonia-bundle-mp3` / AAC / MP4 or WMA dependencies. MP3/AAC/M4A/WMA are
**not supported** by the current native decoder; do not claim complete format parity.
The fully native weighted Fish Speech inference and GPU execution also remain
unimplemented/unverified. See `docs/CONT14_FUNCTIONALITY_MATRIX.md` and
`docs/CONT14_REPORT.md` for the independent functionality and operability ledgers.

## CONT15: executable CPU semantic generation (2026-11-18)

`core::cpu_weighted_ar::CpuWeightedDualAr` and `core::cpu_weighted_naive::CpuWeightedNaive` execute Slow/Fast or Naive transformer IR on named F32 tensors, maintaining logical KV prefix and matching source-specific sampling order. `src::weighted_cpu::CpuSemanticGenerator::from_checkpoint` binds F32/F16/BF16 SafeTensors model weights validated by the manifest to the generator interface. These are **CPU reference implementations** and are not an optimized runtime, authentic full-model parity, GPU inference or real speech synthesis. The Firefly weighted acoustic decoder is not implemented yet.

See `docs/CONT15_REPORT.md` and `docs/CONT15_FUNCTIONALITY_MATRIX.md` for exact functional-vs-operability progress and tests. The `tensorrt-mock` build is not confirmed this cycle; `tensorrt-api` compilation is confirmed, without actual RTX execution.

## CONT16: weighted Firefly CPU execution (2026-11-18)

`core::firefly_weighted::WeightedFireflyExecutor` executes both complete Firefly DAGs in dependency order with verified named checkpoint weights. It supports native LogMel, grouped/depthwise and transposed convolutions, ConvNeXt pointwise/norm/residual, grouped residual FSQ encode/decode, HiFiGAN operations and source-specific masks. Dead intermediate tensors are released after the last dependent node; convolution weights are borrowed rather than copied during every invocation.

`src::weighted_firefly::{CpuFireflyReferenceEncoder,CpuFireflyDecoder}` bridge hashed F32/F16/BF16 SafeTensors into the CPU executor and expose `AcousticDecoder`, `encode_mono`, `encode_reference_bytes` and cache-backed reference encoding. `CpuFireflyDecoder` reuses shared immutable weights across sequence-length profiles. The existing `synthesize` API can now invoke the weighted acoustic decoder, while the semantic backend can be the `CpuSemanticGenerator` from CONT15.

Full 156-node encoder and 264-node decoder synthetic CPU tests passed, including a nonzero-weight waveform test. Run the expensive marked tests explicitly with:

```sh
cargo test --manifest-path models/fish-speech/core/Cargo.toml --locked --offline --release full_synthetic_graph_tests -- --ignored
cargo test --manifest-path models/fish-speech/core/Cargo.toml --locked --offline --release full_nonzero_graph_test -- --ignored
cargo test --manifest-path models/fish-speech/Cargo.toml --locked --offline --lib weighted_synthesis_smoke -- --ignored
```

This is a reference implementation, **not** authentic-checkpoint numerical parity, a production-performance CPU backend, a real pretrained audio demonstration, or CUDA/TensorRT engine execution. Real model weights and the matching tokenizer remain necessary for parity and audible TTS checks. Windows/CUDA checks are tracked separately from functional source-port coverage.

## Unified CPU model lifecycle (CONT17)

`model_runtime::ModelPaths` binds the semantic checkpoint directory, Firefly checkpoint
and the exact Firefly source configuration supplied to the offline exporter.
Expected semantic files are `config.json`, `tokenizer.tiktoken`, `manifest.json`
and SafeTensors shards; the Firefly directory contains `manifest.json` and its
shards. `firefly_config_source` is an explicit path because the original Fish 1.5
Firefly configuration may be a Python source file rather than `config.json`.
The exact SHA-256 of each supplied config is checked against its export manifest;
no Python is evaluated at runtime.

`LoadedFishArtifacts::open(&paths)` performs config, BPE, manifest and shard
verification without constructing inference executors. `into_cpu_model(seed, mode)`
assembles `CpuFishModel` with CPU Semantic (Dual/Naive), Firefly encoder/decoder
and reference cache. It can then `prepare_references(...)`, `synthesize(...)`,
`clear_reference_cache()`, `reload(...)` and `close()`. The mutable model is
single-request-owned by default. `reload` creates the replacement before swapping
so invalid new checkpoint files do not destroy the current model; `close`
releases all owned model state. This is the native CPU reference path, not a
TensorRT-RTX engine or a substitute for real trained checkpoint parity.

`SemanticGenerator::generate_with_cancel` preserves existing `generate` trait
implementations as a default bridge, while the native weighted Dual/Naive CPU
implementations poll between every semantic token and acoustic codebook.
Existing streaming and request APIs are unchanged. A mid-generation cancellation
returns an error rather than partially decoded audio.

The full two-checkpoint integration smoke is intentionally ignored by default:
`cargo test --locked --offline --lib full_synthetic_bundle_can_be_assembled_and_closed -- --ignored`.
It creates complete synthetic SafeTensors, loads both models from the manifests,
invokes `synthesize()` and validates finite 44,100 Hz PCM. This test verifies
plumbing and shape compatibility, **not** voice quality, performance or GPU parity.
