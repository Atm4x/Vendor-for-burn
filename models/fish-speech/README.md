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
