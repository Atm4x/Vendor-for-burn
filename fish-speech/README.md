# Fish Speech offline dependencies

Run these commands from the repository root on Linux. Cargo uses the shared `vendor/` directory via `.cargo/config.toml` and resolves the exact versions in `fish-speech/Cargo.lock`.

```sh
cargo check --manifest-path fish-speech/Cargo.toml --locked --offline
cargo test --manifest-path fish-speech/Cargo.toml --locked --offline --features tensorrt-mock
```

The first command checks the CPU dependencies. The second also links the TensorRT-RTX 1.7 bindings and creates a mock builder and network. It requires a C++ compiler, Clang and its resource headers, and libclang below version 22. The mock path runs without an NVIDIA GPU, CUDA Toolkit, or TensorRT runtime and does not perform inference. The commands were verified on Linux x86_64 with Rust 1.98.1, GCC 13 and libclang 18.

The CPU dependencies cover serialization, hashing, SafeTensors and FP16/BF16, audio decoding and resampling, FFT, and ChaCha RNG. `tiktoken-rs` is available for checkpoints with the tiktoken vocabulary format, such as [Fish Speech v1.5.0](https://github.com/fishaudio/fish-speech/blob/v1.5.0/fish_speech/tokenizer.py). Use the tokenizer assets and special token IDs from the target checkpoint and verify token ID parity against its Python reference. The target checkpoint and native system toolchain are supplied separately.
