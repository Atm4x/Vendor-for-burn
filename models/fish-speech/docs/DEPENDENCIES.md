# Verified dependencies and blockers

## Local environment

- Rust compiler: `rustc 1.100.0 (0ef1bc386 2026-11-09)` and `cargo 1.100.0 (3acc79c0c 2026-11-05)` from provided tar archive. Toolchain installed only within isolated work area, not system-global.
- `clang` and `libclang.so`: 17.0.0 at `/usr/local/swift/usr/lib/libclang.so` (below 22), GCC 14.2, CMake 3.31.6.
- Core crate has **no external dependencies**: `cargo test --offline` succeeds with the provided Rust toolchain.
- `trtx` archive has `Cargo.lock`, all TensorRT-RTX 1.7 headers, `trtx` and `trtx-sys` crate sources, feature `mock` and `v_1_7`.
- It does NOT contain registry source/cache needed by Cargo. `cargo check --offline` on the isolated adapter fails at unresolved registry package `autocxx` (version 0.30.0). This does not exercise or invalidate TRT mock functionality.

## Versions in provided trtx-rs Cargo.lock (selected)

| Crate | Locked version |
|---|---|
| trtx/trtx-sys | 0.9.2+rtx1.7 (workspace) |
| autocxx | 0.30.0 |
| autocxx-build | 0.30.0 |
| bindgen | 0.72.1 |
| cxx | 1.0.205 |
| cc | 1.11.1 |
| bitflags | 2.13.2 |
| libloading | 0.9.0 |
| log | 0.4.35 |
| thiserror | 2.0.21 |
| regex | 1.13.1 |

`trtexec-rs` workspace also references a Git `rustnn` dependency. The standalone adapter uses a different, isolated workspace to avoid requiring this unrelated CLI package. Its remaining blocker is the crates.io registry/cache, not Git rustnn.

## Immediate minimal input for next iteration

- Offline Cargo vendor/registry cache **compatible with the supplied trtx Cargo.lock**, at least `trtx` and `trtx-sys` transitive dependency closure; preferable package `cargo vendor` + `.cargo/config.toml`. No CUDA SDK needed for mock integration.
- If real tokenizer work is next: checkpoint `tokenizer.tiktoken` and actual `config.json`. Without these no real BPE/test IDs or correct actual layer shapes can be established.
- If numeric FSQ parity is next: `vector_quantize_pytorch==1.14.24` source and small golden samples.

## Windows GPU gate only (not required yet)

- TensorRT-RTX SDK/runtime 1.7 x64; compatible CUDA runtime/driver; NVIDIA RTX SM>=7.5; MSVC toolchain; RTX GPU host.
- Actual `model.pth`, Firefly checkpoint, manifest/exporter and PyTorch reference baselines for parity/benchmark (Python only offline; not in final deployment).
- Runtime packaging and per-GPU on-device engine build/cache must be validated, not assumed.

## CONT4 additional observation

The local Python developer environment provides CPU PyTorch 2.10.0 and torchaudio 2.10.0, used only for golden fixture generation. The Rust core's `Cargo.toml` still has no external dependencies. `vector_quantize_pytorch` is NOT locally installed; exact 1.14.24 tagged FSQ equations were inspected from the corresponding source on GitHub, and formula-level fixtures were generated using PyTorch. Actual imported v1.14.24 `GroupedResidualFSQ` numerical parity remains untested.
