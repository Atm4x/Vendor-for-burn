# F5-TTS native CONT01 — local evidence

Date: 2026-11-20. Rust 1.100.0. Baseline Git Bridge commit: `12ef7d2dbe8b62d3a3a530bf9f4ed9611e45fc00`.

## Completed

- New isolated `models/f5_tts` crate. No files outside this folder changed.
- F5TTS_v1_Base fixed model configuration (24 kHz / 100 mel channels / 22 DiT blocks).
- Pure-Rust CFM Euler update, CFG mixing, all EPSS tables and sway sampling.
- Reference-frame conditioning prefix and duration bounds.
- Original character-vocabulary ids, OOV = 0 and batch padding = -1.
- Cross-Lingual 32-class SpeedMapper, punctuation and CJK unit accounting, sentence chunking, duration heuristics; English syllable counter remains an injected contract rather than an incorrect `pyphen` substitute.
- Native `safetensors` header inspection and lazy tensor-byte reads, including `ema_model.` normalization.
- Separate WGPU/CUDA compile-time feature flags, model-specific on-device CFM integration loop and request text-cache release, with no per-step GPU-to-CPU transfer in the integration code.

## Verification

| Gate | Result |
|---|---|
| Source bundle SHA-256 and commit match | PASS |
| Local Git integrity | PASS |
| `cargo test --offline --no-default-features` | PASS — 16 unit tests |
| `cargo check --offline --features cubecl-wgpu` | PASS |
| `cargo check --offline --features cubecl-cuda` | BLOCKED: vendored `tracel-llvm-bundler` build script attempts to download LLVM; this isolated Linux environment has no DNS access |
| CPU CubeCL compilation | BLOCKED by the same LLVM download |
| Runtime WGPU/CUDA inference with actual F5-TTS weights | NOT PERFORMED |
| TensorRT-RTX runtime | NOT IMPLEMENTED |
| DiT/SpeedPredictor/Vocos full functional parity | NOT IMPLEMENTED |

A successful WGPU compile checks type correctness only. It is not an F5-TTS GPU speech generation result. CUDA failure was in a vendor LLVM dependency during its build script, not in the new F5 source.

## Next implementation milestone

Implement an actual DiT forward pass with checkpoint loading and tests against real F5-TTS v1 tensors. Then add the audio frontend, Vocos, and Cross-Lingual SpeedTransformer. Only after the CubeCL parity gates should the separate native TensorRT-RTX backend be introduced.
