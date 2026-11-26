# CONT15 — Fish Speech 1.5 Rust

Baseline: CONT14 source-only ZIP, modified original Python Fish Speech 1.5 pipeline. Keep the two independent ledgers in `docs/CONT14_FUNCTIONALITY_MATRIX.md`.

Functional checklist: 10 DONE, 11 PARTIAL, 1 MISSING (~70% by equal-weight accounting). Do not reduce because no Windows/CUDA verification. Verification: 6/11 PASS; 0 authentic GPU/e2e/Windows gates.

First investigate the remaining codec gap F08. Offline vendor has Symphonia 0.6 WAV/FLAC/OGG but **no** `symphonia-bundle-mp3 0.6.1`; request a source/vendor ZIP of the missing codec plus AAC/MP4 as needed. Do not silently use ffmpeg subprocess as mandatory user runtime. Existing WAV/FLAC/OGG, torchaudio-like sinc resampler, LogMel bridge, cache ID/hash and streaming callback are implemented and tested.

Main next functional priority: full **weighted semantic generator**, not more structural mock ops. Inspect actual `BaseTransformer.forward_generate`, `DualARTransformer.forward_generate` and `decode_one_token_ar` from supplied Python. Connect weight-bound Slow prefill → token logits → sampling → Fast acoustic codebook sequence with reset. Preserve KV capacity vs logical length and GPU-side writes without capacity copying. Then attach real Firefly reference encoder and acoustic decoder to the new `synthesize_with_events` pipeline. Naive path remains F14 MISSING. Authentic checkpoints not present; do not claim actual model parity.

Toolchain Rust 1.100.0; use supplied Git bundle offline vendor and existing `Cargo.lock`. `cargo test --locked --offline` locally. The `tensorrt-api` check compiles, but SDK/GPU remains separately NOT TESTED. No GitHub mutations. Do not edit legacy/obsolete Python files.
