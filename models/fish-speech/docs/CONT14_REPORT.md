# Fish Speech 1.5 Rust CONT14 — Source-first functional continuation

Date: 2026-11-18. Input baseline: CONT13 source-only ZIP; Python reference: provided modified Fish Speech 1.5 pipeline. No GitHub clone, network download or remote mutation was performed.

## Implemented

- Native `ReferenceAudioLoader::load_bytes` and `load_path` in `src/reference_loader.rs` via vendored Symphonia, explicit errors, file-size and decoded length limits, multi-channel average→mono. `ReferencePcm::prepare_mel()` connects to the existing Firefly LogMel CPU input at 44,100 Hz. WAV/FLAC/OGG confirmed by runtime smoke.
- Replaced approximate FFT resampling with a source-faithful implementation of torchaudio's default sinc-Hann resampling (lowpass width 6, rolloff .99, polyphase, pad and output length rules). Independent torchaudio `torch.float32` golden fixtures for 16k and 48k→44.1k passed at abs diff <2e-5 on recorded endpoints and middle points; this does not prove parity at every sample of arbitrary audio.
- `ReferenceCache<T>` supports audio SHA-256 cache, ID cache under a reference root, `.lab` captions, per-request text and cache on/off/clear. Includes path traversal and symlink protections. Corrects Python's `load_by_hash` behavior that can return stale captions when using cached tokens.
- `model_config.rs` maps Fish Python BaseModelArgs/DualAR/Naive JSON defaults and explicit overrides to typed `FishModelConfig`. Rejects unsupported head geometry rather than generating wrong network dimensions.
- `synthesize_with_events()` adds immediate header/segment/final/error callback events with unit tests. Legacy `synthesize` behavior remains available.
- New `fish-reference-inspect` CLI to probe external audio and optional LogMel contract.

## Validation

- `cargo test --locked --offline --manifest-path core/Cargo.toml`: 171 unit + 27 integration = 198 PASS.
- `cargo test --locked --offline`: 15 PASS, no failures.
- Python exporter unit tests: 5 PASS.
- `cargo fmt --all -- --check`: PASS.
- `cargo check --locked --offline --features tensorrt-api`: PASS (compiler log ended with `Finished dev profile`; no GPU/SDK execution).
- External local transient FLAC and OGG decode+resample+mel tests: PASS. WAV stereo and 16k resampling: PASS.
- MP3 transient smoke: BLOCKED — provided offline Symphonia 0.6 vendor lacks `symphonia-bundle-mp3`. Feature activation fails Cargo dependency resolution. MP3 is *not* implemented; no FFmpeg subprocess fallback inserted into deployed runtime.

## Explicit limitations

1. **Not a runnable weighted inference engine.** Slow/Fast AR and Firefly still lack runtime engines/weight-bound execution. Existing samples are CPU math and trait-backed test fixtures.
2. **Codec gap.** WAV/FLAC/OGG are functional. MP3/AAC/M4A/WMA not supported with supplied offline codec set.
3. **No authentic model checkpoint / tokenizer** supplied; cannot demonstrate real tokenizer IDs, model logits, voice identity or genuine speech synthesis.
4. **GPU tests separate.** Windows/TensorRT builder, CUDA async KV update, engine cache, real inference and TorchInductor benchmarks are not tested. Linux `tensorrt-api` compilation only proves FFI type compatibility.
5. **No distributed binary.** Rust source-only is the deliverable; `.cargo/config.toml` points at the session-local vendor path and must be updated when extracting elsewhere.

See `CONT14_FUNCTIONALITY_MATRIX.md`: functionality ~70% by equal-weight checklist, operability 6/11 gates; no authenticated GPU/e2e proof.
