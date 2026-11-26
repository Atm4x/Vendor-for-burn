# Fish Speech 1.5 — CONT22: sequential functional-parity work (F01 → F08 → F11)

## Provenance

- Target: `Atm4x/Vendor-for-burn`, branch `codex/fish-speech`.
- Source: the complete verified CONT21 Git Bridge bundle at `011e87cf535687937c909e3e5cd405f96670d2d4` (`source-head.txt`, SHA-256, and `git bundle verify` all matched).
- Work and verification were performed against the local Git checkout; only the final commit is published with GitHub Connector.

## Changes

### F01 — checkpoint geometry

- Fast AR now derives its default `head_dim` from its own `fast_dim / fast_n_head`, rather than incorrectly borrowing Slow AR's value.
- Explicit zero-valued Fast AR dimensional overrides are rejected rather than silently treated as absent.
- Added tests for differing Fast/Slow head geometry and invalid explicit overrides.
- **Status: PARTIAL.** The full set of original Firefly configurations and checkpoint-name aliases is not implemented or verified.

### F08 — reference-audio decoding

- Added explicit opt-in external FFmpeg codec fallback using `ReferenceAudioLoader::with_ffmpeg(...)`; built-in Symphonia decoding remains primary and does not require FFmpeg.
- Audio data stays in-process/in-memory in the Rust pipeline. External FFmpeg is invoked via direct executable path with pipes, no shell or Python.
- Handles streaming WAV's unknown RIFF/data lengths, enforces 512 MiB transcode-output cap, and closes FFmpeg stdin concurrently with output reads.
- Tests synthesize and decode MP3, AAC/ADTS, ASF/WMA and MP4/M4A using an installed FFmpeg. The test is skipped when FFmpeg is not available.
- **Status: PARTIAL.** The extra codecs require an installed FFmpeg executable; no self-contained native decoder or bundled redistributable is supplied.

### F11 — TensorRT Slow AR model weights, groundwork

- Added deterministic extraction, deduplication and shape checks for `TrtBlueprint`'s named constant weights across AR segments.
- Added checked materialization of FP32 CUDA upload bytes from checkpoint F32/F16/BF16 weights, retaining the expected TensorRT tensor shape.
- Added memory-budget-aware CUDA upload and synchronization before releasing pinned host source bytes.
- Added `upload_ar_weights(...)` entry point for subsequent real Slow/Fast/Naive AR engine binding.
- **Status: PARTIAL.** Real weighted Slow AR execution and persistent device KV integration still need implementation and GPU testing.

## Completed local checks

- `cargo fmt --all -- --check`: PASS.
- `git diff --check`: PASS.
- Outer Rust library: **42 PASS, 2 ignored** (`cargo test --offline --locked --lib`).
- Core Rust default suite: PASS (`cargo test --manifest-path core/Cargo.toml --locked --offline`).
- `cargo check --offline --locked --features cuda-driver`: PASS.
- `cargo check --offline --locked --features 'cuda-driver tensorrt-api'`: PASS with `LIBCLANG_PATH=/usr/local/swift/usr/lib`.
- `cargo test --offline --locked --features tensorrt-mock`: PASS (including 4 blueprint and 1 TensorRT mock integration tests).
- No Windows RTX, real TensorRT engine, model checkpoint, or output-audio parity test occurred; do not extrapolate.

## Parity tracking — separate dimensions

**Functional coverage**: 15 DONE + 7 PARTIAL of 22 = **84.1%** by prior equal-weight DONE=1/PARTIAL=0.5 convention; F01/F08/F11 improved inside PARTIAL but are NOT promoted to DONE.

**Operability**: **7/11 previously validated historical gates**; local Linux compilation/mock regressions are newly reconfirmed, but Windows/CUDA/real-weighted inference remain unverified.

Remaining functional items in order:

1. **F01**: additional Firefly checkpoint/configuration variants, semantic tensor aliases.
2. **F08**: self-contained audio codec coverage independent of external FFmpeg.
3. **F11**: fully bound, live weighted Slow AR prefill and TensorRT output parity.
4. **F12**: weighted Slow AR decode, persistent GPU KV and active-prefix advancement.
5. **F13**: GPU Fast AR sampler, complete autoregressive multi-codebook emission, EOS and reset.
6. **F14**: native Naive AR TensorRT inference.
7. **F22**: unified ownership and memory accounting for all CUDA/TensorRT allocations, workspace, scratch, warmup/unload and cross-request reuse.

## NEXT cycle

Continue F01 against the original Fish Speech config and checkpoint schema when reference assets are available; otherwise proceed to F11 on testable native engine-binding operations rather than reporting a false DONE for F01 or F08. No Python or audio checkpoints are vendored into this repository.
