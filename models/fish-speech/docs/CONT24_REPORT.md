# Fish Speech 1.5 — CONT24: fail-closed Slow/Fast KV prefix preflight

## Source and publication provenance

- Repository: `Atm4x/Vendor-for-burn`, branch `codex/fish-speech`.
- Base: verified CONT23 Git Bridge bundle. `source-head.txt` = `d196af1c577027f6eff8b9221cf6b5d6eb7f45db`, equal to remote branch HEAD at start.
- Bundle SHA-256 (`repository.bundle.sha256`) matched, `git bundle verify` confirmed complete history.
- All implementation and compilation were local using the attached Rust 1.100.0 Linux toolchain and offline vendored dependencies.

## Implementation: F12 prefix correctness (PARTIAL)

1. `KvDeviceOwner::preflight_writes` validates every planned K/V barrier before submitting any TensorRT network. Checks append-only active-prefix position, matching K/V buffer geometry, size consistency with resident device KV, and checked prefix addition. Projected lengths are tracked across the plan without changing the actual live cache slots.
2. `ArStreamExecutor::enqueue` runs the new KV preflight before any allocation, network dispatch, or scratch reservation. An accidental second prefill at position zero after an existing prefill is rejected without corrupting the healthy session; a subsequent valid decode succeeds. Invalid position is no longer detected only after some GPU networks have already launched.
3. `ArStreamExecutor` separately checks required external inputs for **all** network segments before dispatch. A missing weight or runtime tensor first needed in a later segment no longer permits earlier segments to run before failing. This complements the public `ArInputContract`, but also protects direct Core executor callers.
4. `WeightedCudaArSession::cache_active_len` exposes the existing authoritative logical KV-prefix length for debugging/validation across prefill and decode.
5. Regression tests cover repeating stale prefill, continuing with valid decode after rejection, and omitting an input first used in a later network segment with zero network submissions.

### Remaining limitations

- Neither `cached_ar_dispatch` nor TensorRT engine-session lifetime has been redesigned: contexts are still created per segment, and that path synchronizes each segment for safety.
- No real weighted Slow AR prefill/decode output parity, model checkpoint run, Windows RTX execution, or device profiler result was produced.
- F01 and F08 still need original-checkpoint variants and independent codecs, respectively. F11, F12, F13, F14 and F22 remain partial.

## Evidence from local verification

- `cargo fmt --all -- --check`: PASS.
- `git diff --check`: PASS.
- `cargo test --offline --locked -p fish-speech-core-prototype`: main suite **192 passed, 2 ignored, 0 failed**; Core integration suites **all passed** (3, 12, 4, 2, 1, 4, 1, 1 tests respectively).
- `cargo test --offline --locked -p fish-speech-vendor`: **45 passed, 2 ignored, 0 failed**.
- `cargo check --offline --locked --features 'cuda-driver tensorrt-api'`: PASS (`LIBCLANG_PATH=/usr/local/swift/usr/lib`).
- `cargo test --offline --locked --features 'tensorrt-mock'`: main suite **45 passed, 2 ignored, 0 failed**; blueprint **4 passed**, TensorRT mock **1 passed**.
- The initial CUDA-feature and TensorRT-mock compilations exceeded the execution limit after a fresh checkout; subsequent invocations with the same Cargo cache completed successfully. No false PASS was recorded for the incomplete attempts.

## Progress, independently tracked

- Functional parity: 15 DONE, 7 PARTIAL out of 22 (equal-weight score **84.1%**); F12 preflight improves coverage but is not a completed end-to-end feature. No inflation.
- Operability: 7/11 historically verified gates (**63.6%**) unchanged. Linux compilation/mock reconfirmed. Device-specific gates remain NOT TESTED.

## Next cycle

1. Address remaining real-device F11 Slow prefill gap: map genuine input names, check TensorRT engine tensor bindings/profile and compare returned logits to PyTorch on Windows RTX checkpoint.
2. F12: keep serialized engines and execution contexts resident across prefill/decode; remove per-segment synchronization only with proven same-stream ordering and safe completion-boundary retirement. Validate prefix advance and reset on real checkpoints.
3. F13: native GPU Fast AR iterative multi-codebook decoding and sampler/EOS/cache reset.
4. F14: native Naive AR TensorRT full inference.
5. F22: account TensorRT workspace/context scratch and warmup/unload under unified GPU memory budget.

No model weights, audio recordings, or new third-party dependencies have been committed.
