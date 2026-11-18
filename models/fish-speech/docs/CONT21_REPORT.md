# Fish Speech 1.5 — CONT21: verified Git Bridge and CUDA allocation ownership

## Source provenance

- Target branch: `Atm4x/Vendor-for-burn:codex/fish-speech`.
- Source Git Bridge bundle: `8415e3650fcb8043f118b7f126d6999f5d8fa452` in `source-head.txt`; SHA-256 and `git bundle verify` passed.
- The bridge was refreshed by rerunning the export job after detecting that the prior completed run's bundled commit differed from remote HEAD. A rerun retains the original run's `head_sha` metadata; **the archive's `source-head.txt` is authoritative** because the workflow explicitly checks out the branch by name and verifies its own HEAD before bundling.
- Local repository includes `6a9f0393` (CONT6–CONT19 integration); do not re-import older source archives over it.

## Implemented in this cycle

- `src/device_memory.rs`: thread-safe live-allocation ledger with overflow-checked reservation, exact lifetime-based release, peak usage, and optional explicit allocation cap.
- `src/cuda_stream.rs`: BF16 KV cache and generic TensorRT segment outputs now reserve bytes before allocating and release quota only after their last owning `CudaBuffer` drops. Failed CUDA allocations also release their reservation.
- `CudaArDevice::with_shared_budget` permits one budget across multiple workers/requests; `with_memory_budget` configures a device-specific cap; `memory_usage` exposes allocated, peak, and limit bytes. `new` retains the prior API with unbounded accounting by default.
- `vendor/cudarc-0.19.10/.cargo-checksum.json`: corrected the **single mismatched file checksum** for the already-present `src/cudnn/mod.rs` bytes. No vendored Rust source was changed. The mismatch in the published baseline prevented Cargo from checking optional CUDA/TensorRT features.

## Independent verification (Linux, Rust 1.100.0, offline)

- Core tests: 218 passed, 2 ignored; these are numerical/contract and synthetic CPU tests.
- Outer crate unit tests: 40 passed, 2 ignored, with three new device-memory tests.
- TensorRT mock integration: 4 blueprint tests + 1 mock-network test passed.
- `cargo check --locked --offline --features cuda-driver`: passed after vendor checksum reconciliation.
- `cargo check --locked --offline --features 'cuda-driver tensorrt-api'`: compiler-only probe; not real NVIDIA inference.
- NO authentic model/TensorRT runtime/RTX testing was performed.

## Functional progress (primary)

- **15 DONE / 7 PARTIAL / 22 total = 84.1% equal-weight**, unchanged; F22 advanced but remains PARTIAL.
- F01: Firefly configuration variants and checkpoint aliases.
- F08: native MP3/AAC/M4A/WMA offline audio decoding.
- F11: real weighted Slow TensorRT prefill.
- F12: persistent device KV, real cached Slow decode, active prefix.
- F13: GPU Fast multi-codebook inference, sampling and EOS/reset.
- F14: native Naive-AR TensorRT inference.
- F22: full accounting including TensorRT workspaces, external buffers, device-side scratch, warmup/unload and safe session reuse. This cycle covers only CUDA allocations performed through `CudaArDevice` (KV and plan outputs), not workspace bytes or all external allocations.

## Operability progress (independent)

- The inherited CONT20 ledger lists **7/11 previously confirmed local gates**, not an RTX readiness percentage.
- This cycle independently refreshed local Core, outer, mock and compile checks; Windows, CUDA execution, original checkpoint parity, and final speed/quality gates remain unverified.

## Next actions

1. Wire TensorRT workspace and engine lifecycle memory into a single shared resource manager with explicit ownership of contexts and warm sessions.
2. Extend genuine weighted GPU F11–F14 paths without counting their Windows validation as functional implementation.
3. Run real inference on Windows RTX with a real Fish Speech checkpoint; separately measure output parity and performance.
