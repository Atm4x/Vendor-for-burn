# F5-TTS CONT15 → CONT16

Starting point: GitHub branch `codex/f5-tts-rust-cont15` (verify its current HEAD and sync exactly through Git Bridge if the local clone is gone). Retain Rust source only under `models/f5_tts`; do not edit Fish Speech/RVC/Legacy or read MaryML as an architecture baseline. No ONNX, Python runtime or implicit CubeCL fallback.

Primary goal: **functional coverage and mathematical parity**, not first-time Windows/GPU benchmarking.

1. Implement specialized warmed TensorRT-RTX session for the existing CFM Euler-step engine, Vocos spectral decoder and Cross-Lingual SpeedTransformer. The current `trtx_cfm.rs`, `trtx_vocos.rs`, `trtx_speed.rs` expose fixed-shape graph constructors and explicit buffer enqueue contracts but not a complete device allocator/stream/context owner. Follow Fish Speech only for low-level TensorRT API patterns; keep F5 design independent and specialized.
2. Make a clean service API: load/unload model, warm engine/context, accept reference mel/audio/text and return PCM, own external CUDA DLL/SO paths, keep one active TTS model/session, allow different requests and reference voices. Never perform a device→host roundtrip between CFM steps; ping-pong mel buffers. Avoid unsafe lifetime leaks in the runtime owner; stream synchronization on success/failure.
3. Use independent PyTorch fixtures to validate host preparation/CFM schedules/SpeedTransformer and Vocos boundaries. Real TRT graph creation and numerical TRT vs PyTorch requires actual CUDA runtime; mark untested until executed. Mock build cannot certify TRT execution.
4. If you optimize ISTFT to GPU, retain native Rust fallback based on pure FFT/overlap-add, not CubeCL fallback for the TensorRT backend.
5. Preserve separate progress bars: functionality ~92%, workability 43%. Update only with evidence. If publishing, create another dedicated branch from verified CONT15 GitHub HEAD, use Connector rather than web diff, and compare all changed blob SHAs.

Offline: Rust 1.100 at `/mnt/data/f5-cont07/rust/bin`, pinned Cargo vendor union `/mnt/data/f5-cont12/vendor-union`; `CARGO_HOME=/mnt/data/f5-vendor-cont09/cargo-home`, `CARGO_TARGET_DIR=/mnt/data/f5-vendor-cont09/target`.
