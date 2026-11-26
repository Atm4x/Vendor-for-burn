# F5-TTS CONT14 → CONT15

Base from GitHub published CONT14 branch `codex/f5-tts-rust-cont14`, not Fish Speech. Only `models/f5_tts` may change; retain independent CubeCL CUDA/WGPU and TensorRT-RTX without ONNX, Python runtime or implicit fallback. `maryml` only if explicitly justified for model formula parity (not architectural reuse); do not touch Legacy.

Priority 1: **functional parity and numerical verification**. Next build specialized native TensorRT-RTX Vocos (ConvNeXt, spectral head, ISTFT) and/or SpeedTransformer, choose the bounded complete segment; compare each mathematical stage with PyTorch synthetic fixtures. Preserve fixed-size graph contracts and add correct input-data validation; TRT mock proves only compilation and schema, not engine build. Avoid duplicating native backend model math in another framework.

Priority 2: actual TensorRT runtime owner: warmed reusable engine and execution context, external CUDA library/DLL path, buffer allocations and GPU stream synchronization; fixed-shape CFG and timestep scalar updates, repeated CFM Euler steps with device-resident state. Current `trtx_cfm.rs` defines the fused graph and enqueue binding but no allocator/execution loop.

Priority 3: full actual TRT engine build and GPU parity; do not prematurely count real-weight/Windows checks toward functionality. Keep separate 2-bar progress display and report only verified tests.

Local offline dependencies: `f5-cont12/vendor-union`, Rust 1.100, `trtx 0.9.1+rtx1.7` mock and Burn 0.22.1. Do not re-download vendor. GitHub publish in its own branch from exact previous published HEAD using connector; write only `models/f5_tts` files.
