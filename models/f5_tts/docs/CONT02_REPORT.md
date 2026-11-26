# F5-TTS native — CONT02

Source: F5-TTS v1 original Python modules (`dit.py`, `modules.py`) from the user-uploaded archive. `maryml` was not opened. Base repository branch: `codex/fish-speech-cont48-tensorrt` (`12ef7d2dbe8b62d3a3a530bf9f4ed9611e45fc00`). Previous local F5-TTS commit: `7a4d0de35df9fb8c7022fb92421226b7171bc8f9`.

## Completed changes

- Corrected the Burn 0.22 `Int`/`Tensor` imports for CubeCL.
- Installed supplied Tracel LLVM 23.1.0-4 locally (not committed).
- Added tensor-based F5 DiT primitives: time sinusoidal embedding and MLP, RoPE, Q/K/V SDPA, AdaLayerNorm modulation, tanh-GELU FFN, final AdaLN and mel projection. F5TTS_v1_Base uses a 22-block backbone.
- Added F5-specific text embedding with masking, sinusoidal positions and ConvNeXt V2, grouped positional Conv1d audio embedding and GRN.
- Added `F5DiT` with packed conditional/unconditional CFG execution, per-request text/RoPE caches, and `DiTVelocity` integration. Device-resident Euler trajectory is unchanged.
- Checkpoint importer now decodes F32, FP16 and BF16 weights, enforces exact parameter shapes and rejects non-finite values; direct PyTorch-style names are used. No ONNX.
- Added two independent PyTorch 2.10.0 CPU numerical fixtures (small deterministic F5 DiT block and text/InputEmbedding). No user model weights required.

## Verified locally

- `cargo test --offline` (no backend): 17/17 PASS.
- `cargo test --features cubecl-cpu --offline`: 21/21 PASS, including two independent PyTorch numerical parity tests (each `max_abs <= 2e-4`).
- `cargo check --features cubecl-wgpu --offline`: PASS.
- `cargo check --features cubecl-cuda --offline`: PASS.
- `cargo fmt --all`: PASS.
- Linux CPU tested with supplied Tracel LLVM 23.1.0-4. CUDA and WGPU were **not executed** on real GPUs. No complete F5 checkpoint and no synthesized waveform was tested.

## Progress — functionality (estimated weighted implementation coverage)

**36 / 100**. This is an engineering estimate of *completed feature scope*, not an empirical quality score.

| Scope | Earned / possible |
|---|---:|
| Runtime model contracts, vocabulary, checkpoint, CFM | 12 / 15 |
| DiT Transformer backbone, attention, time embedding | 12 / 25 |
| Text encoder and conditioning embeddings | 8 / 15 |
| Mel frontend + Vocos waveform decoding | 0 / 15 |
| Cross-Lingual speed/duration including SpeedTransformer | 4 / 10 |
| Native TensorRT-RTX inference backend | 0 / 10 |
| Complete real-weight inference and voice session workflow | 0 / 10 |

The DiT model path is now expressed as executable tensor operations but **not yet checked with original full-size checkpoint**. Current numerical parity covers reduced-width deterministic test vectors only.

## Progress — operability (independent gates)

**3 / 7 gates (43%)**:

1. CubeCL CPU tests + PyTorch vector parity — PASS.
2. CubeCL WGPU source compilation — PASS; device execution pending.
3. CubeCL CUDA source compilation — PASS; device execution pending.
4. WGPU actual device runtime — NOT CHECKED.
5. CUDA actual device runtime — NOT CHECKED.
6. Windows backend/runtime — NOT CHECKED.
7. Real checkpoint, waveform and RTF — NOT CHECKED.

## Next functionality steps

1. Verify full F5TTS_v1_Base checkpoint layout, inspect loading of EMA model and compare all intermediate tensors against PyTorch (attention, text encoder and output mel).
2. Implement reference-audio frontend: resample, mel STFT/filterbank and RMS normalization.
3. Port Vocos and end-to-end audio waveform decoding.
4. Implement Cross-Lingual SpeedTransformer and `pyphen`-compatible syllabification.
5. Implement TensorRT-RTX native backend separately (no ONNX), preserving model-specific runtime contracts.
6. Execute end-to-end CPU/WGPU/CUDA, then Windows parity/RTF independently of functionality progress.

No Fish Speech, RVC, or legacy source files were modified.
