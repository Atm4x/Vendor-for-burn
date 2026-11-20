# F5-TTS v1 — Native Rust runtime, CONT01

Source baseline: `codex/fish-speech-cont48-tensorrt` at `12ef7d2dbe8b62d3a3a530bf9f4ed9611e45fc00`.
Reference implementation: user-provided `F5-TTS-main (1).zip`, DiT/CFM v1. Cross-lingual contract: user-provided `CrossLingualF5TTSPipeline` and `SpeedPredictor` source.

## Non-negotiable architecture

- This crate is **not** a wrapper around maryml. No maryml files have been opened, copied or changed.
- The canonical model is F5TTS_v1_Base only. There is no generic tensor graph/NN framework, no universal TTS interface, no ONNX conversion or runtime dependency.
- Native alternatives: `CubeCL` (CUDA and WGPU) and `TensorRT-RTX` (native TensorRT API). They do **not** share engine allocations, streams, or executable caches.
- The model path is `reference audio -> mel -> DiT + CFM -> vocoder -> waveform`. Cross-Lingual inserts `SpeedPredictor` and alternate duration/chunking decisions, **not a second DiT implementation**.
- Device-resident CFM state: no per-step host reads. CFG, time embeddings and recurrent conditioning remain in the backend. Clear per-request text caches on both success and error.
- Weight source of truth is the original `.safetensors` with optional `ema_model.` prefix. Import checks names, byte ranges, and shapes; no silent coercion or heuristic renaming.

## Implementation status

| Slice | Current state |
|---|---|
| DiT/CFM shape and v1 configuration | Rust contracts implemented; tests authored |
| Exact Euler, CFG and EPSS/sway timestep equations | CPU numerical reference implemented |
| Cross-Lingual speed classes, duration and sentence chunking | Rust contracts implemented; English `pyphen` dictionary adapter not implemented |
| Checkpoint metadata and single-tensor lazy byte loading | Implemented; untested on real checkpoint |
| CubeCL CUDA/WGPU | Burn/CubeCL feature wiring and on-device Euler loop implemented; no DiT forward yet |
| DiT input embedding, ConvNeXt, attention and 22 blocks | Not implemented |
| Vocos and audio processing | Not implemented |
| Cross-Lingual SpeedTransformer (6 layers) | Not implemented |
| TensorRT-RTX native backend | Not implemented |
| Real model parity and Windows CUDA/WGPU runtime verification | Not performed |

## Next vertical slices

1. Build a strongly validated `V1Weights` schema, reusing only the model-specific keys, and zero-copy/batched weight uploads where possible.
2. Implement `DiT` with native Burn/CubeCL modules: text embedding + four ConvNeXtV2 blocks, 2× mel input projection, position conv, timestep embedding, 22 adaptive-normalization Transformer blocks, final AdaLN and mel projection. Exact RoPE and CFG packing.
3. Implement 24 kHz mel frontend and Vocos backend, then deterministic sampling with PyTorch reference vectors and real v1 weights.
4. Add Cross-Lingual SpeedTransformer and compatible English hyphenation (e.g. patterns equivalent to `pyphen en_US`); do not substitute a naive syllable estimate.
5. Add independent TensorRT-RTX native engine for DiT/SpeedPredictor/Vocos, without ONNX.
6. Benchmark RTF, initialization, NFE latency, VRAM and waveform parity separately on NVIDIA Windows and WGPU hardware.

## Tests and acceptance gates

*Functional completeness* and *hardware operability* are distinct. A correct timetable implementation is **not** a working DiT. A compiling CUDA backend is **not** proof that audio inference works. A real CubeCL DiT forward + numeric comparison must precede any claim of functional parity.

No original Fish Speech files or legacy code should be changed while implementing F5-TTS.
