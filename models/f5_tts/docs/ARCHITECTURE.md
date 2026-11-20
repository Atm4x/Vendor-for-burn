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

## Initial implementation status (CONT01, historical)

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

## Original planned vertical slices (historical)

1. Build a strongly validated `V1Weights` schema, reusing only the model-specific keys, and zero-copy/batched weight uploads where possible.
2. Implement `DiT` with native Burn/CubeCL modules: text embedding + four ConvNeXtV2 blocks, 2× mel input projection, position conv, timestep embedding, 22 adaptive-normalization Transformer blocks, final AdaLN and mel projection. Exact RoPE and CFG packing.
3. Implement 24 kHz mel frontend and Vocos backend, then deterministic sampling with PyTorch reference vectors and real v1 weights.
4. Add Cross-Lingual SpeedTransformer and compatible English hyphenation (e.g. patterns equivalent to `pyphen en_US`); do not substitute a naive syllable estimate.
5. Add independent TensorRT-RTX native engine for DiT/SpeedPredictor/Vocos, without ONNX.
6. Benchmark RTF, initialization, NFE latency, VRAM and waveform parity separately on NVIDIA Windows and WGPU hardware.

## Tests and acceptance gates

*Functional completeness* and *hardware operability* are distinct. A correct timetable implementation is **not** a working DiT. A compiling CUDA backend is **not** proof that audio inference works. A real CubeCL DiT forward + numeric comparison must precede any claim of functional parity.

No original Fish Speech files or legacy code should be changed while implementing F5-TTS.

## CONT02 — model-specific DiT execution

- `dit.rs`: timestep embedding, RoPE, modulation/attention/FFN blocks, backbone, parameterized checkpoint loader.
- `dit_input.rs`: F5-specific text embedding, 4-layer ConvNeXt V2, input projection and grouped positional convolutions.
- `dit_model.rs`: F5 DiT forward with CFG packing, text/RoPE request caching, `DiTVelocity` bridge.
- Model weights are imported by exact PyTorch names; tensors remain on the selected Burn/CubeCL device during DiT and CFM loops.
- Only reduced-size PyTorch reference slices have numerical parity tests so far. Complete model inference and Vocos are pending.

## Current architecture after CONT06

- `text_frontend.rs`: model-specific standard reference+generation text contract, English/CJK punctuation normalization, deterministic native tokenization, explicit pinyin provider interface and `[voice]` script parser. No Python processing or tokenizer is required for ASCII English. Chinese tone sandhi is **not bundled**: the caller must supply a correct pinyin provider; with `NoPinyin`, raw Chinese is rejected rather than silently mapped to unknown IDs.
- `pipeline.rs`: standard text, multi-speaker script and cross-lingual English entrypoints; one seeded RNG stream across sequential chunks, without per-chunk reseeding. The RNG is intentionally **not PyTorch-equivalent** yet.
- `resample.rs`: audio input resampling to 24 kHz by Hann-windowed sinc polyphase filters, following torchaudio defaults. Reference channel downmix and RMS are computed before resampling, consistent with the reference pipeline. Supported rates are bounded for memory safety.
- `dit_model.rs`: exact-F5-v1 production configuration is unchanged. A private verified-constructor seam supports a reduced-dimensional end-to-end DiT PyTorch fixture with both CFG branches. The **test has not been run through CubeCL** because this runtime lacks offline Burn crates.
- Backends: CubeCL CPU/CUDA/WGPU remain Burn-backed. TensorRT-RTX is a separate planned backend, not implemented here. No ONNX.

### Current functional boundaries

- Core DiT/CFM/Vocos path is implemented but the production model is not yet numerically validated end-to-end. Complete DiT fixture is available; running it on CubeCL CPU is a required next step.
- Chinese `rjieba` segmentation and polyphonic pinyin tone sandhi are not implemented as a self-contained Rust subsystem. Do not claim Chinese end-to-end support until both are numerically validated.
- `torch.randn` RNG parity, entire model checkpoint mapping and TensorRT-RTX are separate remaining functional work, not evidence of GPU workability.
