# F5-TTS Native — CONT06: functional coverage before hardware verification

## Scope / provenance

- Input: source-only CONT05 ZIP, which matches published `codex/f5-tts-rust-cont05` at GitHub HEAD `3f9caf880d31a73890314f72b2126d097d293156`.
- All modifications confined to `models/f5_tts` in an isolated local workspace.
- Reference sources: user-supplied original `F5-TTS-main (1).zip` and the previously provided Cross-Lingual Python classes. The `maryml` archive was **not opened**.
- No ONNX. No generic TTS/NN engine. CubeCL CPU/CUDA/WGPU remains the primary designed execution engine; separate TensorRT-RTX remains planned.

## Functional changes

1. `text_frontend.rs`: native standard reference transcript punctuation, reference+generation concatenation, UTF-8 byte-based duration heuristic including short-text speed correction and fixed total-duration override. Native ASCII, punctuation normalization, F5 vocabulary mapping, Chinese syllable adapter with explicit `NoPinyin` rejection and a pluggable tone-sandhi provider contract.
2. Multi-voice `[voice]` script parser and corresponding `StandardF5Runtime::synthesize_script` with voice-specific audio/transcript and fallback to `main`; joins segments exactly by concatenation.
3. `StandardF5Runtime::synthesize_text` / `synthesize_text_with_pinyin` and `CrossLingualF5Runtime::synthesize_english` / `synthesize_with_pinyin`: built-in model-specific text entrypoints, no longer dependent on an external token-ID closure for the English path.
4. `resample.rs`: native polyphase Hann-windowed sinc resampler to 24 kHz with torchaudio default lowpass width 6 and rolloff 0.99, enabling 8/16/22.05/44.1/48/96 kHz audio input. RMS, channel downmix and optional gain occur before resampling.
5. Request-owned RNG state shared across Cross-Lingual and multi-voice chunks; fixes reseeding by `seed + chunk_index`. The RNG algorithm itself still differs from PyTorch and is not claimed bit-identical.
6. `dit_model.rs`: end-to-end reduced-size DiT PyTorch parity fixture, including timestep input, text ConvNeXt, conditional/unconditional inputs, two DiT blocks, final AdaLN, output projection and CFG. Production `DiTConfig::V1_BASE` remains fixed. The fixture is ready but the CubeCL execution test has **not yet been run**.

## Numerical / compilation evidence (current cycle only)

- `rustc 1.100.0` standalone `text_frontend.rs` + `vocab.rs` harness: **5/5 PASS**.
- Rust polyphase sinc vs PyTorch/torchaudio `2.10.0` reference, identical FP32 input values: **six sample-rate cases PASS** (8k, 16k, 22.05k, 44.1k, 48k, 96k → 24k); maximum absolute error over all cases **5.961e-8**.
- Independent PyTorch full-DiT reduced-configuration reference: fixture generation **PASS** (53 weight tensors, 2 layers, 2 CFG branches). Rust parity test added, **not executed**.
- `cargo fmt --check`: PASS.
- `cargo test --offline`: **BLOCKED**, dependency resolver reports missing cached `burn` package. No claim of the full suite PASS.
- Previous CONT05 numerical validations belong to the earlier cycle; they were **not rerun**.
- CUDA/WGPU/Windows hardware inference, pretrained model weight execution and TensorRT-RTX: **not tested** in this cycle.

## Functional coverage estimate: 67/100 (previous 59/100)

| Area | Completed / total | Remaining |
|---|---:|---|
| Core runtime/checkpoint/CFM | 14/15 | Final semantic integration details |
| DiT computation | 12/25 | Execute complete DiT test; long-context/model-scale parity; RNG parity |
| Text preprocessing and normalization | 14/15 | Self-contained polyphonic Chinese segmentation/pinyin |
| Audio reference, mel, Vocos | 13/15 | Exact production Vocos weight mapping and remaining audio semantics |
| Cross-Lingual segmentation/speed | 9/10 | Numerical end-to-end parity / chunk source RNG |
| TensorRT-RTX backend | 0/10 | Entire backend |
| Integration, IPC, voice scripts | 5/10 | Worker/session/IPC switching and acceptance |
| **Total** | **67/100** | **33/100** |

These weights reflect feature coverage, not a verified-running percentage. Full pretrained audio generation is NOT certified.

## Workability (secondary): 3/7 = 43%, unchanged

Previously confirmed in CONT05: standalone CPU computations and CUDA/WGPU compile. New standalone Rust numerical tests pass, but the full dependency-backed Burn test suite is unavailable in this container. Hardware workability is not extended by this result.

## Priorities for CONT07

1. Restore local offline Burn dependency cache from the correct Fish Speech source environment without modifying other model folders. Run the complete `dit_full_small.safetensors` PyTorch parity test and resolve **actual numerical** mismatches, including CFG and caching. This is a numerical correctness gate, not a GPU operability exercise.
2. Implement self-contained Chinese word segmentation + context-aware tone-sandhi pinyin, with a Python reference corpus. Avoid a hand-written limited dictionary falsely presented as complete pinyin support.
3. Extend deterministic full-path CPU parity to Vocos real structure and F5 CFM trajectory. Then introduce the independent native TensorRT-RTX backend.
4. Later: hardware acceleration, Windows, production checkpoint execution, voice IPC and benchmark RTF.
