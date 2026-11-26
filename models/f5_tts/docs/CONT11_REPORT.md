# F5-TTS Rust — CONT11: end-to-end numerical parity through PCM16

Date: 2026-11-20. Based on GitHub branch `codex/f5-tts-rust-cont10` commit `56fef5783ef51ab7195f1943579f42662473401c`. All modifications confined to `models/f5_tts`. `maryml` and ONNX were not used.

## Objective and implementation

Validate the existing native 100-mel F5 inference path from fixed Gaussian noise and reference conditioning through **DiT → CFG → CFM Euler integration → reference inpainting → generated mel extraction → Vocos ConvNeXt and spectral head → ISTFT → mono PCM16 WAV**. The calculation is entirely Rust/CubeCL CPU; PyTorch runs only in the independent fixture generator.

The independent PyTorch generator `tests/tools/generate_e2e_audio_fixture.py` reuses the previously audited F5 DiT and Vocos oracle architectures, with a 100-channel, 16-dimensional, two-layer synthetic DiT and a two-layer, 8-dimensional Vocos. DiT and Vocos read byte-for-byte identical safetensors weights in the Rust tests. Two cases share the same checkpoints but differ in conditioning, seed-derived noise, reference prefix length, CFG, duration and timeline:

1. **Uniform Euler:** three integration steps, 9 mel frames, 3 reference frames, CFG=1.7, gain=0.75.
2. **EPSS + sway:** five integration steps (6 timeline points), 8 mel frames, 2 reference frames, CFG=2.2, sway=-1, gain=0.6.

Both cases compare four independently calculated arrays, plus final quantized audio. The fixture generator is reproducible: SHA256 of all generated test assets did not change on rerun. Unlike isolated Vocos or DiT checks, these cases exercise the actual Rust `integrate` path and a real connection between model components.

The Vocos oracle was made import-safe by moving fixture writes into its `__main__` entrypoint. The DiT oracle gained an optional explicit timeline, preserving its previous default behavior. No production neural-network forward algorithm needed alteration.

## Quantitative result — CubeCL CPU vs PyTorch CPU

| Stage | Uniform (max abs) | EPSS + sway (max abs) |
|---|---:|---:|
| Full DiT/CFM/inpainted mel | 3.13e-7 | 5.66e-7 |
| Generated mel | 3.13e-7 | 5.66e-7 |
| Vocos spectral parameters | 1.49e-7 | 1.94e-7 |
| Floating-point waveform / ISTFT | 2.2e-8 | 2.2e-8 |
| PCM16 encoded samples | 0 mismatch | 0 mismatch |

Both integration cases **PASS**, with per-stage float tolerance of `2e-4` and final PCM quantization comparison. Each produces 1536 PCM samples (6 frames × 256 samples). This proves numerical parity for the chosen *synthetic reduced-width models*, not pretrained model quality or runtime performance on GPU. The independent PyTorch source is a mathematical oracle matching the original operators, not execution of production F5-TTS on real checkpoints.

## Regression checks and incidental correction

- `cargo test --offline --locked --test offline_frontend`: **35/35 PASS**.
- `cargo test --offline --locked --lib`: **46/46 PASS** (without optional CubeCL feature).
- `cargo test --offline --locked --features cubecl-cpu --lib vocos_backbone_head_and_waveform_match_pytorch`: **PASS**.
- `cargo check --offline --locked --features cubecl-wgpu`: **PASS**, no warnings.
- `cargo check --offline --locked --features cubecl-cuda`: **PASS**, no warnings.
- Full CPU suite with all 62 CubeCL tests was *not* run in one command; no claim of 62/62 PASS.

An existing old unit test erroneously rejected a 16 kHz source sample rate, despite the accepted and previously verified resampling support. That stale assertion was corrected to reject invalid rates 0 and 384001. Removed a production unused import (moved the sample-rate constant into the test module) to keep WGPU/CUDA compilations warning-free. No change to the reference resampling semantics.

## Reproduce offline

Rust 1.100.0, Burn 0.22.1, `cubecl-cpu` LLVM 23.1.0. Cargo vendor from GitHub Actions run [44741264056](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44741264056); vendor is an external artifact, **not** included in the commit.

Set `CARGO_HOME`, `CARGO_TARGET_DIR`, `LLVM_SYS_231_PREFIX` and `LLVM_CONFIG_PATH` to a working local LLVM/Cargo environment. Use `--offline --locked` with the vendored Cargo source configuration. Then in `models/f5_tts`:

- `python tests/tools/generate_e2e_audio_fixture.py`
- `cargo test --offline --locked --features cubecl-cpu --lib full_dit_cfm_vocos_pcm_matches_independent_pytorch -- --test-threads=1 --nocapture`
- `cargo test --offline --locked --features cubecl-cpu --lib full_dit_cfm_vocos_epss_sway_pcm_matches_pytorch -- --test-threads=1 --nocapture`

## Remaining functional gaps

1. Confirm matching numerics for **production-width 1024-dim, 22-layer** model at the model-level and test integration against original pretrained checkpoint when later prioritized. Current tests verify 22-layer *depth* separately and 100-channel complete audio on a two-layer synthetic model.
2. Achieve complete native Chinese segmentation, polyphonic lexicon and tone sandhi parity with `rjieba`/`pypinyin`. Existing Chinese frontend is partial.
3. Implement separate **TensorRT-RTX** backend, maintaining the same model-specific execution contract without ONNX.
4. Add complete **CUDA/WGPU execution**, Windows, true checkpoint compatibility and benchmarks to the *separate* workability track; do not conflate compilation with working GPU inference.

## Progress

- **Functionality:** estimated **78%**, up from 76%, reflecting the confirmed end-to-end numerical calculation on two representative timeline modes and their complete PyTorch oracle fixtures. This remains an engineering estimate, not percentage of tests passed.
- **Workability:** **43% (3/7)**, unchanged. No hardware inference, Windows, or pretrained runtime validation performed.
