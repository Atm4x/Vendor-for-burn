# F5-TTS native — CONT03

**Source:** isolated F5-TTS inference implementation (`models/f5_tts`). Started from locally restored CONT02 commit `a813dc4a74ca66cdd24753c0a7f0483d82d602f3`, corresponding to GitHub CONT02 `390b0bab35c5e9ec67160bdf2f86d76999a2b198`. All changes were made locally before GitHub publication. `maryml` was not opened. No ONNX, no changes to Fish Speech/RVC/Legacy code.

## Functionality implemented in this contour

- Added a dedicated Vocos-compatible **mel frontend** in pure Rust (`mel.rs`): 24 kHz, mono, Hann window, FFT 1024, hop 256, 100 HTK mel channels, power 1, reflect padding, log clamp 1e-5. The FFT is radix-2, O(N log N), not a naive DFT. The mel-filterbank multiplication is CPU baseline; optimization/placement on GPU remains future work.
- Added reference PCM preprocessing (`reference.rs`): aligned channel downmix, clipping to 12 seconds, RMS restoration contract (0.1 target), mel extraction, preparation of padded reference mel and extraction of the generated-only mel suffix. The initial frontend accepts **24 kHz PCM only**, and rejects other sample rates; production-quality resampling is not yet implemented.
- Corrected CubeCL CFM Euler integration (`backend/cubecl.rs`): return the original reference-mel prefix unchanged, as done by Python `CFM.sample()` after integrating noisy latents, while retaining the generated suffix. The method now accepts explicit `reference_frames` rather than trying to infer the reference span from zeros in the tensor.
- Added deterministic `torchaudio` CPU fixture of real-valued reference waveform and mel output, plus standalone development-only fixture generator. The runtime library does not depend on Python or Torch.

## Numerical and compiler results

- `cargo test --offline --lib` (default): **22/22 PASS**.
- `cargo test --features cubecl-cpu --offline --lib`: **27/27 PASS**, using the supplied local Tracel LLVM 23.1.0; includes the original DiT block/ConvNeXt fixtures, new mel parity and reference-prefix CFM integration test.
- `cargo check --features cubecl-wgpu --offline`: **PASS** (not device execution).
- `cargo check --features cubecl-cuda --offline`: **PASS** (not device execution).
- `cargo fmt --all`: **PASS**.
- New mel parity test matches independent PyTorch 2.10 + torchaudio 2.10 CPU fixture within max absolute log-mel error **2e-3** on a broadband signal. This is not a blanket numerical bound for silence/low-energy bins or all waveforms.

## Functional progress: 43/100 (engineering coverage estimate)

| Subsystem | Completed / total |
|---|---:|
| Runtime, vocabulary, checkpoints, CFM and mel conditioning | 13 / 15 |
| DiT Transformer, time embedding, Attention | 12 / 25 |
| Text encoder, ConvNeXt and conditioning embeddings | 8 / 15 |
| Audio frontend and Vocos waveform decoding | 6 / 15 |
| Cross-Lingual, SpeedTransformer and duration | 4 / 10 |
| Native TensorRT-RTX backend | 0 / 10 |
| End-to-end real model weights, IPC, voice runtime | 0 / 10 |

## Operability: 3/7 independently verified gates (43%)

1. CubeCL CPU tests with PyTorch parity **PASS**.
2. CubeCL WGPU `cargo check` **PASS** (device runtime pending).
3. CubeCL CUDA `cargo check` **PASS** (device runtime pending).
4. WGPU GPU execution **NOT TESTED**.
5. CUDA GPU execution **NOT TESTED**.
6. Windows backend/runtime **NOT TESTED**.
7. Real checkpoint and waveform/RTF **NOT TESTED**.

## Remaining critical path

- Load a full real F5TTS_v1_Base checkpoint and confirm all 22 layers, EMA weight names and end-to-end DiT numerical parity with PyTorch.
- Implement Vocos decoder (neural ConvNeXt backbone + head/ISTFT), WAV ingest and resampling, then generate a waveform from a real checkpoint.
- Implement Cross-Lingual SpeedTransformer and compatible syllable tokenization, with separate numerical fixtures.
- Add independent TensorRT-RTX backend without ONNX; maintain CubeCL CUDA/WGPU.
- Validate GPU runtime and Windows separately from functional scope, including RTF.
