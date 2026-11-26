# F5-TTS native — CONT04

## Scope and source

Started from local CONT03 commit `383b1a61f00af6787119666649ae7d647f33151e`, corresponding to published `codex/f5-tts-rust-cont03` (`3aa268f3757e50c6a048571b31664f2f09a9afcf`). The upstream F5-TTS v1 source supplied by the user is the reference for the DiT/CFM paths; the separately shipped Vocos package was **not** part of the supplied F5 source ZIP. Vocos graph structure used here is a model-specific implementation that must still be checked against the exact `charactr/vocos-mel-24khz/config.yaml` and original model parameters. `maryml` remains unopened; no ONNX and no changes to the Fish Speech, RVC, or Legacy paths.

## Implemented in CONT04

- `fft.rs`: FFT and inverse FFT radix-2 (1024-point), Hann-window ISTFT with normalized overlap-add and same-length alignment. Independent PyTorch 2.10 CPU reference fixture for spectral frames. The ISTFT CPU stage runs once per decode, not once per DiT/CFM iteration.
- `mel.rs`: uses one shared FFT primitive with the new ISTFT path (preserved mel numerical parity).
- `vocos.rs`: native Burn/CubeCL ConvNeXt backbone, Vocos mel projection, 1D depthwise kernels, LayerNorm, pointwise FFN, GRN, linear spectral head and complex spectrogram synthesis. All neural operations remain in Burn until the final head tensor download. Supports a configuration struct for precise model-specific weight shapes. Uses `.safetensors` checkpoints; the original PyTorch `pytorch_model.bin` format is **not yet directly supported**.
- `pipeline.rs`: specialized F5 v1 orchestrator accepting normalized token IDs, a prepared 24 kHz PCM reference and an explicitly requested total frame count. It connects the resident model DiT and CFM with generated-only mel and Vocos. It uses a deterministic native Gaussian sampler, **not PyTorch's exact RNG**, so seed-to-seed waveform equality with PyTorch is not claimed.
- `wav.rs`: mono PCM16 WAV packaging, 24 kHz, with finite-input checks and clipping.
- Independent toy-size (8-dimensional / 2-layer) PyTorch fixtures test the neural Vocos head and the final waveform. This proves the chosen slice's numeric behavior, **not** parity of the whole official pretrained Vocos network or its serialized checkpoint.

## Tests and proof scope

- `cargo test --offline --lib`: **27/27 PASS**.
- `cargo test --features cubecl-cpu --offline --lib`: **34/34 PASS**; includes both preexisting DiT parity slices plus the new Vocos head/ISTFT PyTorch parity fixtures.
- `cargo check --features cubecl-wgpu --offline`: **PASS**.
- `cargo check --features cubecl-cuda --offline`: **PASS**.
- `cargo fmt --all`: **PASS**.
- Native GPU inference **NOT TESTED**, Windows backend **NOT TESTED**, F5 v1 full checkpoint load **NOT TESTED**, actual pretrained Vocos checkpoint load **NOT TESTED** and audible speech **NOT YET GENERATED**.

## Functional progress: 50/100 (engineering coverage estimate)

| Subsystem | Completed / total |
|---|---:|
| Runtime, vocabulary, checkpoints, CFM and mel conditioning | 13 / 15 |
| DiT Transformer, time embedding, attention | 12 / 25 |
| Text encoder, ConvNeXt and conditioning embeddings | 8 / 15 |
| Audio frontend and Vocos waveform decoding | 12 / 15 |
| Cross-Lingual, SpeedTransformer and duration | 4 / 10 |
| Native TensorRT-RTX backend | 0 / 10 |
| Real-weight execution, IPC, end-to-end voice runtime | 1 / 10 |

## Operability: 3/7 independently verified gates (43%)

1. CubeCL CPU tests, including limited independent PyTorch parity: PASS.
2. WGPU feature builds: PASS, hardware runtime untested.
3. CUDA feature builds: PASS, hardware runtime untested.
4. Real WGPU GPU execution: NOT TESTED.
5. Real CUDA GPU execution: NOT TESTED.
6. Windows backend/runtime: NOT TESTED.
7. Full original checkpoint + speech waveform and RTF: NOT TESTED.

## Next critical steps

1. Inspect the **exact** pretrained Vocos `config.yaml` / actual weights in a future local source bundle, match every weight and phase convention, and implement a supported weights import (direct native `.bin` or explicit offline conversion without an ONNX execution dependency).
2. Validate full DiT v1 checkpoint against reference F5 and then an end-to-end synthetic/PyTorch parity run at the same supplied Gaussian tensor.
3. Implement Cross-Lingual SpeedPredictor and language tokenization/segmentation with PyTorch fixtures.
4. Implement native TensorRT-RTX graph and backend-specific execution once CubeCL is numerically verified; run Windows WGPU/CUDA/RTX tests separately from functionality.
