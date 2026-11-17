# Native GigaAM v3 CTC inference (Rust / Burn 0.22.1)

This module implements the GigaAM v3 CTC inference network directly with Burn tensors: WAV decoding and resampling, torchaudio-compatible HTK log-mel preprocessing, Conv2D strided subsampling, Macaron Conformer with pre-projection RoPE attention, depthwise Conv1D, CTC classifier and greedy decoding. **No ONNX Runtime, Candle or Python is needed during inference.**

## Usage

Place a supported model weight file (`v3_ctc.ckpt`, `model.ckpt`, `model.safetensors` or `v3_ctc.safetensors`) into a model directory, alongside `config.json`, containing `vocabulary` (a list of character strings), `n_mels`, `n_layers`, `d_model`, `n_heads`, `ff_expansion`, `conv_kernel`, `subsample_kernel`, `subsample_channels` and `num_classes` (`len(vocabulary) + 1`). Omitted numeric fields default to the upstream `ConformerEncoder` constructor's values; these defaults are **not** a verified configuration of any downloaded checkpoint. Always verify configuration from the specific checkpoint's `hyper_parameters` before running inference. The checkpoint must use original GigaAM `state_dict` keys, with the `encoder.` and `head.` prefixes; SafeTensors should contain these keys directly.

```
cargo run --offline -p gigaam-rs --release -- transcribe ./weights ./MitaKind.wav
cargo run --offline -p gigaam-rs --release --no-default-features --features cuda -- transcribe ./weights ./MitaKind.wav --device cuda
```

The transcription CLI supports PCM integer and float WAV, mono or multichannel inputs, resamples to 16 kHz, and splits long recordings into <=24-second pieces near low-energy boundaries. The segmentation is a basic energy heuristic, not neural VAD. `v3_e2e_ctc`, multilingual SentencePiece checkpoints and RNN-T decoding are **not** implemented by this crate.

Parity with the upstream PyTorch model and real-checkpoint validation are required before production use. Audio-only tests cannot establish acoustic recognition quality.

## License

Source architecture adapted from GigaAM (MIT; Copyright (c) 2024 GigaChat Team); original license is retained in `LICENSE`. Any independently distributed model weights are subject to their own terms.

## Verified locally (Rust 1.100.0)

- `cargo check --offline -p gigaam-rs` and `cargo test --offline -p gigaam-rs --lib` succeed (5 unit tests).
- PyTorch-vs-Burn **synthetic** Conformer+CTC parity: max absolute difference of normalized logits `3.22e-6`; acoustic log-mel max difference `5.79e-4`.
- The supplied 89.94-second `MitaKind.wav` passes 44.1 kHz to 16 kHz resampling, four-way segmentation, and finite log-mel generation.
- Full pretrained ASR recognition is **not verified**: `v3_ctc.ckpt` was not provided. The toy checkpoint is not an ASR model.
- `--features cuda` targets the Burn CUDA backend but was not built or executed on Windows in this Linux environment. The optional CubeCL CPU backend attempts an LLVM download; the verified offline backend is `flex`.
