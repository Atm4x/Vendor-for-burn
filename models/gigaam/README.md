# Native GigaAM ASR / SSL inference (Rust / Burn 0.22.1)

This crate implements the GigaAM Conformer family in Rust using Burn tensors. No ONNX Runtime, Candle, or Python is needed **at inference time**.

## Implemented variants

| Model family | Inference | Decoder / output | Verification |
|---|---|---|---|
| `v3_ctc` | Conformer + CTC | Character greedy | Synthetic checkpoint and PyTorch parity |
| `v3_rnnt` | Conformer + LSTM predictor + joint | RNN-T greedy | Synthetic blank/nonblank parity with PyTorch |
| `v3_e2e_ctc` | Conformer + CTC | SentencePiece ModelProto | Synthetic checkpoint and SentencePiece decoding |
| `v3_e2e_rnnt` | Conformer + LSTM predictor + joint | SentencePiece ModelProto | Synthetic checkpoint and decoding |
| `v3_ssl` | Conformer encoder | Float32 frame embeddings | Synthetic PyTorch parity |
| `v1_ctc`, `v2_ctc` (factor-4 Conformer) | Conv1D/2D + rotary/relative attention | Character greedy + optional word timestamps | Synthetic full-encoder parity for rel_pos/Conv1D |
| `v1_rnnt`, `v2_rnnt` (factor-4 Conformer) | Shared Conformer + RNN-T | Greedy + optional timestamps | Building blocks only; official model not checked |
| `emo` (linear classifier head) | Conformer + mean pooling + linear head | Named emotion probabilities | Synthetic end-to-end checkpoint and PyTorch parity |
| `multilingual_ctc`, `multilingual_large_ctc` | Same Conformer/CTC building blocks | Configured character vocabulary | No official checkpoint tested |
| `multilingual_ssl`, `multilingual_large_ssl` | Same Conformer encoder | Frame embeddings | No official checkpoint tested |

The shared Conformer supports rotary or relative positional attention, Conv1D or Conv2D factor-4 subsampling, and convolution BatchNorm or LayerNorm. Original checkpoint `cfg` fields must be transferred into `config.json` accurately. Other subsampling factors, advanced emotion heads, batch inference, neural VAD, automatic checkpoint-config extraction, and all official-weight parity checks are **not yet implemented/verified**. v1/v2 and `emo` should be treated as experimental until checked with original checkpoint configurations. The source expects model-specific dimensions from the checkpoint. Model names are not interchangeable with weight files.

## Configuration

Place a checkpoint (`<model_name>.ckpt`, `<model_name>.safetensors`, `model.ckpt`, or `model.safetensors`) alongside a `config.json` manifest in a local model directory. For backwards compatibility, CTC also accepts `v3_ctc.ckpt` and `v3_ctc.safetensors`. Original PyTorch `state_dict` parameter names must be preserved, including the `encoder.` and `head.` prefixes.

The required dimensions are `n_mels`, `n_layers`, `d_model`, `n_heads`, `ff_expansion`, `conv_kernel`, `subsample_kernel`, `subsample_channels`, and `num_classes`. Optional architecture settings are `self_attention_model` (`rotary` / `rel_pos`), `subsampling` (`conv1d` / `conv2d`), `subsampling_factor` (currently 4), and `conv_norm_type` (`batch_norm` / `layer_norm`). The defaults describe the v3-style architecture, **not** a reliable description of every checkpoint. They **must match the selected checkpoint**; the defaults of upstream Python classes are not a verified GigaAM v3 checkpoint configuration.

For RNN-T, add the `rnnt` object with `pred_hidden`, `pred_rnn_layers`, `joint_hidden`, and optional `max_symbols_per_step` (defaults to 10, as upstream). Token ID `num_classes-1` is reserved for blank. RNN-T states are committed only after a nonblank emission, matching the original `RNNTGreedyDecoding` algorithm. The predictor handles single-sample inference.

For `emo`, use `model_type: "emo"`, `id2name` as an ordered string array or a map of consecutive numeric string IDs, and a linear classifier `head.weight` / `head.bias` (or `head.0.weight` / `head.0.bias`). `GigaAm::emotion_probs_16k()` returns named class probabilities. Only single-recording inputs of at most 25 seconds are supported by the emotion API.

For non-E2E ASR use `vocabulary` as an ordered array of characters (length `num_classes-1`). For E2E, put the original `<model_name>_tokenizer.model` file beside the weights, or set `sentencepiece_model` to another basename in `config.json`. The included pure-Rust Protobuf decoder reads SentencePiece vocabulary IDs and decodes ordinary, unknown, control, and byte-fallback tokens. Training tokenization is not needed for ASR output decoding.

Example RNN-T manifest (illustrative fields; replace all dimensions with the checkpoint's exact configuration):

```json
{
  "model_name": "v3_rnnt",
  "n_mels": 64,
  "n_layers": 16,
  "d_model": 768,
  "n_heads": 16,
  "ff_expansion": 4,
  "conv_kernel": 31,
  "subsample_kernel": 3,
  "subsample_channels": 768,
  "num_classes": 34,
  "vocabulary": ["...must supply all 33 class strings..."],
  "rnnt": {"pred_hidden": 768, "pred_rnn_layers": 2, "joint_hidden": 768}
}
```

**Do not use this illustrative example unchanged**: no official checkpoint configuration has been inferred from the model name, and the abbreviated vocabulary is intentionally invalid.

## CLI

```text
cargo run --offline -p gigaam-rs -- transcribe ./weights ./short.wav --device cpu
cargo run --offline -p gigaam-rs -- transcribe ./weights ./MitaKind.wav --device cpu
cargo run --offline -p gigaam-rs -- embed ./ssl-weights ./short.wav ./embeddings.f32
cargo run --offline -p gigaam-rs -- parity ./ctc-weights ./short.wav ./parity-output
cargo run --offline -p gigaam-rs -- probe-audio ./MitaKind.wav
```

`transcribe` automatically splits long WAV recordings into <=24 second segments near low-energy boundaries; this is a heuristic, **not** a VAD model. The SSL `embed` command writes row-major `[time, hidden]` float32 and prints the shape. The diagnostic `parity` command writes `mel.f32` and *unnormalized* CTC `logits.f32` for reference comparison. `parity-encoder` writes `mel.f32` and row-major `encoder.f32` for SSL checkpoint parity. The `--word-timestamps` flag emits approximate word intervals using GigaAM's original encoder-frame-to-audio-duration mapping; the intervals are not forced-alignment boundaries, and longform chunk splitting remains energy-based rather than neural VAD.

Default build uses Burn Flex CPU, with CUDA (`--no-default-features --features cuda`) and WGPU (`--no-default-features --features wgpu`) selected at compile time. Windows CUDA still requires a native build and GPU validation. No Luminal AOT integration is implied by support for Burn CUDA.

## Verification (Rust 1.100.0, Linux Flex CPU)

- `cargo check --offline -p gigaam-rs` — PASS.
- `cargo test --offline -p gigaam-rs --lib` — **9/9 PASS**.
- CTC logits from previous implementation are byte-identical after sharing Conformer encoder with RNN-T/SSL.
- Python/PyTorch synthetic Conformer+CTC log-probability parity: maximum absolute difference **3.22e-6** (previous stage).
- RNN-T synthetic blank-only case and two-symbols-per-frame case match PyTorch text and token counts; standalone two-step LSTM matches PyTorch to **1e-6**.
- SSL Conformer output on 1-second synthetic WAV matches PyTorch within **9.32e-5** max absolute error.
- E2E CTC and RNN-T synthetic checkpoint plus a locally trained SentencePiece ModelProto decode successfully.
- Synthetic v2-style Conformer with `rel_pos` attention, `conv1d` subsampling, `layer_norm`: PyTorch forward parity on 51×8 output: **max 5.96e-7, mean 1.42e-7**.
- Synthetic v3-style Conformer with `rotary`, `conv2d`, `batch_norm`: PyTorch forward sanity parity on 51×8 output: **max 1.26e-3, mean 1.97e-4** (requires further accuracy profiling).
- Synthetic legacy-style emotion checkpoint (`head.weight`, `head.bias`): Rust `0.544427 / 0.180733 / 0.274840`, matched PyTorch within the printed precision.
- Synthetic CTC with `rel_pos`/Conv1D: CLI `--word-timestamps` produced an aligned word interval for a 2-second input.
- Supplied 89.94-second `MitaKind.wav`: resampling 44.1 kHz → 16 kHz, four-way segmentation, finite mel features. **No official pretrained GigaAM ASR checkpoint was available**, so real speech recognition is not validated.

## Licensing

This implementation is derived from the GigaAM architecture and code (MIT; Copyright (c) 2024 GigaChat Team); see the retained `LICENSE`. Check the license of any downloaded weights separately. Neither the checkpoint nor the user's audio is included in the crate.
