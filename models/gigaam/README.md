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

The shared Conformer supports rotary or relative positional attention, Conv1D or Conv2D strided subsampling factors 2/4/8/16/32 (powers of two), and convolution BatchNorm or LayerNorm. Factors 2 and 8 have been tested against the original PyTorch encoder on small random-weight configurations. Original checkpoint `cfg` fields must be transferred into `config.json` accurately. Arbitrary non-power-of-two subsampling, advanced emotion heads, GPU-vectorized batch execution, neural VAD, automatic checkpoint-config extraction, and all official-weight parity checks are **not yet implemented/verified**. Functional batch APIs run each recording independently, preserving output order without padding-induced changes. This is not a performance claim. v1/v2 and `emo` should be treated as experimental until checked with original checkpoint configurations. The source expects model-specific dimensions from the checkpoint. Model names are not interchangeable with weight files. Known model names are resolved through an exact registry; unknown names must supply an explicit `model_type`, and conflicting registered name/type combinations fail at load time instead of silently falling back to CTC.

## Configuration

Place a checkpoint (`<model_name>.ckpt`, `<model_name>.safetensors`, `model.ckpt`, or `model.safetensors`) alongside a `config.json` manifest in a local model directory. `GigaAm::load()` also accepts an explicit `.ckpt`/`.pth`/`.pt`/`.safetensors` **file path** with a sibling `config.json`. The five upstream short names (`ctc`, `rnnt`, `e2e_ctc`, `e2e_rnnt`, `ssl`) resolve to `v3_*` weights/tokenizer filenames, matching Python's `load_model()` aliases. For backwards compatibility, CTC also accepts `v3_ctc.ckpt` and `v3_ctc.safetensors`. Original PyTorch `state_dict` parameter names must be preserved, including the `encoder.` and `head.` prefixes.

The required dimensions are `n_mels`, `n_layers`, `d_model`, `n_heads`, `ff_expansion`, `conv_kernel`, `subsample_kernel`, `subsample_channels`, and `num_classes`. Optional architecture settings are `self_attention_model` (`rotary` / `rel_pos`), `subsampling` (`conv1d` / `conv2d`), `subsampling_factor` (currently 4), `conv_norm_type` (`batch_norm` / `layer_norm`), and `pos_emb_max_len` (defaults to **5000**, as in the original Python encoder). Note that the Python `RotaryPositionalEmbedding` also uses `pos_emb_max_len` as the **RoPE frequency base**: setting it to the common Transformer value 10000 silently changes GigaAM's output. The defaults describe the v3-style architecture, **not** a reliable description of every checkpoint. They **must match the selected checkpoint**; the defaults of upstream Python classes are not a verified GigaAM v3 checkpoint configuration.

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
cargo run --offline -p gigaam-rs -- transcribe-batch ./weights ./audio_paths.json
cargo run --offline -p gigaam-rs -- transcribe ./weights ./short.wav --word-timestamps
cargo run --offline -p gigaam-rs -- emotion ./emo-weights ./short.wav
cargo run --offline -p gigaam-rs -- parity-encoder ./ssl-weights ./short.wav ./encoder-parity
```

### Checkpoint inspection (pure Rust)

Use `gigaam-rs inspect-checkpoint ./v3_ctc.ckpt` to examine parameter shapes and likely encoder/head layers without running PyTorch or allocating converted FP32 weights; `--all` prints every tensor. This also accepts `.safetensors`. It is useful for auditing a newly downloaded model. The official checkpoint `cfg` is serialized as Python OmegaConf objects, so the metadata is **not automatically converted into `config.json`**: tensor shapes alone cannot establish attention heads, positional encoding policy, normalization or tokenizer ordering. Do not infer these silently from the filename. Real-weight parity remains required.

### Batch and audio input

`GigaAm::transcribe_batch_16k`, `transcribe_batch_with_timestamps_16k`, `infer_embeddings_batch_16k`, and `emotion_probs_batch_16k` accept slices of individual 16-kHz PCM recordings and produce ordered results. Each recording is executed independently (not a single padded GPU batch); failures are annotated with the input index. `transcribe-batch <model_dir> <audio_paths.json>` reads a JSON array of paths and prints per-file JSON transcriptions.

The CLI's `load_audio_16k` accepts PCM WAV, FLAC, and Ogg/Vorbis via pure Rust decoders; unsupported formats such as MP3/AAC still need conversion upstream. The WAV path remains the original `hound` decoder for parity. The Symphonia codec dependency uses MPL-2.0; see `THIRD_PARTY_NOTICES.md`.

### External VAD / NeuroMita integration

The library does **not** instantiate a second VAD. Pass detected speech intervals from the application's VAD as half-open sample ranges via `audio::SpeechRegion { start_sample, end_sample }` and call `GigaAm::transcribe_regions_16k` or `GigaAm::transcribe_regions_with_timestamps_16k`. Regions must be ordered, nonoverlapping, nonempty and in range; longer than 24 seconds are subdivided automatically. Gaps are preserved in the returned absolute timestamps. An empty region list produces an empty transcription list, rather than unexpectedly transcribing silence.

The CLI accepts `--segments-json vad.json` (only with `transcribe`), where `vad.json` is an array of start/end times **in seconds**, not sample positions:

```json
[{"start": 0.4, "end": 2.3}, {"start": 5.1, "end": 7.8}]
```

Example: `gigaam-rs transcribe ./weights ./MitaKind.wav --segments-json vad.json --word-timestamps`. Fractional timestamps are rounded to the nearest 16 kHz sample; invalid, nonfinite, out-of-order and overlapping intervals are rejected. Without this file, longform retains the low-energy heuristic (not neural VAD). The library performs no online VAD or streaming audio capture.

`transcribe` automatically splits long WAV recordings into <=24 second segments near low-energy boundaries; this is a heuristic, **not** a VAD model. The SSL `embed` command writes row-major `[time, hidden]` float32 and prints the shape. The diagnostic `parity` command writes `mel.f32` and *unnormalized* CTC `logits.f32` for reference comparison. `parity-encoder` writes `mel.f32` and row-major `encoder.f32` for SSL checkpoint parity. The `--word-timestamps` flag emits approximate word intervals using GigaAM's original encoder-frame-to-audio-duration mapping; the intervals are not forced-alignment boundaries, and longform chunk splitting remains energy-based rather than neural VAD.

Default build uses Burn Flex CPU, with CUDA (`--no-default-features --features cuda`) and WGPU (`--no-default-features --features wgpu`) selected at compile time. Windows CUDA still requires a native build and GPU validation. No Luminal AOT integration is implied by support for Burn CUDA.

## Verification (Rust 1.100.0, Linux Flex CPU)

- `cargo check --offline -p gigaam-rs` — PASS.
- `cargo test --offline -p gigaam-rs --lib` — **22/22 PASS** (current test suite).
- CTC logits from previous implementation are byte-identical after sharing Conformer encoder with RNN-T/SSL.
- Python/PyTorch synthetic Conformer+CTC log-probability parity: maximum absolute difference **3.22e-6** (previous stage).
- RNN-T synthetic blank-only case and two-symbols-per-frame case match PyTorch text and token counts; standalone two-step LSTM matches PyTorch to **1e-6**.
- SSL Conformer output on 1-second synthetic WAV matches PyTorch within **9.32e-5** max absolute error.
- E2E CTC and RNN-T synthetic checkpoint plus a locally trained SentencePiece ModelProto decode successfully.
- Synthetic v2-style Conformer with `rel_pos` attention, `conv1d` subsampling, `layer_norm`: PyTorch forward parity on 51×8 output: **max 5.96e-7, mean 1.42e-7**.
- Synthetic v3-style Conformer with `rotary`, `conv2d`, `batch_norm`: after correcting the original `pos_emb_max_len=5000` RoPE frequency base, **max 4.77e-7, mean 1.01e-7** against PyTorch on the 51×8 output. Prior `1.26e-3` discrepancy was caused by using a hard-coded base of 10000.
- Synthetic legacy-style emotion checkpoint (`head.weight`, `head.bias`): Rust `0.544427 / 0.180733 / 0.274840`, matched PyTorch within the printed precision.
- Synthetic CTC with `rel_pos`/Conv1D: CLI `--word-timestamps` produced an aligned word interval for a 2-second input.
- PyTorch ZIP checkpoint (`.ckpt` containing `state_dict` and `cfg`) loaded through `burn-store`: output **bit-identical** to the same weights in `.safetensors` on a 51×8 synthetic Conformer.
- Explicit exact-name model registry rejects unknown names and conflicting architecture kinds.
- Supplied 89.94-second `MitaKind.wav`: resampling 44.1 kHz → 16 kHz, four-way segmentation, finite mel features. **No official pretrained GigaAM ASR checkpoint was available**, so real speech recognition is not validated.

## Licensing

This implementation is derived from the GigaAM architecture and code (MIT; Copyright (c) 2024 GigaChat Team); see the retained `LICENSE`. Check the license of any downloaded weights separately. Neither the checkpoint nor the user's audio is included in the crate.

### Additional parity verification (small synthetic weights, Rust Flex CPU)

- Python short aliases resolve to the correct version-3 checkpoint basename.
- Loading an explicit `.ckpt` versus the containing model directory produced byte-identical SSL encoder output.
- `transcribe-batch` produced ordered results for two local recordings.
- WAV, FLAC and Ogg/Vorbis decoded and produced the same CTC text on a local synthetic checkpoint.
- Dynamic `subsampling_factor=2` with Conv2D + rotary: output `[101,8]`, max absolute difference **4.77e-7** vs original PyTorch encoder.
- Dynamic `subsampling_factor=8` with Conv1D + relative attention: output `[26,8]`, max absolute difference **5.96e-7**.

These do **not** establish official-checkpoint parity, GPU timing, or support for the original Python training/ONNX-export API.
