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

The shared Conformer supports rotary or relative positional attention, Conv1D or Conv2D strided subsampling factors 2/4/8/16/32 (powers of two), and convolution BatchNorm or LayerNorm. Factors 2 and 8 have been tested against the original PyTorch encoder on small random-weight configurations. Original PyTorch ZIP checkpoints containing `cfg` (OmegaConf DictConfig) or fine-tuning `hyper_parameters.model_cfg` now load their embedded configuration **natively in Rust** without `config.json`, when the architecture uses supported GigaAM operators. A `config.json` remains an explicit override and is still required for Safetensors-only files or unsupported checkpoint metadata. Arbitrary non-power-of-two subsampling, advanced emotion heads, GPU-vectorized batch execution, neural VAD, and all official-weight parity checks are **not yet implemented/verified**. Functional batch APIs run each recording independently, preserving output order without padding-induced changes. This is not a performance claim. v1/v2 and `emo` should be treated as experimental until checked with original checkpoint configurations. The source expects model-specific dimensions from the checkpoint. Model names are not interchangeable with weight files. Known model names are resolved through an exact registry; unknown names must supply an explicit `model_type`, and conflicting registered name/type combinations fail at load time instead of silently falling back to CTC.

## Configuration

For an original PyTorch ZIP checkpoint (`.ckpt`, `.pt`, `.pth`) with an embedded GigaAM `cfg`, you may pass the checkpoint path directly **without a sidecar**; or put exactly one checkpoint in a model directory. The native parser extracts OmegaConf scalar/list/dictionary nodes as data without executing Python pickle constructors. The model family is taken from `cfg.model_name`, fine-tuning `hyper_parameters.model_name`, or the checkpoint filename. Unknown families fail explicitly. To override the checkpoint configuration or load Safetensors, place a checkpoint (`<model_name>.ckpt`, `<model_name>.safetensors`, `model.ckpt`, or `model.safetensors`) alongside a `config.json` manifest in a local model directory. `GigaAm::load()` also accepts an explicit `.ckpt`/`.pth`/`.pt`/`.safetensors` **file path** with a sibling `config.json`. The five upstream short names (`ctc`, `rnnt`, `e2e_ctc`, `e2e_rnnt`, `ssl`) resolve to `v3_*` weights/tokenizer filenames, matching Python's `load_model()` aliases. For backwards compatibility, CTC also accepts `v3_ctc.ckpt` and `v3_ctc.safetensors`. Original PyTorch `state_dict` parameter names must be preserved, including the `encoder.` and `head.` prefixes.

The required dimensions are `n_mels`, `n_layers`, `d_model`, `n_heads`, `ff_expansion`, `conv_kernel`, `subsample_kernel`, `subsample_channels`, and `num_classes`. Optional architecture settings are `self_attention_model` (`rotary` / `rel_pos`), `subsampling` (`conv1d` / `conv2d`), `subsampling_factor` (power of two, 2–32), `conv_norm_type` (`batch_norm` / `layer_norm`), and `pos_emb_max_len` (defaults to **5000**, as in the original Python encoder). Note that the Python `RotaryPositionalEmbedding` also uses `pos_emb_max_len` as the **RoPE frequency base**: setting it to the common Transformer value 10000 silently changes GigaAM's output. The defaults describe the v3-style architecture, **not** a reliable description of every checkpoint. They **must match the selected checkpoint**; the defaults of upstream Python classes are not a verified GigaAM v3 checkpoint configuration.

The frontend follows the exact checkpoint STFT configuration. For official GigaAM v3 CTC the embedded OmegaConf specifies `n_fft=320`, `win_length=320`, `hop_length=160`, `center=false`, 64 HTK mel bands and no mel normalization. The mel calculation uses an uncentered STFT (no reflect padding), so a 16 kHz one-second input produces 99 mel frames, not 101. The model manifest optionally includes a `mel` object with `n_fft`, `win_length`, `hop_length`, and `center`; omitted values retain the older 400/400/160 centered defaults for legacy `config.json` and synthetic fixtures. Both modes have Torchaudio parity fixtures, including the case `win_length < n_fft`. Unsupported preprocessor normalization, mel scale, nonzero padding and power settings are still rejected explicitly.

For RNN-T, add the `rnnt` object with `pred_hidden`, `pred_rnn_layers`, `joint_hidden`, and optional `max_symbols_per_step` (defaults to 10, as upstream). Token ID `num_classes-1` is reserved for blank. RNN-T states are committed only after a nonblank emission, matching the original `RNNTGreedyDecoding` algorithm. The predictor handles single-sample inference.

For `emo`, use `model_type: "emo"`, `id2name` as an ordered string array or a map of consecutive numeric string IDs, and a linear classifier `head.weight` / `head.bias` (or `head.0.weight` / `head.0.bias`). `GigaAm::emotion_probs_16k()` returns named class probabilities. Emotion inference accepts long recordings, matching the upstream `get_probs` behavior; `transcribe()` alone retains the 25-second input guard.

For non-E2E ASR use `vocabulary` as an ordered array of characters (length `num_classes-1`). For E2E, put the original `<model_name>_tokenizer.model` file beside the weights, or set `sentencepiece_model` to another basename in `config.json`. The included pure-Rust Protobuf decoder reads SentencePiece vocabulary IDs and decodes ordinary, unknown, control, and byte-fallback tokens. Training tokenization is not needed for ASR output decoding. SentencePiece byte fallback follows the reference decoder’s one-replacement-per-invalid-byte behavior (rather than Rust `from_utf8_lossy` coalescing some malformed UTF-8 sequences).

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

## Verified official Windows/CUDA checkpoint

A user-supplied Windows x64 test on an RTX 4060 (2026-11-17) confirmed that the native **Burn/CubeCL CUDA** inference path runs the official `v3_ctc.ckpt` (220,718,498 parameters) end-to-end on an 89.939-second Russian WAV. The test suite reported **10/10 PASS** and 6.758 seconds elapsed for `real-model-asr`, including model loading and audio processing. The model produced four nonempty transcription segments. The original CTC vocabulary contains no punctuation; the E2E family (`v3_e2e_ctc`, `v3_e2e_rnnt`) supports punctuation and text normalization but still requires independent official-checkpoint testing. Windows PowerShell 5.1 originally mangled Cyrillic when capturing native stdout; the updated tester captures both streams with explicit UTF-8. The observed result is not a verified WER metric or a kernel-only benchmark.

## CLI

```text
cargo run --offline -p gigaam-rs --no-default-features --features cpu -- transcribe ./weights ./short.wav --device cpu
cargo run --offline -p gigaam-rs --no-default-features --features cpu -- transcribe ./weights ./MitaKind.wav --device cpu
cargo run --offline -p gigaam-rs -- embed ./ssl-weights ./short.wav ./embeddings.f32
cargo run --offline -p gigaam-rs -- parity ./ctc-weights ./short.wav ./parity-output
cargo run --offline -p gigaam-rs -- probe-audio ./MitaKind.wav
cargo run --offline -p gigaam-rs -- inspect-config ./v3_ctc.ckpt --raw
cargo run --offline -p gigaam-rs -- transcribe-batch ./weights ./audio_paths.json
cargo run --offline -p gigaam-rs -- transcribe ./weights ./short.wav --word-timestamps
cargo run --offline -p gigaam-rs -- emotion ./emo-weights ./short.wav
cargo run --offline -p gigaam-rs -- parity-encoder ./ssl-weights ./short.wav ./encoder-parity
```

### Checkpoint inspection (pure Rust)

Use `gigaam-rs inspect-checkpoint ./v3_ctc.ckpt` to examine parameter shapes and likely encoder/head layers without running PyTorch or allocating converted FP32 weights; `--all` prints every tensor. This also accepts `.safetensors`. It is useful for auditing a newly downloaded model. Use `inspect-config <model.ckpt>` for native OmegaConf parsing and a GigaAM inference manifest, or `inspect-config <model.ckpt> --raw` for the original Hydra configuration. Unsupported settings produce errors; real-weight parity remains required.

### Batch and audio input

`GigaAm::transcribe_batch_16k`, `transcribe_batch_with_timestamps_16k`, `infer_embeddings_batch_16k`, and `emotion_probs_batch_16k` accept slices of individual 16-kHz PCM recordings and produce ordered results. Each recording is executed independently (not a single padded GPU batch); failures are annotated with the input index. `transcribe-batch <model_dir> <audio_paths.json>` reads a JSON array of paths and prints per-file JSON transcriptions.

The CLI's `load_audio_16k` accepts PCM WAV, FLAC, and Ogg/Vorbis via pure Rust decoders; unsupported formats such as MP3/AAC still need conversion upstream. The WAV path remains the original `hound` decoder for parity. The Symphonia codec dependency uses MPL-2.0; see `THIRD_PARTY_NOTICES.md`.

### External VAD / NeuroMita integration

The library does **not** instantiate a second VAD. Pass detected speech intervals from the application's VAD as half-open sample ranges via `audio::SpeechRegion { start_sample, end_sample }` and call `GigaAm::transcribe_regions_16k` or `GigaAm::transcribe_regions_with_timestamps_16k`. Regions must be ordered, nonoverlapping, nonempty and in range; longer than 24 seconds are subdivided automatically. Gaps are preserved in the returned absolute timestamps. An empty region list produces an empty transcription list, rather than unexpectedly transcribing silence.

The CLI additionally supports `--segments-json vad.json --group-vad-python`, which applies the Python `segment_audio_file` interval grouping policy to already-detected speech intervals (without loading its VAD model). The original ungrouped `--segments-json` behavior is retained. A permanent fixture contains 160 VAD grouping examples generated by the upstream Python implementation, including floating-point threshold boundaries. Grouped chunks may exceed the short-form 25-second `transcribe()` contract, as in the Python longform path.

The CLI accepts `--segments-json vad.json` (only with `transcribe`), where `vad.json` is an array of start/end times **in seconds**, not sample positions:

```json
[{"start": 0.4, "end": 2.3}, {"start": 5.1, "end": 7.8}]
```

Example: `gigaam-rs transcribe ./weights ./MitaKind.wav --segments-json vad.json --word-timestamps`. Fractional timestamps are rounded to the nearest 16 kHz sample; invalid, nonfinite, out-of-order and overlapping intervals are rejected. Without this file, longform retains the low-energy heuristic (not neural VAD). The library performs no online VAD or streaming audio capture.

`transcribe` automatically splits long WAV recordings into <=24 second segments near low-energy boundaries; this is a heuristic, **not** a VAD model. The SSL `embed` command writes row-major `[time, hidden]` float32 and prints the shape. The diagnostic `parity` command writes `mel.f32` and *unnormalized* CTC `logits.f32` for reference comparison. `parity-encoder` writes `mel.f32` and row-major `encoder.f32` for SSL checkpoint parity. The `--word-timestamps` flag emits approximate word intervals using GigaAM's original encoder-frame-to-audio-duration mapping; the intervals are not forced-alignment boundaries, and longform chunk splitting remains energy-based rather than neural VAD.

The default inference backend is **Burn CUDA powered by CubeCL** (`cargo build -p gigaam-rs --release`). CUDA is the production backend on systems with supported NVIDIA GPUs; no Flex backend is selected implicitly. For an offline **CubeCL CPU (LLVM JIT)** verification build use `cargo test --offline -p gigaam-rs --no-default-features --features cpu` and `--device cpu`. The pure-Rust Flex fallback is opt-in only: `cargo test --offline -p gigaam-rs --no-default-features --features flex`, then `--device flex`. WGPU is opt-in via `--no-default-features --features wgpu`. Combining `cpu` with `cuda` supports optional CPU parity tests in a CUDA build; `--device cpu` always means CubeCL CPU, never Flex. If `--device` is omitted the CLI selects CUDA when compiled, otherwise CubeCL CPU, then WGPU, then Flex, in that order. The preprocessing, weights, Conformer, LSTM, and decoders use one backend-agnostic Burn tensor implementation; there are no Flex-only model paths. Actual NVIDIA GPU correctness and performance still require a native GPU test. No Luminal integration is required.

## Verification (Rust 1.100.0, offline CubeCL CPU + opt-in Flex)

- `cargo check --offline -p gigaam-rs --no-default-features --features cpu` — PASS with Tracel LLVM 23.1.0 installed from the offline `linux-x64.tar.xz` bundle.
- `cargo test --offline -p gigaam-rs --no-default-features --features cpu --tests` — **34/34 unit + 5/5 integration PASS**. The CPU backend is CubeCL LLVM JIT, not Flex.
- `cargo test --offline -p gigaam-rs --no-default-features --features flex --tests` — **34/34 unit + 5/5 integration PASS**, with Flex explicitly enabled.
- Tiny deterministic synthetic Safetensors checkpoints under `tests/fixtures/backend/` run **CTC, RNN-T, SSL and Emo** end-to-end on CubeCL CPU and Flex with one shared Rust implementation. These fixtures do not contain pretrained weights and cannot establish ASR quality.
- On the same synthetic CTC checkpoint and 16 kHz audio, the CubeCL CPU and Flex logit outputs differ by at most **1.1920929e-7** (44 float32 elements); log-mel features match bit-for-bit.
- VAD grouping: **160/160 upstream Python reference interval cases PASS** with `--group-vad-python`, including exact-duration float-threshold cases.

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

### Native checkpoint configuration (stage 7)

- Native read-only PyTorch ZIP `data.pkl` / OmegaConf extraction (protocols 2, 3, 4 and 5) works with original `cfg` and fine-tuning `hyper_parameters.model_cfg`; no Python runtime, executable pickle callbacks or sidecar conversion needed.
- The loader checks the FeatureExtractor contract (16 kHz / n_fft 400 / hop 160 / window 400 / centered HTK mel) before accepting an embedded config. Incompatible or unknown architecture settings must be handled explicitly rather than guessed.
- Synthetic SSL Conformer checkpoint: embedded OmegaConf vs explicit `config.json` produced **bit-identical** 14×8 embeddings on Rust Flex CPU.
- Synthetic v2 CTC, Emo and fine-tuned CTC checkpoints: automatic config and explicit sidecar runs produced identical Rust CLI transcriptions / emotion probabilities.
- An in-repository small synthetic OmegaConf checkpoint is used as a unit-test fixture. Additional malformed-pickle tests ensure failure without panicking. **These are not official GigaAM weights.**


### Python-reference inference preprocessing and text parity (stage 8)

- `tests/mel_parity.rs` compares the native 64-bin 16-kHz log-mel extraction with Torchaudio 2.10 CPU on identical synthetic PCM (`n_fft=400`, `hop=160`, `win=400`, HTK). Maximum absolute error **0.001114**, mean **0.0000112** in log-mel units; both tests run entirely on Rust without requiring PyTorch at test time. This numerical bound is for the supplied synthetic input, not arbitrary audio.
- `tests/sentencepiece_parity.rs` verifies **308/308** decoded token sequences against SentencePiece 0.2.1 generated output, including malformed UTF-8 byte-fallback sequences. The source parser previously differed on **44/308** of these inputs. The test model and expected outputs were generated from synthetic text, are committed as fixtures, and require no Python/SentencePiece installation at test time.
- `text::normalize_raw_text` ports the original GigaAM Unicode/whitespace/punctuation normalization function. All **35** reference cases from upstream `tests/test_normalize.py` pass on Rust.
- These tests cover the **preprocessor and text decoder**, not the full end-to-end ASR accuracy of any official GigaAM weights. The user-supplied 89.939-second `MitaKind.wav` is accepted by `probe-audio` at 16 kHz and split into four segments; there is still no recognized transcript without a genuine checkpoint.

Open requirements: compare every official checkpoint and its embedded architecture against this reader; confirm GPU / Windows execution separately. No claim of official model parity, real-audio WER or production readiness is made by these synthetic fixtures.

### Source-level parity hardening (stage 9)

- Word timestamps now follow the original `gigaam.timestamps_utils.frames_to_words` boundary semantics, including an empty SentencePiece `▁` token counting toward the word's first encoder frame. `tests/timestamps_parity.rs` checks 128 deterministic cases generated by executing that function from the upstream Python source; the fixture is self-contained and runs without Python at test time.
- Word timestamps from longform or external-VAD regions are rounded to milliseconds after adding the original sample offset, following upstream `transcribe_longform`. Segment boundaries themselves preserve sample-level precision.
- The native log-mel extractor rejects PCM of 200 samples or fewer, matching Torchaudio's reflect-padding requirement for `n_fft=400`, as well as nonfinite PCM input.
- The 25-second guard remains on **short-form transcription**. It no longer restricts SSL embeddings, emotion classification or diagnostic CTC logits: upstream `GigaAM.embed_audio` and `GigaAMEmo.get_probs` have no short-form guard. These unrestricted forwards can consume substantial CPU/GPU memory on long audio; chunking is still recommended.
- This update adds source-level/fixture parity only. Official model weights, CUDA execution and end-to-end ASR accuracy have **not** been validated.
