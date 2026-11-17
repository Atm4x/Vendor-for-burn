# GigaAM Python → Rust: inference parity contract

Reference: user-provided `GigaAM-main.zip`, source files under `gigaam/`. This is a **source-to-source audit**, not proof of numerical parity for official checkpoints. No Python VAD model is included in the Rust inference service; callers supply speech intervals.

| Python reference | Rust implementation | Verification and remaining difference |
| --- | --- | --- |
| `gigaam/__init__.py::load_model` (five v3 aliases, official and fine-tuned `.ckpt`, download/cache, hash, fp16/flash policy) | `GigaAm::load(path, device)` and native checkpoint metadata | Checkpoint configurations work for supplied v3 checkpoints. No automatic downloads, upstream hash cache, exact fp16 policy, or arbitrary fine-tuned checkpoint fallback when metadata is absent. |
| `preprocess.py::load_audio` (FFmpeg to mono, 16 kHz, signed 16-bit PCM) | `audio::load_audio_16k` (WAV/FLAC/Vorbis, pure Rust; sinc resampling) | Supported formats decode. Byte-level and floating-point parity with FFmpeg's s16le conversion/resampler is **not verified**; MP3, AAC and additional formats require separate implementation. |
| `preprocess.py::FeatureExtractor` / `SpecScaler` | `audio::log_mel_with_config` | Torchaudio reference fixtures test normal and v3 uncentered STFT geometries. Official checkpoint frontend numerical parity on real audio remains to be quantified. |
| `model.py::GigaAMASR.transcribe` (short audio <=25 s, optional word timestamps, no VAD) | `GigaAmService::transcribe_short_file/16k`, `Segmentation::FullRecording` (service default) | Implemented explicit short-form contract and length guard. `Segmentation::Energy` is retained **only as an opt-in legacy heuristic**, not Python parity. |
| `model.py::GigaAMASR.transcribe_longform` (VAD intervals, collated `DataLoader` batches, timestamp offsets) | `GigaAmService::transcribe_longform_regions_16k`, external `SpeechRegion`, Python-compatible reference grouping by default | Original interval grouping and timestamp fixtures exist. GPU **padded batched** encoder inference still missing; current processing is per segment, so throughput and numerical behavior of padded batches are not established. |
| `types.py::TranscriptionResult` / `LongformTranscriptionResult` | `ShortTranscriptionResult` / `TranscriptResult`, `words()` / `has_word_timestamps()` | Source-aligned text/words semantics for short and longform, plus optional transport metadata. |
| `model.py::GigaAM.embed_audio` | `GigaAmService::embed_file/16k`, `EmbeddingResult` | SSL encoder and shapes tested with synthetic weights. Official SSL weights, output tensors and lengths require PyTorch parity. |
| `model.py::GigaAMEmo.get_probs` (dictionary) | `GigaAmService::emotions_dict_file/16k` | Dict-shaped response, existing vector API retained. Official Emo logits/probabilities require PyTorch parity. |
| `decoding.py::CTCGreedyDecoding`, `RNNTGreedyDecoding`, SentencePiece | `decode.rs`, `rnnt.rs`, `tokenizer.rs` | Synthetic parity and 308 SentencePiece cases. Official CTC logits, RNNT token IDs, blank transitions and greedy emissions not yet numerically compared. |
| `timestamps_utils.py` (frame-to-word) | `decode::words_from_frames`, timed transcriptions | Deterministic upstream-reference cases passed; official model word boundaries still need comparison. |
| `model.py::to_onnx` / `forward_for_export`, `onnx_utils.py` | No equivalent Rust exporter | Not implemented. Python export is a distinct utility, not required to run native Burn inference. |
| `train_utils/` (training, evaluation, fine-tuning) | No training subsystem | Out of scope for inference-only port; separate work if training parity is explicitly required. |

## API and process semantics

- `GigaAmService` owns exactly one loaded model; repeated requests do not reload weights.
- `gigaam-rs serve` supports `transcribe` (strict short-form by default), `transcribe_longform` (required caller-owned `segments`), `transcribe_batch` (short-form array), `embed`, `emotion` (JSON dictionary), `status`, `shutdown`.
- Input segments are seconds in JSONL, mapped into validated half-open 16 kHz sample ranges. `segments: []` yields empty longform output.
- `transcribe_longform` applies the **original grouping policy to provided speech intervals** by default (`group_vad: false` opts out), and never executes VAD inference.
- Functional batch methods currently process clips independently. Do not call this equivalent to Python's padded GPU batch until measured and proven.
- The legacy `gigaam-rs transcribe` command still accepts long recordings via an energy-based fallback for backward compatibility. Use the typed service API or explicit `transcribe_longform` for source-faithful invocation.

## Gates before claiming production parity

1. On the same actual checkpoint and WAV, compare frontend mel tensors and encoder activations (shapes, absolute error, SNR) against original Python.
2. Compare CTC logits and decoded tokens/frames; for RNN-T compare token IDs, timestamps and predictor/JOIN emissions. Text-only PASS is insufficient.
3. Compare short/longform boundary and word timestamp results using **identical external VAD intervals**.
4. Compare single-item and padded batched encoder outputs for uneven lengths, at batch sizes 1/2/4/16.
5. Repeat warm-loop requests on the intended Windows GPU, recording latency and VRAM; Windows-hosted CI can compile but cannot attest to CUDA GPU execution.

## Official checkpoint diagnostic protocol

The inference engine now exports the **same pre-head encoder activations for CTC, RNN-T, SSL and Emo**, not just SSL. `gigaam-rs parity-encoder <model> <audio.wav> <rust-output-dir>` writes `mel.f32` and `encoder.f32` in the Python-compatible `[n_mels, time]` and `[time, d_model]` layouts. For CTC also run `gigaam-rs parity <model> <audio.wav> <rust-output-dir>` to write **pre-log-softmax** `logits.f32` in `[time, classes]`. Note: the Python CTC head's ordinary `forward()` yields *log probabilities*, whereas `head.decoder_layers` yields raw logits, so reference generation must compare the latter against Rust.

`tools/official_reference_dump.py` imports the supplied upstream Python package and loads original checkpoint weights. It writes the same `.f32` arrays plus `reference.json` with encoder/token/frame metadata. `tools/compare_reference.py` compares native outputs against the Python arrays, checks expected lengths and prints max/mean absolute error, RMSE and SNR. These tools are **reference diagnostics only**, never Python inference dependencies in the Rust service.

For reproducible audio, first convert both sides to **exactly the same mono 16-kHz PCM16 WAV** with FFmpeg, because original Python `load_audio` performs FFmpeg s16le conversion and native Rust decoding/resampling is not bit-identical for arbitrary inputs. Use speech snippets <=25 seconds for short-form ASR. Use the same trained checkpoint and tokenizer; Python `--upstream` must point to `GigaAM-main` with original Python requirements installed. Avoid treating Python CUDA autocast FP16 outputs as an FP32 oracle without documenting precision.

```text
python models/gigaam/tools/official_reference_dump.py --upstream PATH_TO_GigaAM-main --checkpoint v3_ctc.ckpt --audio audio16k.wav --output py_reference --device cpu

gigaam-rs parity-encoder v3_ctc.ckpt audio16k.wav rust_reference --device cuda
gigaam-rs parity v3_ctc.ckpt audio16k.wav rust_reference --device cuda
python models/gigaam/tools/compare_reference.py --python py_reference --rust rust_reference
```

No official checkpoint numerical parity measurement has been recorded in this environment, because the real weights are available on the user's Windows machine rather than in this local runtime.

## CONT3: closing source-mapped runtime gaps

The user-supplied Python source is the authority for these contracts:

- `encoder.py::StridingSubsampling._mask_time`, `ConformerEncoder.forward`, `ConformerConvolution.forward`: native `EncoderModel::encode_padded` now does genuine batched Conformer execution (not an independent loop). It applies per-sample input and per-layer subsampling masks, masked attention and depthwise-conv padding; outputs are trimmed by true subsampled length. CTC and RNN-T decoding remains per sample, as in the upstream `model.py` longform loop. Additional synthetic tests compare batch sizes 1, 2, 4 and 16 against single-item encoder output. **Official-weight CUDA tensor parity remains unverified.**
- `model.py::GigaAMASR.transcribe_longform`: the service uses external VAD intervals (no VAD engine), the previous Python grouping port, and batches groups of 16 by default (upstream `fr_batch_size`). `GigaAmService::with_batch_size(n)` and `gigaam-rs serve ... --batch-size n` can override this. Longform chunks are *not* subject to the short-form 25-second guard.
- `preprocess.py::load_audio`: `load_audio_ffmpeg_16k` reproduces the Python FFmpeg invocation, mono 16k PCM16 / 32768. `--python-audio` selects it for CLI or `serve` and requires FFmpeg on PATH. By default the native WAV/FLAC/Vorbis decoder remains unchanged; MP3/AAC-like files can use the FFmpeg fallback. MP3 decode was tested on synthetic audio, not official PyTorch accuracy.
- `__init__.py::_MODEL_HASHES`, `_download_model` and `_download_tokenizer`: `model_registry::resolve_model_path` implements the 16 registered names and v3 aliases, local cache resolution, original MD5 for named checkpoint artifacts, and E2E or v1_RNNT tokenizer lookup. Downloads are explicitly opt-in and compiled only with `model-download`; the model remains cached after resolution. `GigaAmService::load_named` and CLI model-name arguments are supported.

No legacy/obsolete scripts or VAD implementation were added. Decoder source and normalization are unchanged per user instruction.

The following **source APIs are not fully equivalent**, and must not be marked complete:

- `load_model(fp16_encoder=True, use_flash=...)`: native Burn uses F32 tensors; Python's optional half-precision encoder and Flash Attention flags are not faithfully reproduced. Native CUDA kernel selection must be assessed separately.
- Fine-tuned `.ckpt` fallback when the checkpoint has no self-sufficient embedded model configuration or tokenizer: only existing locally resolvable checkpoint configurations are supported; Python's base-model weight overlay requires a separate source-faithful implementation and fixture.
- `model.py::to_onnx`: ONNX graph export remains a separate deployment workflow; Burn's native inference runtime does not currently export the Python module graph. Do not claim ONNX parity.
- Real CUDA speed, memory and numerical parity of the *new* batching, PCM16 FFmpeg frontend and official model names must be checked on Windows with actual weights. The previous user's 14/14 PASS verified the old binary, not CONT3.

## CONT4 — Fine-tuned checkpoint fallback

Python `gigaam.__init__.load_model` first tries a fine-tuned checkpoint's
`hyper_parameters.model_cfg`. If it is absent, or `decoding.model_path` points
outside the checkpoint, Python resolves the registered `hyper_parameters.model_name`
and uses that base model's architecture and tokenizer, then loads the fine-tuned
`state_dict`. Rust now implements this resolution using `GigaAm::load_with_cache`
and `GigaAmService::load_named`; it does **not** replace fine-tuned weights with
base weights. The base checkpoint is used for configuration only. A local
checkpoint can resolve cached base weights offline; missing base artifacts
produce an explicit error (download remains opt-in).

The service's timestamped `transcribe_batch_16k` now also honors
`GigaAmService::with_batch_size`, matching the non-timestamped path's
allocation cap. No changes to CTC/RNN-T decoding, external VAD, or legacy code.

Scope boundary: Python's optional FP16 encoder / Flash Attention are numerical
and runtime optimization choices, not missing ASR operations. A native ONNX
exporter remains unimplemented; the Python export utility is not invoked by
native Burn inference.
