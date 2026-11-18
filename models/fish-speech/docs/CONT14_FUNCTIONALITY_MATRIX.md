# Fish Speech 1.5 Rust — functional parity, CONT14

**Source:** supplied `Fish-speech-pipeline-main(1).zip` (not Fish Speech S2).  This checklist counts behavior with callable Rust code independently of CUDA/Windows tests. `DONE=1`, `PARTIAL=0.5`, `MISSING=0`; every item gets the same bookkeeping weight, not the same implementation effort.

| ID | Python functionality | State | Rust implementation / still missing |
|---|---|---|---|
| F01 | Semantic/Firefly checkpoint config and `model_type` | PARTIAL | New `src/model_config.rs` parses DualAR/Naive JSON and original defaults. Firefly config and real checkpoint aliases still need runtime mapping. |
| F02 | Converted SafeTensors checkpoint loader | DONE | Manifest, SHA-256 and type/shape validation already present. Actual model weights not supplied. |
| F03 | Fish BPE tokenizer | DONE | Tiktoken parser, Fish regex, 1036 special tokens. Actual `tokenizer.tiktoken` absent. |
| F04 | `clean_text()` | DONE | Rust equivalent with Python-derived fixtures. |
| F05 | `split_text()` | DONE | UTF-8 byte and punctuation handling, 144 golden cases. |
| F06 | Conversation serialization/VQ parts | DONE | Rust `conversation.rs`. |
| F07 | `encode_tokens()` reference/non-reference prompts | DONE | Rust prompt planner. |
| F08 | `ReferenceLoader.load_audio()` | PARTIAL | **CONT14:** native WAV/FLAC/OGG from file/bytes, multi-channel downmix. MP3/AAC/M4A/WMA codecs missing from the offline vendor. |
| F09 | Mono/resampling/LogMel | DONE | **CONT14:** torchaudio-compatible native sinc-Hann resampler and direct Firefly LogMel boundary. Golden 16k/48k→44.1k passes. |
| F10 | Firefly reference ConvNeXt+FSQ encoder | PARTIAL | Structural TensorRT graph and CPU math; no unified weighted executable inference. |
| F11 | Slow AR prefill | PARTIAL | Model graph, attention, stream layout; no executable weighted prefill. |
| F12 | Slow AR cached decode | PARTIAL | Cached decode contracts; no full logits-producing runtime. |
| F13 | Fast AR acoustic codebook loop | PARTIAL | Steps and cache contracts exist; full weighted generation not connected. |
| F14 | `NaiveTransformer` | MISSING | Config type is recognized, but Naive forward/generate runtime absent. |
| F15 | Repetition penalty/top-p/Exp(1)/EOS | PARTIAL | CPU sampling math, but not bound to actual GPU AR generator. |
| F16 | `generate_long()` | PARTIAL | Rust chunk planner and context accounting; actual model generation missing. |
| F17 | Firefly FSQ decode + HiFiGAN | PARTIAL | Structural graph and CPU fixture ops; no weighted audio executor. |
| F18 | Concatenate PCM and 44.1k output | DONE | Rust `synthesize` concatenates sample segments. |
| F19 | End-to-end orchestration | PARTIAL | Testable trait pipeline; actual semantic generator and acoustic decoder missing. |
| F20 | Streaming `header/segment/final/error` | DONE | **CONT14:** callback-based `synthesize_with_events` emits events *during* synthesis; status and ordering tested against mock generator. |
| F21 | Reference cache by ID/hash | DONE | **CONT14:** SHA-256 audio token cache, `reference_id` and `.lab`, cache on/off and clear; intentional stale-text fix. |
| F22 | Runtime cancellation/lifecycle | PARTIAL | Segment-level cancellation exists; proper model switching, queue and GPU resource lifecycle still missing. |

**CONT14 accounting: 10 DONE + 11 PARTIAL + 1 MISSING = 22 items.** Score: `(10+11×0.5)/22 = 70.45%`, displayed as **~70%**. This is coverage bookkeeping, **not** proof of correct full weighted TTS.

### Independent operability / verification (not part of functional score)

| Gate | Status |
|---|---|
| V01 — Core tests | PASS: 171 unit + 27 integration = **198** |
| V02 — Outer Rust crate tests | PASS: **15** |
| V03 — Offline Python checkpoint exporter | PASS: **5** |
| V04 — Original-Python text golden | PASS: **144** cases (CONT13) |
| V05 — Rust→TensorRT 1.7 API compilation | PASS (`cargo check --locked --offline --features tensorrt-api`); no engine execution |
| V06 — TensorRT mock tests | Not rerun in CONT14; prior PASS does not establish current-run PASS |
| V07 — Authentic model checkpoint/tokenizer parity | BLOCKED: weights and authentic tokenizer absent |
| V08 — Real TensorRT engine build/execution | NOT TESTED — no SDK/GPU |
| V09 — Authentic end-to-end speech | NOT TESTED |
| V10 — Windows packaging/profiling | NOT TESTED |
| V11 — Reference audio contracts | PASS: WAV/FLAC/OGG CPU smoke, torchaudio resampling goldens, LogMel bridge |

Verification gate count: **6/11 PASS**, **0/3 authentic GPU/e2e/Windows acceptance gates**. This percentage must not be used as the software's functional readiness.

### Next priority

1. Obtain offline `symphonia-bundle-mp3 0.6.1` and associated vendor dependencies (plus AAC/MP4 codecs if supporting these formats), then complete F08.
2. Implement runnable weighted Slow/Fast AR with device buffers, QKV, KV writes and per-step codebooks, preserving exact `sample()`/EOS semantics (F11–F13, F15–F16).
3. Implement runnable Firefly encode/decode and bind them to the synthesis traits (F10, F17, F19).
4. Implement Naive model path (F14) and full model lifecycle (F22).
5. Obtain actual checkpoints, tokenizer and TensorRT-RTX SDK for **separate** numerical/GPU acceptance, not to reduce the functional score.
