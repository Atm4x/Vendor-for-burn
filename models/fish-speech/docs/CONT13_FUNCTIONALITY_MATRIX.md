# Fish Speech 1.5 — Functional parity matrix, CONT13

Source of truth: provided Fish-speech-pipeline-main(1).zip, not upstream S2. This matrix is a checklist of **behavior implemented in Rust**; GPU/Windows validation and performance are excluded.

Scoring rule for communication: `DONE = 1`, `PARTIAL = 0.5`, `MISSING = 0`. All items have *equal bookkeeping weight*, not equal engineering effort. `DONE` means the behavior has a callable Rust implementation with local tests, not that it has been proven equivalent on the actual model checkpoint. `PARTIAL` means some contracts or stages exist but there is not a complete functional path. The result should never be described as readiness for real inference.

| # | Original behavior | State | Rust scope and exact remaining functionality |
|---|---|---|---|
| F01 | Semantic/Firefly checkpoint config, `model_type` | PARTIAL | Typed `config.rs` plus manifests; finish authentic `config.json` to model-graph mapping and checkpoint aliases |
| F02 | Read converted model weights and validate tensor requirements | DONE | `src/weights.rs` SafeTensors with checksums, manifest, requirements; real original checkpoints still require external parity checks |
| F03 | Fish BPE tokenizer with 1036 specials, regex | DONE | New `src/fish_tokenizer.rs` loads `.tiktoken`; synthetic BPE tests pass; authentic tokenizer file remains unavailable for parity |
| F04 | `clean_text()` | DONE | New `core/src/text.rs` (quotes, emoji ranges, comma collapse) |
| F05 | `split_text()` | DONE | New `core/src/text.rs`, Python UTF-8 byte/float/sentence splitting; 144 original-Python-derived fixtures |
| F06 | `Conversation.encode_for_inference()`, VQ parts | DONE | Existing `core/src/conversation.rs`; calls are used by new prompt planner |
| F07 | `encode_tokens()` reference / reference-free prompts | DONE | New `core/src/prompt.rs`, user+assistant voice, proper closing semantics and VQ channels |
| F08 | `ReferenceLoader.load_audio()` | MISSING | File/bytes WAV/MP3 decode, arbitrary channels, audio format detection |
| F09 | Mono, resample, LogMel | PARTIAL | LogMel CPU and mono/44100 input exist; general resampling/multichannel loading missing |
| F10 | Firefly reference encoder ConvNeXt+FSQ | PARTIAL | TensorRT graph definition and CPU micro ops; no unified weighted runnable reference encoder |
| F11 | Slow AR prefill | PARTIAL | Graph, QKV/RoPE/GQA, stream plan; no weighted full prefill implementation |
| F12 | Slow AR next-token decode | PARTIAL | Structural pipeline and CPU state; no runnable model step producing logits |
| F13 | Fast AR codebooks sequence | PARTIAL | Cache/reset and graph contracts; no full generator connected to sampling |
| F14 | `NaiveTransformer` alternative | MISSING | Model kind exists, but full naive inference graph and runtime do not |
| F15 | Repetition penalty, top-p, Exp(1), EOS | PARTIAL | CPU sampling math and generation state; GPU/session sampling connected to weighted model not implemented |
| F16 | `generate_long()` iterative generation and context | PARTIAL | New chunk planner tracks prompts, generated acoustic codes and history; actual token model execution not connected |
| F17 | Firefly decode FSQ + HiFiGAN | PARTIAL | TensorRT graph and CPU operation contracts; no full weighted audio decode execution |
| F18 | Concatenation of decoded PCM, 44100 Hz | DONE | New `core/src/synthesis.rs` produces final single PCM for all segments and samples via backend contracts |
| F19 | End-to-end TTS orchestration | PARTIAL | New `synthesize()` handles prompt→generator→decoder→PCM, with fake backend tests; no real generator or decoder implementation bound |
| F20 | Streaming segment/header/final result variants | MISSING | Original wrapper has these events; native API currently returns final PCM, not stream events |
| F21 | `ReferenceLoader.load_by_id/load_by_hash` cache | MISSING | Reference ID and hash caching and cache on/off not implemented |
| F22 | Runtime lifecycle and cancellation | PARTIAL | Session and chunk cancellation only; no full threaded queue, within-token cancellation, model switching/unload |

**Accounting:** 7 DONE + 11 PARTIAL + 4 MISSING = 22 behavior units. Weighted checklist indicator: `(7 + 0.5*11)/22 = 56.8%` (display **~57%**). Raw closed behaviors: `7/22 = 31.8%`. This is a tracking convention only; several partial components are much more expensive than a closed one.

## Separate operability / verification checklist

| # | Verification | State |
|---|---|---|
| V01 | Rust 1.100.0 Core suite | PASS: 169 unit + 28 integration = 197 tests |
| V02 | Outer crate suite | PASS: 5 tests |
| V03 | Python exporter | PASS: 5 tests |
| V04 | Original-Python text splitter golden | PASS: 144 cases |
| V05 | Rust→TensorRT-RTX 1.7 bindings compile (`tensorrt-api`) | PASS (not GPU execution) |
| V06 | Current-run `tensorrt-mock` build | NOT REPEATED |
| V07 | Authentic checkpoint/tokenizer/audio golden parity | BLOCKED: no real checkpoint artifacts |
| V08 | TensorRT engine build and execution on RTX | NOT TESTED |
| V09 | End-to-end text+reference→PCM using authentic model | NOT TESTED |
| V10 | Windows package, Nsight, TorchInductor comparison | NOT TESTED |

**Verification summary:** 5/10 checklist gates have evidence; **0 actual GPU/e2e gates** have evidence. Do not equate 5/10 tests with a 50%-working synthesizer.

## Functional priorities, in order

1. Implement reference audio bytes/file decoding, mono mixdown and resampling. Keep the current mono 44.1 kHz LogMel CPU path.
2. Implement authoritative checkpoint `config.json` mapping and an executable (not just structurally compiled) Slow prefill/decode and Fast codebook generation contract using real weight tensors.
3. Bind actual generator and Firefly decoder to `SemanticGenerator` and `AcousticDecoder`, preserving first generated column semantics, acoustic indices and EOS.
4. Cover `NaiveTransformer` because the original checkpoint may select `model_type=naive`.
5. Implement reference-ID/hash caching and original streaming result semantics. Add cancellation within token/codebook generation and real model load/unload ownership.

CUDA implementation belongs in the **functional checklist** only where it is actual missing application code (e.g. a concrete native backend), never as a penalty for missing Windows/GPU *verification*. Windows tests and performance remain entirely in the separate verification checklist.
