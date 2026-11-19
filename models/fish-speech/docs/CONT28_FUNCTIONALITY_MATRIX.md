# Fish Speech 1.5 — CONT28 functional acceptance matrix

## Scope and rules

Source of truth: the uploaded `Fish-speech-pipeline-main (1).zip`, particularly its single `FIREFLY_GAN_VQ_CONFIG`, `BaseModelArgs`, `NaiveModelArgs`, `DualARModelArgs` and their checkpoint-loading code. This is **not** a promise to support arbitrary configurations absent from this source.

A top-level F item is DONE when its behavior for that source is implemented in Rust and covered by local deterministic tests. Executing an authentic checkpoint on Windows/RTX is an independent *operability* gate, not a reason to keep a fully implemented CPU behavior PARTIAL. Conversely, a compiled TensorRT graph or a mock cannot establish a production TensorRT end-to-end path. Every incomplete feature retains an explicit acceptance checklist below.

Top-level status: **16 DONE, 6 PARTIAL, 0 MISSING / 22**. Historical metric: `(16 + 6 * 0.5) / 22 = 86.36%` (was 84.09%). It is a coarse *status index*, not a measurement of time remaining.

A second meter tracks **the actual six incomplete components**: 22 completed acceptance gates / 36 defined gates = **61.11%**. As future gates complete, this meter changes even before an entire F item reaches DONE. These gates represent source/implementation milestones, not equal engineering effort, and their count/definition must not be silently changed.

## Top-level inventory

| ID | Function | State |
|---|---|---|
| F01 | Original Semantic DualAR/Naive and Firefly 1.5 config/checkpoint contract | **DONE CONT28** |
| F02 | SafeTensors conversion/loading | DONE |
| F03 | Fish tokenizer | DONE |
| F04 | Text cleanup | DONE |
| F05 | Text splitting | DONE |
| F06 | Conversation serialization and VQ | DONE |
| F07 | Prompt/reference token encoding | DONE |
| F08 | Audio reference formats | PARTIAL |
| F09 | Resampling and LogMel | DONE |
| F10 | Weighted Firefly encode | DONE |
| F11 | Native weighted Slow prefill | PARTIAL |
| F12 | Native cached Slow decode | PARTIAL |
| F13 | Native DualAR Fast codebooks | PARTIAL |
| F14 | Native Naive AR in TensorRT | PARTIAL |
| F15 | Sampling, repetition penalty, EOS (CPU reference) | DONE |
| F16 | Long-text generation | DONE |
| F17 | Weighted Firefly decode | DONE |
| F18 | PCM, 44.1 kHz | DONE |
| F19 | End-to-end CPU TTS assembly (synthetic fixtures) | DONE |
| F20 | Streaming events | DONE |
| F21 | Reference cache | DONE |
| F22 | GPU/request lifecycle | PARTIAL |

## F01 — DONE for the exact supplied 1.5 implementation

- [x] Semantic `dual_ar` and `naive` dataclass field compatibility, including source `__post_init__` overwriting Slow `head_dim`.
- [x] Reject unknown fields, reward-model-only mode and Naive configurations with DualAR Fast fields rather than silently accepting incompatible model semantics.
- [x] Firefly 1.5 literal config parsed without executing arbitrary Python and validated against the fixed Rust graph.
- [x] Python-source-derived names/shapes for all **450 Firefly weights**, with canonical `generator.`/`model.` checkpoint prefix and weight-norm/QKV conversion tests.
- [x] Two-component, hash-validated manifest loader rejects incompatible configs before checkpoint shard loading; model wiring already has CPU checkpoint construction.

Out-of-scope: hypothetical Firefly variants not in the supplied Python source. If another real variant/checkpoint appears, register it as a separate compatibility target, not a perpetual blocker of F01.

## F08 — 4/5

- [x] WAV/FLAC/OGG decoding via Symphonia and mono conversion.
- [x] Optional MP3/AAC/M4A/WMA decode through an explicitly supplied local FFmpeg executable (no Python runtime).
- [x] Configurable `ReferenceAudioLoader` injected into the real weighted reference encoder and exposed on `CpuFishModel`.
- [x] Cache invalidation when swapping audio loader; source resampling and sample-rate contract preserved.
- [ ] Offline **self-contained** codec implementations or a licensed redistributable decoder backend for MP3, AAC/MP4 and WMA/ASF (the current vendored Symphonia set lacks MP3/AAC/MP4).

## F11 — 4/6

- [x] Source-ordered weighted Slow prefill and CPU reference interpreter.
- [x] TensorRT engine construction/load/cache contracts.
- [x] Checked weight upload, shape/binding validation and preflight before enqueue.
- [x] Lowered-output-to-original-ValueId mapping and logits handoff to common generation controller.
- [ ] Automatic preparation from loaded model artifacts in public native GPU facade (currently CPU is the model-runtime default).
- [ ] Production multi-request prefill route all the way to the TTS facade without manually assembled prepared plans.

## F12 — 5/7

- [x] CPU stateful Slow decode, active-prefix cache and RoPE semantics.
- [x] KV prefix validation before first network call.
- [x] Resident TensorRT runtime, engine and execution-context ownership.
- [x] Prepared bounded Slow prefill/decode route through shared DualAR controller.
- [x] Same-stream sequencing and deferred resource retirement/synchronization.
- [ ] Dynamic or windowed decode profiles with bounded compilation/context reuse.
- [ ] Production GPU model-runtime selection and reuse across whole requests.

## F13 — 3/6

- [x] Python-source-order Fast codebook loop in the shared controller.
- [x] Fast context remains on GPU across prepared segments.
- [x] Fast cache reset and semantic-to-first-codebook behavior.
- [ ] Device-side logits sampler.
- [ ] Remove per-token GPU→CPU logits readback/sync.
- [ ] GPU-side EOS/cancellation/termination contract with bounded transfers.

## F14 — 2/5

- [x] Weighted native CPU Naive AR generator.
- [x] Naive graph IR / basic TensorRT lowering and weight requirements.
- [ ] Complete resident TensorRT Naive segment execution path.
- [ ] Connect Naive GPU backend to the public generation/facade contract.
- [ ] GPU multi-codebook logits sampling, EOS and cancellation lifecycle.

## F22 — 4/7

- [x] Request queue, per-token cancellation and lifecycle controls.
- [x] Safe external GPU-buffer leases and stream completion boundaries.
- [x] Bounded serialized-engine cache.
- [x] Explicit CUDA allocation budget and reservations.
- [ ] Unified accounting of TensorRT internal execution-context/workspace and outside allocations.
- [ ] Genuine GPU warmup/unload transitions.
- [ ] Safe bounded, reusable scratch/device buffer pool across requests.

## Separate operability gates (not included in functional score)

Historical baseline **7/11 verified**; Windows NVIDIA runtime, authentic checkpoint tensors, numeric parity and packaging remain independent. CONT28 executes Rust Core, outer crate and Python exporter tests on Linux; no claim of successful Windows/RTX execution. Linux CUDA/TensorRT feature test build timed out during dependency compilation, so its CONT28 result is NOT VERIFIED (prior CONT27 feature tests remain evidence for that earlier commit only).
