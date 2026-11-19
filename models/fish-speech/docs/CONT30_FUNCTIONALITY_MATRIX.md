# Fish Speech 1.5 — CONT30 functional acceptance matrix

## Basis and accounting

Source contract is the uploaded `Fish-speech-pipeline-main (1).zip`, with the Rust implementation under `models/fish-speech`. GPU operability is **separate** from implemented API functionality. A PASS on Linux with TensorRT API bindings or the mock is **not** an RTX execution or numerical PyTorch parity result.

**Functional status:** 18 DONE, 4 PARTIAL, 0 MISSING / 22. Historical status index `(18 + 4×0.5)/22 = 90.91%` (CONT29: 88.64%).

**Fixed acceptance baseline:** 26/36 = **72.22%** (CONT29: 24/36 = 66.67%). The original denominator includes F08, F11, F12, F13, F14 and F22 and is not changed for presentation purposes.

**Operability:** historically 7/11 = **63.64%**. No Windows/RTX real-checkpoint inference was performed.

## Status and remaining requirements

| ID | Scope | Acceptance gates | State | Remaining functionality |
|---|---|---:|---|---|
| F01–F07, F09–F10, F15–F21 | Original Fish 1.5 tokenizer, model configs, CPU generator, Firefly, TTS, streaming, cache | — | DONE | None within accepted source scope |
| F08 | Audio reference codecs | 4/5 | PARTIAL | Self-contained MP3/AAC/MP4 and WMA/ASF decode without FFmpeg/system backend |
| F11 | Weighted native Slow prefill | 6/6 | DONE | None at source/API level; RTX verification is separate |
| F12 | Native cached Slow decode | **7/7** | **DONE CONT30** | None at source/API level; genuine TensorRT execution remains an operability gate |
| F13 | Fast AR GPU generation | 3/6 | PARTIAL | Device-side logits sampler; remove per-codebook readback/sync; GPU-side EOS/termination and bounded cancellation transfers |
| F14 | Naive AR TensorRT | 2/5 | PARTIAL | Resident Naive execution; expose Naive GPU generation/facade; GPU multi-codebook sampling/EOS/cancel lifecycle |
| F22 | GPU/request lifecycle | 4/7 | PARTIAL | Account TensorRT context/workspace and unmanaged allocations; GPU warmup/unload; reusable bounded scratch/device pool |

## F12 — seven completed source-level gates

- [x] CPU stateful Slow decode, RoPE, cache active-prefix semantics.
- [x] Preflight checks for KV positions and all segment inputs before enqueue.
- [x] Request-scoped resident TensorRT engines and contexts with safe ownership.
- [x] Shared DualAR backend: prefill/decode through public `CudaFishModel` runtime facade.
- [x] Same-stream GPU sequencing, deferred retirement and synchronization on success, error and unwind.
- [x] **Bounded bucketed Slow decode:** at most logarithmically many network graph shapes instead of one per absolute token. RoPE coefficients, attention mask and KV write location use the true runtime position. No truncated/sliding-window context is introduced. Static-KV tensors are zero-initialized by existing CUDA cache allocation. Different logical positions in a bucket reuse identical TensorRT blueprint fingerprints.
- [x] **Production facade and model-level reuse:** `CudaSemanticGenerator` selects the bucket, caches up to 8 prepared plan sets across calls, and retains its bounded serialized TensorRT engine registry. Within each request, execution contexts and persistent KV live for the full prefill/decode loop. Contexts are intentionally destroyed at request retirement; they are **not** cross-request pooled or shared unsafely.

## Verification performed for CONT30

- Core `cargo test --offline --locked`: 227 PASS, 2 ignored (199 unit + 28 integration).
- Outer `cargo test --offline --locked --lib`: 53 PASS, 2 ignored.
- Outer `cargo test --offline --locked --lib --features cuda-driver,tensorrt-api`: 63 PASS, 2 ignored.
- `cargo test --offline --locked --features tensorrt-mock`: 53 unit PASS, 2 ignored, and 8 external mock/blueprint/resident tests PASS.
- Rust formatting and Git whitespace checks: PASS.
- Pure CPU numerical parity: runtime provided RoPE matches static RoPE, and masked bucket GQA matches exact-prefix GQA with the same logical active length. Bucket boundaries, KV offsets, blueprint identity and plan LRU reuse have deterministic tests.
- No GPU, Windows, authentic checkpoint, TensorRT engine **execution**, latency benchmark or PyTorch CUDA numeric comparison: **NOT TESTED**.

## Next functional target

F13 (3/6): eliminate per-token GPU→CPU logits copies and host-side sampling. Design and test a GPU-side sampler with correct source sampling semantics, EOS and cancellation without introducing a second generation controller.
