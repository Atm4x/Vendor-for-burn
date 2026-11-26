# Fish Speech 1.5 — CONT25: request-scoped resident TensorRT contexts (F12)

## Source provenance

- Repository: `Atm4x/Vendor-for-burn`, branch `codex/fish-speech`.
- Base: Git Bridge `fish-speech-cont24-git-bridge.zip`; `source-head.txt` is `da185f3ad4c93c8d782e7b72c8e486571a76f5b1`, equal to GitHub HEAD before work. ZIP checksum and `git bundle verify` passed.
- Local-only implementation and tests with Rust 1.100.0, offline workspace vendor and Clang 17. No direct GitHub downloads for development.

## F12: new request-scoped GPU executor

1. `trtx_engine::with_resident_engine_sessions` creates one TensorRT runtime and all requested deserialized engines and execution contexts before the first enqueue. A `BTreeMap` keyed by validated serialized-engine identity exclusively borrows contexts throughout the request callback. Contexts, engines, runtime and pinned serialized images remain alive until a completion synchronizer runs, including Rust errors and caught panics. Duplicate/empty identities and images are rejected before initialization.
2. `EngineRegistry::with_resident_sessions` loads/builds all distinct blueprint images, validates identities, preserves leases for the lifetime of the session, and reuses one image/context per engine identity.
3. `cuda_trtx::with_resident_cuda_ar_session` precompiles all network phases from supplied Slow/Fast `ArStreamPlan`s, creates the CUDA primary context on the *selected* ordinal before creating TensorRT engines, and provides one `WeightedCudaArSession` for a caller-driven prefill/decode sequence. No TensorRT synchronization is inserted between network segments; CUDA KV copies and TensorRT enqueue share the same stream.
4. `CudaArDevice::with_context` reuses the **same** bound CUDA context for stream creation and enqueues. It avoids creating engine contexts on the wrong device before the session is established.
5. On a normal return, request error or panic, the wrapper calls `WeightedCudaArSession::retire_completed()` while CUDA buffers and TensorRT contexts are still alive, then tears them down. A stream synchronization failure is reported as an error; safe post-failure device teardown cannot be proved in this CPU-only environment.
6. The older `cached_ar_dispatch` path is retained temporarily for compatibility but marked deprecated because it synchronizes on every segment. Existing call sites in this repository do not invoke that helper.

## Regression evidence

- Three new `tensorrt-mock` integration cases: distinct resident engines; reuse of the same `ExecutionContext` for repeated decode; exactly one completion synchronization; error and panic synchronization; malformed image/key rejection.
- New Core regression: Slow prefill followed by two decode submissions on one stream, no intermediate sync, one completion sync and KV logical prefix preserved.
- Rust Core: **193 PASS, 2 ignored**, plus eight passing integration suites.
- Outer `fish-speech-vendor`: **45 PASS, 2 ignored**.
- `tensorrt-mock`: **45 PASS, 2 ignored** plus **4 blueprint mock, 3 resident context mock and 1 TensorRT mock PASS**.
- `cargo check --offline --locked --features 'cuda-driver tensorrt-api'`: **PASS** with `LIBCLANG_PATH=/usr/local/swift/usr/lib`.
- `cargo fmt --all -- --check` and `git diff --check`: **PASS**.
- No real Windows/NVIDIA GPU inference, authentic checkpoint/logit parity or TensorRT runtime benchmark was run. Do not attribute numeric speedups.

## F01: why the component is still PARTIAL

Source inspection shows that `src/model_config.rs` parses semantic `dual_ar`/`naive` variants but not a typed Firefly configuration schema. `core/src/firefly_ir.rs` builds the known Firefly 1.5 architecture with fixed shapes; `tools/export_checkpoint.py` normalizes selected `generator.` prefixes and both weight-normalization representations but does not prove all possible upstream checkpoint-key aliases. The Python shape-audit helper `tools/audit_firefly_shapes.py` expects the **original Python `vqgan/modules` sources** as an argument; those sources are not bundled in this Git Bridge repository.

F01 does not fundamentally require an RTX or a trained checkpoint to implement. It requires an explicit upstream Firefly config/variant contract plus full alias mapping and schema/shape tests. Authentic checkpoints are only needed for independent numerical and artifact compatibility gates. A fixed one-variant implementation is not enough to relabel F01 DONE when the tracker promises complete variant coverage.

## Progress and next steps

- Functional coverage: **15 DONE + 7 PARTIAL / 22 = 84.1%**. F12 has a real resident-session implementation now, but lacks integration into the top-level generation service and actual end-to-end execution of generated Slow tokens; therefore it stays PARTIAL.
- Operability: **7/11 historically confirmed gates = 63.6%**. Existing Linux/Core/mock gates pass; Windows RTX and genuine model output remain not tested.
- Next F12: expose a typed Slow prefill/decode controller over this session to the actual inference service, handle output logits and state transitions, and prove error/reset behavior in a mock end-to-end request.
- Next F01: obtain/check the actual Python Firefly config definitions and normalize/configure variants without relying on one fixed Rust graph; add collision-safe alias resolution and fixtures before marking DONE.
- F22 still needs unified TensorRT workspace/context allocation accounting and failure-aware resource retirement. F13/F14 remain unclosed.

No checkpoint weights, trained voice files, audio recordings or new third-party dependencies are included.
