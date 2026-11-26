# Fish Speech 1.5 — CONT23: F11 runtime binding and weighted CUDA session

## Source provenance

- Repository: `Atm4x/Vendor-for-burn`, target branch `codex/fish-speech`.
- Source: verified CONT22 Git Bridge bundle whose `source-head.txt` is `9c9cc526697a4e9ebe9762d5c1533ffc150bb394`.
- Verified archive checksum and `git bundle verify` in local checkout; no remote source reads or remote builds were used for implementation.
- Rust 1.100.0 local toolchain; offline vendor dependencies.

## Implemented

### F11 — weighted AR device/session integration (PARTIAL)

1. `ArInputContract` extracts stable named runtime inputs and static checkpoint weights from a validated partitioned Slow/Fast AR stream plan. Previously materialized outputs and internally owned KV cache inputs are not incorrectly requested from the caller. It detects conflicting shapes and validates buffer sizes and missing/unknown inputs **before** an enqueue can partially execute.
2. `collect_ar_weight_types` merges weight requirements for multiple prefill/decode/Fast AR plans, reusing shared weights and rejecting incompatible tensor geometry. Empty plan sets are invalid.
3. `upload_plan_set_weights` uploads unique checkpoint tensors once to the existing CUDA memory-budgeted allocations. FP16/BF16/F32-to-FP32 transformations are processed **one tensor at a time** to avoid holding a second full checkpoint-sized FP32 copy in host RAM. CUDA upload completion is synchronized even on error before host data is released.
4. `WeightedCudaArSession` owns the `ArStreamExecutor` and the persistent uploaded weight bank. It supports ordered runtime tensor uploads, preflight-validated prefill/decode submission, output readback, request reset, Fast prefix reset, and stream-work retirement.
5. `cached_ar_dispatch` connects the existing `EngineRegistry`/`EngineIdentity` compile-load cache, TensorRT execution context, validated CUDA regions, and CUDA stream into the `ArStreamDevice` dispatch callback. The current correctness-oriented path synchronizes **after each TensorRT segment** and constructs contexts per session call. This is intentionally not a throughput-optimized resident-context implementation.
6. `CudaArDevice::download_tensor_bytes` waits for CUDA completion before returning host data. This allows Slow AR output logits to flow to a CPU sampler without host-buffer use-after-free.
7. Core `ArStreamExecutor::device_mut` provides controlled access to the executor's existing stream-owning device for ordered uploads/readback.

### F01 and F08

- No claim of completion. F01 checkpoint variants and key aliases require source/checkpoint-grounded mappings. F08 FFmpeg remains an opt-in external codec fallback; self-contained MP3/AAC/M4A/WMA decoding is not provided in the offline vendor.

## Local verification

- `cargo fmt --all -- --check`: PASS.
- `git diff --check`: PASS.
- Outer crate: **45 passed, 2 ignored, 0 failed**, including new contract tests.
- Core: **190 passed, 2 ignored, 0 failed** in the main suite; other Core integration tests passed.
- `cargo check --offline --locked --features 'cuda-driver tensorrt-api'`: PASS with `LIBCLANG_PATH=/usr/local/swift/usr/lib`.
- `cargo test --offline --locked --features tensorrt-mock`: PASS, including 4 blueprint tests and 1 TensorRT mock test.
- **No actual TensorRT-RTX engine creation/execution, Windows CUDA inference, checkpoint/audio parity or performance measurements.** Those remain NOT TESTED. Successful binding compilation is not real-device validation.

## Separate progress trackers

- **Functional parity:** 15 DONE + 7 PARTIAL / 22 = **84.1%**, unchanged. F11 gained GPU resident/session interfaces and real dispatch plumbing, but real weighted Slow prefill is not yet GPU-validated; do not mark DONE.
- **Operability:** 7/11 historically validated gates = **63.6%**, unchanged. Local build and mock tests reconfirmed; Windows RTX, native TensorRT execution and genuine checkpoint output parity remain unverified.

## Next work in functional order

1. F01: validate exact original Firefly model-config variants and semantic checkpoint aliases from source/checkpoint examples; no speculative renames.
2. F08: self-contained codecs or a shipped offline decoder, with licensing and test fixtures. No unconditional FFmpeg requirement.
3. F11: run genuine weighted Slow prefill, map returned logits, verify TensorRT bindings, shape/profile and PyTorch parity on Windows RTX.
4. F12: persistent native TensorRT engines/contexts without per-segment sync, Slow decode and KV active-prefix progression with real checkpoint.
5. F13: GPU Fast AR multi-codebook sampler/EOS/cache reset.
6. F14: native Naive AR TensorRT full inference.
7. F22: account TensorRT internal workspaces, CUDA scratch/context residency and warmup/unload under unified GPU budget.

No trained model weights, audio files or external codecs are committed.
