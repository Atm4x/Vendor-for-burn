# Fish Speech 1.5 / CubeCL — CONT38

Date: 2026-11-19. Source branch `codex/fish-speech`; based on CONT37.

## Outcome

**C03 is 3/4; total implementation checklist is 11/26 = 42.3%.** This measures code/contract gates, **not measured inference parity**. No real CubeCL kernel execution, Windows RTX parity, or end-to-end synthesis was run in this environment.

### Implemented

1. `core/src/naive_generation.rs`: backend-neutral `NaiveArBackend`, `NaiveArForward`, and `generate_naive_ar`. Preserves the original Naive sequence: one Slow AR forward produces both semantic logits and all acoustic codebook logits; fixed `temperature=0.1`, `top_p=0.1`, `repetition_penalty=1` for semantic; request settings for codebooks; 16-history handling; cancellation before and during codebooks; append EOS acoustic column before finishing.
2. `core/src/cpu_weighted_naive.rs`: now delegates generation order to this common controller. Existing CPU tests and the new scripted controller tests pass; the CPU weight interpreter remains a separate backend.
3. `src/cubecl_backend/generation.rs`: adds `CubeClSemanticGenerator`, `CubeClDualArBackend`, and `CubeClNaiveBackend`. CubeCL uses the same `generate_dual_ar` and `generate_naive_ar` controllers as CPU. The backend dispatches the already compiled CubeCL graphs through request-scoped BF16 KV state. No TensorRT code paths or fallback are used.
4. `src/model_runtime.rs`: `CubeClFishModel` alias, `FishModel::open_cubecl`, and `LoadedFishArtifacts::into_cubecl_model`; common tokenizer, reference encoder, CPU Firefly decoder, and streaming TTS machinery remain shared.
5. A one-entry cached `CubeClModel` per prompt geometry reuses prepared graphs and uploaded semantic weights; each synthesis request receives a fresh KV owner. Geometry changes invalidate that cache. Position-specialized Slow decode remains limited to 64 generated tokens, with an explicit error rather than silent overcompilation.
6. Added synchronous stream retirement for the existing synchronous generator boundary. Both success and error paths explicitly retire work, and `CubeClRequest` now synchronizes outstanding work on Drop. The controller retires previous work between acoustic columns to prevent scratch growth.
7. Added validation tests for prompt geometry, unsupported token IDs, logits dtype/shape, Naive zero-limit semantics, scripted Naive EOS and cancellation, and previous-column routing.

### Verification

| Command | Result |
| --- | --- |
| `cargo test --offline --locked --features cubecl-backend --lib` | 72 passed, 2 ignored |
| `cargo test --offline --locked --lib` | 58 passed, 2 ignored |
| `cargo test --offline --locked --manifest-path core/Cargo.toml --lib` | 206 passed, 2 ignored |
| `cargo check --offline --locked --features 'cubecl-backend cuda-driver tensorrt-api'` | PASS |
| `cargo fmt --all -- --check` / `git diff --check` | PASS |
| Runtime-executed CubeCL numerical parity | NOT RUN; Tracel LLVM 23.1 unavailable |
| Windows RTX kernel launch / full TTS checkpoint audio parity | NOT RUN |

### Remaining C03 / next stage

- **C03 4th gate:** consolidate the old TensorRT-specific `ArStreamDevice` / `TrtBlueprint` boundary with the backend-neutral Graph execution interface. Current `GraphExecutionPlan` and CubeCL model/request are independent; the old TensorRT adapter remains intentionally unchanged to avoid disturbing the user's unpushed TensorRT fixes.
- **C04:** device-resident semantic/acoustic sampling and EOS; currently CubeCL reads logits to CPU and uses shared host sampling. Fast context is also transferred via host between Slow and Fast. Porting sample kernels, and reusing a device-resident fast context, are pending.
- **C01/C02 numerical parity:** execute real CubeCL kernels before declaring correctness. Rust type-checking, preflight and CPU scripted tests are not numerical parity evidence.
- **C05:** CubeCL Firefly decode/encode, separate from the already shared CPU Firefly.
- **C06:** CUDA/RTX runtime, checkpoint trace parity, throughput/RTF and profiling.

No TensorRT source modules, vendored libraries, or legacy/obsolete scripts were modified in CONT38. No Git Bridge export is required: the local source tree was already current at CONT37.
