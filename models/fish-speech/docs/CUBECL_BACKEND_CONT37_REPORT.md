# Fish Speech CubeCL — CONT37 report

Date: 2026-11-19. Branch: `codex/fish-speech`. Base: CONT36 tree `1988179de48fad42d1007b08951627ddb5ba6d01` (GitHub commit `2e1f28a034e6eea9a7b5bc132b65c03260851f3f`).

## Scope and outcome

Continued C03 without touching the TensorRT backend or its unpublished local fixes. Added a CubeCL model/request ownership layer and a backend-neutral Graph IR execution phase plan. This is **not** a complete CubeCL `SemanticGenerator`; the real CubeCL CPU/CUDA runtime cannot be exercised in this environment without the missing Tracel LLVM bundle.

### New and changed modules

- `core/src/graph_stream_plan.rs`: `GraphExecutionPlan`, `GraphExecutionPhase::{Compute,KvWrites}` and typed `GraphKvWrite` records based only on `Graph IR`, `ValueId` and `Op`. The plan preserves topological node order and partitions network/KV phases without a TensorRT blueprint. The CubeCL executor now actually traverses this plan. Existing `ArStreamDevice` and TensorRT's older `ArStreamPlan` remain unchanged for compatibility; their final decoupling is **still pending**.
- `src/cubecl_backend/runtime.rs`: reusable buffer-handle execution (`execute_handles`), separate host staging/readback, model-weight uploads, BF16 buffer checks, zero-filled typed KV allocation, and request-input validation that excludes model weights and request-owned cache handles. The existing one-shot `CubeClGraphExecutor::run` remains available.
- `src/cubecl_backend/session.rs`: `CubeClPreparedModel` merges graph weight/cache requirements and rejects cross-profile incompatibilities before allocation. `from_transformer` prepares Slow prefill, Slow decode (position-specialized) and DualAR Fast plans, with an explicit 64-token profile limit. `CubeClModel` uploads F32 weights once, shared between requests, and can ingest manifest-verified `LoadedWeights` directly with one-tensor-at-a-time F16/BF16→F32 conversion.
- `CubeClRequest`: owns device BF16 KV cache handles per request; each `KvUpdate` result replaces its cache binding without round-tripping KV through CPU. Runtime inputs (tokens and LUT) are uploaded per step; only non-KV graph outputs are downloaded. Holds in-flight handles until an explicit async stream synchronization. Enforces monotonic Slow/Fast positions, request-scoped Fast reset, resource limits, cancellation before/after submission, poison on partial execution failure, and explicit synchronized `reset_request`.

### Important restrictions

- The runtime code builds but **there is no verified CubeCL kernel execution**; tests of ownership and request sequencing are host-side preflight tests. A real request needs a supported CubeCL runtime/client and numerical checks.
- One `CubeClRequest::run` call still reads semantic/fast logits on the host. Native semantic/acoustic sampling and the `SemanticGenerator`/public `FishModel` adapter remain for C03/C04.
- Slow decode graphs are currently position-specialized. Dynamic bucketed profiles, persistent scratch allocation reuse and device-side error reporting for device-generated token IDs are future work.
- `retire_completed` and `reset_request` are async; callers must await synchronization before discarding the model/request owning outstanding handles. On errors after launch, the request is poisoned until reset.
- Original TensorRT `ArStreamDevice` still accepts `TrtBlueprint`. A second backend-neutral execution plan has been introduced and is used by CubeCL, but the legacy TensorRT trait compatibility boundary has not been fully consolidated.
- TensorRT reportedly runs on the user's machine with unpublished fixes; these were not available, inspected or changed here.

## Verification

| Gate | Result |
|---|---|
| `cargo test --offline --locked --features cubecl-backend --lib` | PASS: **69**, ignored 2 |
| `cargo test --offline --locked --lib` | PASS: **58**, ignored 2 |
| `cargo test --offline --locked --manifest-path core/Cargo.toml --lib` | PASS: **204**, ignored 2 |
| `cargo check --offline --locked --features 'cubecl-backend cuda-driver tensorrt-api'` | PASS |
| `cargo fmt --all -- --check`; `git diff --check` | PASS |
| CubeCL CPU execution / numeric parity | NOT RUN — Tracel LLVM 23.1 unavailable |
| CubeCL CUDA/Windows RTX execution | NOT RUN |

## Fixed progress accounting

**Overall C01–C06: 10/26 = 38.5% implementation gates.** C01 3/4, C02 5/6, C03 2/4, C04 0/4, C05 0/4, C06 0/4. These are implementation gates, not evidence of end-to-end GPU operability. C03 credited model-scoped weights/compiled-plan reuse and request-scoped KV/synchronization/cancellation/rollback contracts. C03's backend-neutral *legacy* `ArStreamDevice` consolidation and complete `SemanticGenerator` adapter are still outstanding.

## Next iteration

1. Add CubeCL DualAR and Naive adapters to `SemanticGenerator`, driven by the existing generation state machine, with backend-owned KV. Keep non-KV CPU readback for first parity harness; avoid duplicating TTS orchestration.
2. Complete the `ArStreamDevice`/plan type boundary so TensorRT-specific `TrtBlueprint` stays only within the TensorRT adapter.
3. Obtain a working supported CubeCL runtime and run a **real** operator parity test, including BF16 KV, RoPE and GQA, before calling C01/C02 fully complete.
4. Only then optimize/replace host logits sampling with device-resident CubeCL sampling (C04).
