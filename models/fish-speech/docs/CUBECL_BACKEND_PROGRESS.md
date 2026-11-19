# Fish Speech 1.5 — CubeCL implementation tracker

Baseline: `codex/fish-speech`; updated during CONT41 (2026-11-19).

**Meaning:** This is a checklist of implementation/verification gates, not a benchmark, a statement of GPU operability, or the percentage of the original Fish Speech functional port. The checklist denominator is fixed at **26 gates**, allocated C01=4, C02=6, C03=4, C04=4, C05=4, C06=4. A gate for numeric parity stays open until the actual CubeCL runtime executes and its results are compared with a reference. Pure graph compilation does not satisfy that gate.

**Overall: 14/26 = 53.8% implementation gates**. Real CubeCL CPU numerical execution: verified against `CpuIrInterpreter` for synthetic operator fixtures. CubeCL CUDA/Windows inference: unverified.

## C01 — bootstrap: 4/4

- [x] Optional `cubecl-backend` feature, no TensorRT dependency.
- [x] `Graph IR` compilation and deterministic binding preflight; reject invalid or unsupported graphs before dispatch.
- [x] Device kernels for base F32 operators: linear, linear+bias, RMSNorm, SiLU, Add, Mul, Identity, LastToken.
- [x] Runtime-executed base F32 kernel numerical parity with `CpuIrInterpreter` on CubeCL CPU (Tracel LLVM 23.1.0-4), including Linear, LinearBias, Identity, RMSNorm, SiLU, Add, Mul and LastToken, across multi-batch fixtures.

## C02 — Transformer operations: 6/6

- [x] `TokenEmbedding` and gated `FishEmbedding` with external-token and LUT checks.
- [x] Query/key/value unpacking plus position-correct interleaved RoPE (BF16-rounded rotary coefficients).
- [x] Typed native BF16 KV storage, write and prefix extraction; no silent F32 device-cache substitution.
- [x] Right-aligned masked causal grouped-query attention over BF16 K/V and F32 Q.
- [x] Complete synthetic DualAR (Slow/Fast) and Naive graph precompilation; no TensorRT lowering.
- [x] Runtime-executed CubeCL CPU operator parity, including embeddings, QKV/RoPE, BF16 KV update/prefix and multi-batch/multi-head causal GQA. BF16 external inputs are normalized to BF16 representable values before reference comparison; native BF16 support was validated by actual kernel execution. CUDA parity remains a separate C06 acceptance requirement.

**C02 caveats:** these are synthetic operator-level tests, not real checkpoint or end-to-end parity. Implementations currently use correctness-first scalar kernels; device-generated embedding indices still require a device-side error channel. Reuse of RoPE coefficient uploads and GPU attention performance remain future optimizations.

## C03 — stream-ordered CubeCL autoregression: 4/4

- [x] Backend-neutral execution plan/segments (decouple `ArStreamDevice` from `TrtBlueprint`). The stream trait is now generic over `N: ArNetworkSegment`; `ArStreamPlan<N>` and `ArStreamExecutor<D, N>` reuse typed inputs/outputs and KV barriers without a TensorRT network type. The old TensorRT split is retained as the `TrtBlueprint` adapter; CubeCL continues to use the backend-neutral `GraphExecutionPlan`. A fake non-TensorRT network executes the same preflight, segment sequencing, and KV-copy contract in Core tests.
- [x] Reusable model-scoped weight uploads, compiled graph cache and device buffer lifetime. `CubeClPreparedModel`/`CubeClModel` own prepared profiles and one-time device weight handles; direct `LoadedWeights` upload is supported.
- [x] Request-scoped KV, synchronization, cancellation and rollback/reset semantics. `CubeClRequest` preserves BF16 device cache handles and retains scratch until async stream synchronization; cancellation and post-enqueue failures poison the request. Runtime execution still requires hardware testing.
- [x] Native CubeCL `SemanticGenerator` for DualAR and Naive behind the existing TTS API. `CubeClSemanticGenerator` dispatches to `CubeClRequest` and the shared Core DualAR/Naive controllers; CPU host sampling/readback remains pending C04. Public `CubeClFishModel`/`open_cubecl` are available.

## C04 — complete generation: 0/4

- [ ] Native CubeCL semantic top-p/repetition sampling.
- [ ] Native CubeCL Fast and Naive acoustic codebook sampling.
- [ ] Device-resident tokens, EOS and streaming/cancel correctness.
- [ ] End-to-end CubeCL semantic generation + CPU Firefly, compared with the CPU trace.

## C05 — CubeCL Firefly: 0/4

- [ ] FSQ encode/decode.
- [ ] ConvNeXt encoder and related transforms.
- [ ] HiFiGAN / ConvTranspose audio decoder.
- [ ] Full reference encoding, audio output and numerical parity.

## C06 — Windows NVIDIA acceptance: 0/4

- [ ] Actual CubeCL CUDA runtime/compile/launch on Windows RTX.
- [ ] Checkpoint + audio trace parity versus Python / TensorRT with fixed seeds.
- [ ] Load/first-token/generation/RTF throughput and memory benchmarks.
- [ ] Profiler-driven kernel optimization and resource lifetime acceptance.

## Verification ledger — CONT41

| Check | Status |
| --- | --- |
| Rust 1.100.0, Tracel LLVM 23.1.0-4 `cargo test --offline --locked --features cubecl-cpu --lib` | PASS: 81 passed, 2 ignored (real CPU kernels) |
| `cargo fmt --all -- --check`; `git diff --check` | PASS |
| CubeCL greedy sampling on CPU, EOS/repetition/offset versus host | PASS: correctness-first kernel; C04 still open |
| CubeCL CUDA/Windows, full checkpoint inference, CPU Firefly end-to-end | NOT RUN / NOT CONFIRMED |

## Historical verification — CONT39

| Check | Status |
| --- | --- |
| `cargo test --offline --locked --manifest-path core/Cargo.toml --lib` | PASS: 207 passed, 2 ignored (includes independent-network fake executor test) |
| `cargo test --offline --locked --lib` | PASS: 58 passed, 2 ignored |
| `cargo check --offline --locked --features 'cubecl-backend cuda-driver tensorrt-api'` | PASS |
| `cargo test --offline --locked --features cubecl-backend --lib` | PASS: 72 passed, 2 ignored |
| `cargo fmt --all -- --check`; `git diff --check` | PASS |
| Actual CubeCL numerical execution, CUDA kernel launch, Windows RTX | NOT RUN / NOT CONFIRMED |

CONT39 moves `Binding` to `stream_binding.rs`, with a compatible re-export from `trtx_blueprint.rs`. TensorRT's existing stream device retains its exact behavior through `ArStreamDevice<TrtBlueprint>`. This is **not** an integration of the separate `codex/fish-speech-tensorrt` branch, and no remote diff was read. GPU-based CubeCL sampling, parity, and Firefly remain C04–C06.
