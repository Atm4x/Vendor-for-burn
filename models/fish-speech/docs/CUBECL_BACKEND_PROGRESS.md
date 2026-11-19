# Fish Speech 1.5 — CubeCL implementation tracker

Baseline: `codex/fish-speech`; updated during CONT37 (2026-11-19).

**Meaning:** This is a checklist of implementation/verification gates, not a benchmark, a statement of GPU operability, or the percentage of the original Fish Speech functional port. The checklist denominator is fixed at **26 gates**, allocated C01=4, C02=6, C03=4, C04=4, C05=4, C06=4. A gate for numeric parity stays open until the actual CubeCL runtime executes and its results are compared with a reference. Pure graph compilation does not satisfy that gate.

**Overall: 10/26 = 38.5% implementation gates**. CubeCL numerical execution: unverified. CubeCL CUDA/Windows inference: unverified.

## C01 — bootstrap: 3/4

- [x] Optional `cubecl-backend` feature, no TensorRT dependency.
- [x] `Graph IR` compilation and deterministic binding preflight; reject invalid or unsupported graphs before dispatch.
- [x] Device kernels for base F32 operators: linear, linear+bias, RMSNorm, SiLU, Add, Mul, Identity, LastToken.
- [ ] Actual CubeCL kernel numerical parity with `CpuIrInterpreter`. `cubecl-cpu` requires Tracel LLVM 23.1, unavailable in the offline environment.

## C02 — Transformer operations: 5/6

- [x] `TokenEmbedding` and gated `FishEmbedding` with external-token and LUT checks.
- [x] Query/key/value unpacking plus position-correct interleaved RoPE (BF16-rounded rotary coefficients).
- [x] Typed native BF16 KV storage, write and prefix extraction; no silent F32 device-cache substitution.
- [x] Right-aligned masked causal grouped-query attention over BF16 K/V and F32 Q.
- [x] Complete synthetic DualAR (Slow/Fast) and Naive graph precompilation; no TensorRT lowering.
- [ ] Runtime-executed operator parity, including BF16 KV and GQA, on CubeCL CPU/CUDA. BF16 support must be probed on the selected device; no real kernel execution has been verified here.

**C02 caveats:** implementations currently use correctness-first scalar reference kernels and host-staged graph bindings. This is not a persistent KV device owner, an optimized attention engine or a streaming generation backend. `TokenEmbedding` and `FishEmbedding` currently require external token IDs; device-generated indices are rejected at compile time pending a device-side error channel. Reuse of RoPE coefficient uploads should be optimized after numerical verification.

## C03 — stream-ordered CubeCL autoregression: 2/4

- [ ] Backend-neutral execution plan/segments (decouple `ArStreamDevice` from `TrtBlueprint`). `GraphExecutionPlan` is now backend-neutral and used by CubeCL, but the old TensorRT-specific trait is retained and still requires consolidation.
- [x] Reusable model-scoped weight uploads, compiled graph cache and device buffer lifetime. `CubeClPreparedModel`/`CubeClModel` own prepared profiles and one-time device weight handles; direct `LoadedWeights` upload is supported.
- [x] Request-scoped KV, synchronization, cancellation and rollback/reset semantics. `CubeClRequest` preserves BF16 device cache handles and retains scratch until async stream synchronization; cancellation and post-enqueue failures poison the request. Runtime execution still requires hardware testing.
- [ ] Native CubeCL `SemanticGenerator` for DualAR and Naive behind the existing TTS API.

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

## Verification ledger — CONT37

| Check | Status |
| --- | --- |
| `cargo test --offline --locked --features cubecl-backend --lib` | PASS: 69 library tests, 2 ignored |
| `cargo test --offline --locked --lib` | PASS: 58 tests, 2 ignored |
| `cargo test --offline --locked --manifest-path core/Cargo.toml --lib` | PASS: 204 tests, 2 ignored |
| `cargo check --offline --locked --features 'cubecl-backend cuda-driver tensorrt-api'` | PASS |
| `cargo fmt --all -- --check`; `git diff --check` | PASS |
| Actual CubeCL numerical execution, CUDA kernel launch, Windows RTX | NOT RUN / NOT CONFIRMED |

Existing TensorRT-RTX remains untouched. The user's successful TensorRT run involved unpushed changes not inspected here. The current C03 work is a model/request ownership and execution-plan contract, not a verified TensorRT/CubeCL parity result.
