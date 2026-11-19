# Fish Speech 1.5 — CubeCL implementation tracker

Baseline: `codex/fish-speech`; updated during CONT46 (2026-11-19).

**Meaning:** This is a checklist of implementation/verification gates, not a benchmark, a statement of GPU operability, or the percentage of the original Fish Speech functional port. The checklist denominator is fixed at **26 gates**, allocated C01=4, C02=6, C03=4, C04=4, C05=4, C06=4. A gate for numeric parity stays open until the actual CubeCL runtime executes and its results are compared with a reference. Pure graph compilation does not satisfy that gate.

**Overall: 18/26 = 69.2% implementation gates**. Real CubeCL CPU numerical execution: verified against `CpuIrInterpreter` for synthetic operator fixtures. CubeCL CUDA/Windows inference: unverified.

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

## C04 — complete generation: 3/4

- [x] Native CubeCL semantic top-p/repetition sampling: device-resident logits with EOS output; CPU runtime compared against `logits_to_probs` and seeded `SessionRng` (synthetic vocabularies, ties, offsets, history penalties). The top-p kernel is correctness-first and single-lane; independent exponential draws are currently generated on the host and uploaded, not computed on GPU.
- [x] Native CubeCL Fast and Naive acoustic codebook sampling: both selection modes dispatch directly against device F32 logits. Naive uses an explicit `NaiveArForward::DevicePending` backend contract, not sentinel empty host logits. CPU controller test checks callback order, 16-token acoustic history, and EOS-column handling.
- [x] Device-resident token forwarding across Slow/Naive/Fast graphs, with request-owned sampled I32 handles, host-validated token shadows, sampled EOS readback, and controlled async scratch lifetime. Synthetic end-to-end generation passed for DualAR and Naive in greedy and stochastic top-p, including first-frame EOS and cancellation before the next frame on CubeCL CPU. Selected IDs still synchronize with CPU for session control and final trace; they are not uploaded back into the next graph. CUDA/Windows still unverified.
- [ ] End-to-end CubeCL semantic generation + CPU Firefly, compared with the CPU trace.

## C05 — CubeCL Firefly: 1/4

- [x] Native grouped residual FSQ encode/decode, executed on CubeCL CPU and compared against Core CPU and existing Python-source-formula golden fixtures. Multi-batch, multi-group, multiple quantizer counts, ignored `-1` codes, malformed indices, non-finite inputs and 27 geometry combinations tested. Public `CubeClFsqExecutor` exposes encode and decode; CPU is a reference, never a compute fallback. These APIs currently transfer inputs/outputs to the host and do not yet integrate with the Firefly ConvNeXt/HiFiGAN graph.
- [ ] ConvNeXt encoder and related transforms. CONT45 implemented native ConvNeXt and grouped/downsampling Conv1d. **CONT46 implements a complete 156-node reference encoder DAG from an explicit LogMel boundary through masks, backbone stages, feature downsampling and FSQ quantization.** Kernel execution is device-resident across ConvNeXt and FSQ, with readback only of final codes and FSQ error status. Both complete single-batch and two-batch synthetic checkpoint tests passed on CubeCL CPU; the final codes match Core CPU FSQ applied to reference features. Graph dependencies, checkpoint keys, dimensions and finite values are preflighted before dispatch. **Still open for this gate:** native CubeCL LogMel (the optional PCM entrypoint currently uses an explicit CPU preprocessor), reusable/model-scoped checkpoint weight handles, and full reference checkpoint parity. Do not interpret synthetic parity as actual-model acceptance.
- [ ] HiFiGAN / ConvTranspose audio decoder.
- [ ] Full reference encoding, audio output and numerical parity.

## C06 — Windows NVIDIA acceptance: 0/4

- [ ] Actual CubeCL CUDA runtime/compile/launch on Windows RTX.
- [ ] Checkpoint + audio trace parity versus Python / TensorRT with fixed seeds.
- [ ] Load/first-token/generation/RTF throughput and memory benchmarks.
- [ ] Profiler-driven kernel optimization and resource lifetime acceptance.

## Verification ledger — CONT46

| Check | Status |
| --- | --- |
| Graph IR LogMel → downsample Conv1d → channel LayerNorm → ConvNeXt residual vs Core CPU reference | PASS: real CubeCL CPU kernel execution; synthetic 128-channel stage, all three intermediate boundaries |
| Entire 156-node reference encoder DAG from external LogMel through FSQ encode | PASS: 1-batch synthetic checkpoint, FSQ codes match CPU FSQ on independently executed reference features |
| Two-batch CPU LogMel preprocessing → device encoder → FSQ encode | PASS: explicit CPU transform boundary and native CubeCL feature/FSQ path |
| Native FSQ encode accepts device-owned F32 handles without feature readback | PASS: 2 groups, 2 quantizers, 2 batches compared to CPU and uploaded path |
| Deterministic preflight rejects missing/NaN/malformed checkpoint bindings and incompatible LogMel shapes | PASS |
| Native CubeCL LogMel and real-checkpoint numeric parity | NOT IMPLEMENTED / NOT VERIFIED |
| Windows CUDA, TRT branch integration and GPU throughput | NOT VERIFIED / UNCHANGED |
| Full `cubecl-cpu` library regression | PASS: 102 passed, 4 ignored (both large encoder tests separately run and passed) |
| Separate ignored full encoder tests | PASS: 1/1 for single batch, 1/1 for two batches |
| `cubecl-backend` feature regression | PASS: 72 passed, 2 ignored |
| Default library / Core library | PASS: 58 passed, 2 ignored / 208 passed, 2 ignored |
| Rustfmt / git diff hygiene | PASS |

## Verification ledger — CONT45

| Check | Status |
| --- | --- |
| Native CubeCL CPU ConvNeXt block vs Core `ConvNextBlock::forward` | PASS: varied batch/channel/frame profiles, gamma on/off |
| Three chained ConvNeXt blocks with no intermediate host readback | PASS |
| Grouped/depthwise/downsampling Conv1d and dilation vs Core reference | PASS: five profiles |
| Firefly `reference_encode` Graph IR checkpoint-key binding for a real 128-channel stage-0 block | PASS: synthetic weights, preflight of missing parameter |
| Complete encoder reference audio to FSQ codes; CUDA/Windows GPU | NOT VERIFIED |

## Verification ledger — CONT44

| Check | Status |
| --- | --- |
| CubeCL CPU grouped residual FSQ encode/decode against Rust reference and golden fixtures | PASS |
| 27 group/quantizer/frame combinations plus negative cases | PASS |
| CubeCL CPU library | PASS: 94 passed, 2 ignored |
| CubeCL feature library | PASS: 72 passed, 2 ignored |
| Default library | PASS: 58 passed, 2 ignored |
| Core library | PASS: 208 passed, 2 ignored |
| Formatting and diff check | PASS |
| Original Python package execution / real weights / Windows CUDA | NOT VERIFIED |

## Historical verification — CONT43

| Check | Status |
| --- | --- |
| Real CubeCL CPU device-token packing into I32 column; semantic-derived acoustic code | PASS |
| DualAR + Naive real CubeCL CPU 2-frame synthetic generation with device-to-device Slow/Fast token inputs | PASS: greedy and top-p |
| Synthetic first-frame EOS and cancellation between frames | PASS for DualAR + Naive |
| Naive EOS sentinel `i32::MAX` fix | PASS: prior `u32::MAX` blocked all Naive device sampling |
| CubeCL CPU library | PASS: 90 passed, 2 ignored |
| CubeCL feature library | PASS: 72 passed, 2 ignored |
| Default library | PASS: 58 passed, 2 ignored |
| Core library | PASS: 208 passed, 2 ignored |
| `cargo check --offline --locked --features cubecl-backend`, formatting and diff check | PASS |
| Full real-weight CubeCL + CPU Firefly parity; Windows CUDA & throughput | NOT VERIFIED |

## Historical verification — CONT42

| Check | Status |
| --- | --- |
| Rust 1.100.0 + CubeCL CPU real runtime top-p numerical parity versus Core seeded sampler | PASS: 3 top-p CPU tests, including NaN rejection and 4/16/64/257 vocabularies |
| Core Naive device-pending controller with backend sampling, fixed semantic params, callback ordering and EOS | PASS |
| CubeCL CPU `cargo test --offline --locked --features cubecl-cpu --lib` | PASS: 84 passed, 2 ignored |
| `cargo test --offline --locked --features cubecl-backend --lib` | PASS: 72 passed, 2 ignored |
| `cargo test --offline --locked --lib` | PASS: 58 passed, 2 ignored |
| Core `cargo test --offline --locked --manifest-path core/Cargo.toml --lib` | PASS: 208 passed, 2 ignored |
| Formatting + `git diff --check` | PASS |
| GPU-resident token graph chaining, end-to-end checkpoint+Firefly, Windows CUDA, benchmark | NOT VERIFIED; separate open criteria |

## Historical verification — CONT41

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
