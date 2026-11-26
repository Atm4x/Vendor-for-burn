# Fish Speech — CubeCL backend CONT35

Date: 2026-11-19. Target: `models/fish-speech` in `Atm4x/Vendor-for-burn`, branch `codex/fish-speech`.

## Result

Started a **new, independent CubeCL backend**. This is the first executable graph slice, **not** a complete CubeCL TTS pipeline. Existing TensorRT-RTX, CPU reference and F01–F22 functionality remain separate. Full design: `docs/CUBECL_BACKEND_SPEC_2026-11-19.md`.

### Implementation

- New Cargo features: `cubecl-backend` (CubeCL kernel/graph definitions, no platform runtime); `cubecl-cpu` (host validation runtime, needs Tracel LLVM); `cubecl-cuda` (native CUDA runtime, needs Tracel LLVM). Optional `cubecl = 0.11.3` from the already vendored dependency tree; default features remain empty. Existing `cuda-driver + tensorrt-api` stays unchanged.
- New `src/cubecl_backend.rs`: independent `CubeClGraph::compile` plus `CubeClGraphExecutor::run`, consuming canonical `core::graph_ir::Graph`. No `trtx` dependency and no TensorRT calls.
- Real CubeCL kernels (compiled with the CubeCL macros): F32 `Add`, `Mul`, `SiLu`, `Linear`, `LinearBias`, `RmsNorm`, `LastToken`; `Identity` shares an existing device handle. Tensor layouts are validated through `Graph::validate`. This initial GEMM and RMSNorm are deliberately simple, correctness-oriented kernels, **not** optimized inference kernels.
- Preflight rejects unsupported ops/BF16, unknown/missing/shape-mismatched inputs before allocating/uploading and dispatching. No silent CPU fallback.
- Host-input → CubeCL graph → host-output execution harness; `NamedTensors` are used only at this correctness-testing boundary. Full model generation must retain tensors on device between stages.
- Tests: preflight of supported graph and input rejection; immediate compile-time rejection of unsupported embedding; CubeCL CPU numerical parity test scaffold covering LinearBias→SiLu→Add→Mul→RmsNorm→LastToken against `CpuIrInterpreter`.
- Restored deterministic offline vendor metadata: `vendor/** -text` and normalized checksum manifests for 17 previously mismatching packages (281 tracked text files); no third-party source code was changed. These changes are necessary because the git checkout normalizes CRLF but Cargo checksums were recorded against CRLF content. Fresh offline Cargo feature compilation now succeeds on the ordinary repository checkout.

## Verification

| Gate | Status |
|---|---|
| `cargo check --offline --features cubecl-backend` (regular checkout) | PASS |
| `cargo test --offline --features cubecl-backend --lib` | 60 PASS / 2 ignored (with local checksum-repaired vendor mirror before metadata fix) |
| `cargo test --offline --locked --lib` (no CubeCL) | 58 PASS / 2 ignored |
| `cargo check --offline --locked --features 'cuda-driver tensorrt-api'` | PASS |
| `cargo test --offline --features cubecl-cpu --lib` | BLOCKED: `tracel-llvm-bundler 23.1.0-4` tried to download Tracel LLVM and environment DNS/network is disabled; Rust compilation failed before tests could run |
| Actual GPU CubeCL kernel launch, BF16 KV, semantic/Firefly end-to-end | NOT TESTED / NOT IMPLEMENTED |

**No numerical parity has been claimed.** CubeCL kernels are compiled Rust/CubeCL source, but runtime execution requires the external Tracel LLVM package and a functioning CubeCL runtime. Current preflight tests do not execute kernels.

## Remaining steps

1. Obtain/prepare Tracel LLVM 23.1.0-4 for CPU/CUDA runtime builds, and run the actual reference-parity tests. Verify the kernels' compiled native code, not just macro expansion.
2. Implement device-resident `GraphDevice` interface and a backend-neutral execution segment plan; remove the `TrtBlueprint` dependency from the shared stream executor without breaking the TensorRT backend.
3. Finish Transformer operator coverage: TokenEmbedding, FishEmbedding, QKV split, RoPE, causal GQA, KV write/prefix with BF16 precision and bucketed runtime positions.
4. Connect CubeCL to `SemanticGenerator` for both DualAR and Naive AR, then device-resident sampling and TTS with CPU Firefly.
5. Native CubeCL Firefly and Windows RTX numerical/performance gates. Only then claim two complete GPU backends.

## Progress accounting

CubeCL is a new backend, **not** an additional condition of the old Fish Speech F01–F22 metric. Initial stage C01: source implementation and validation started; numerical runtime parity blocked by missing LLVM. C02–C06 not complete. TensorRT functional coverage and Windows RTX operability percentages are unchanged.
