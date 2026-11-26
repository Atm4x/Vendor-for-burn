# Fish Speech 1.5 — CubeCL Backend Specification

Status: **implementation started; not an end-to-end GPU TTS backend**. Target repository: `Atm4x/Vendor-for-burn`, `models/fish-speech`. Source baseline: CONT34 Git tree `0f6e1395e6664c20e06f9eed143f167dff109e25` (corresponds to GitHub commit `ed881a2e079e5765979b4d4138b2997914605aa6`). Original Python reference: `Fish-speech-pipeline-main (1).zip`.

## 1. Goal and non-goals

Create a **second, native CubeCL computation backend** for Fish Speech 1.5. Keep existing TensorRT-RTX and CPU references untouched. CubeCL must execute real model operators; merely translating names to a `TrtBlueprint`, or delegating to TensorRT, does **not** count. Support initial CubeCL CUDA and eventually wgpu; the same kernels should have a CPU runtime for testability where feasible. Do not require Python at runtime. Do not introduce Burn into the critical runtime merely for its tensor API: use CubeCL directly, while allowing optimized Burn-CubeCL kernels after benchmarking. Do not force CubeCL and TensorRT to share underlying CUDA handles or allocator; they are **alternative** backends, not a single mixed CUDA stream.

The initial milestone is a small **real CubeCL graph-execution vertical slice**, tested against `CpuIrInterpreter`. It is not acceptable to relabel TensorRT-based TTS as CubeCL.

## 2. Existing architecture and reuse

- `core/src/graph_ir.rs`: the backend-independent `Graph`, `Node`, `Op`, `ValueId`, `TensorType`, `ScalarType` and origin / output definitions. This is the primary canonical computation contract.
- `core/src/transformer_ir.rs`: Slow prefill/decode, DualAR Fast, Naive AR graph construction; reuse unchanged.
- `core/src/ar_generation.rs`: generator state, semantic/acoustic sampling, EOS ordering and cancellation; reuse via a backend adapter.
- `core/src/kv_device_owner.rs` and `ar_stream_executor.rs`: stream-ordered KV ownership principles and state transitions can be reused. **However**, `ArStreamDevice::enqueue_network` currently accepts `TrtBlueprint`. That concrete coupling must be removed before using this abstraction for CubeCL; do not route CubeCL through TensorRT-labelled types as the final architecture.
- `core/src/firefly_ir.rs`: canonical Firefly graph; reuse for subsequent CubeCL encoder/decoder.
- `src/model_runtime.rs`: shared TTS façade; reuse without copying orchestration code.
- `src/weights.rs`: manifest-validated SafeTensors loading, shape/dtype matching. Runtime-specific upload is necessary.
- `src/cuda_*` and `src/trtx_*`: TensorRT implementation details; reference semantics only, not a library dependency for CubeCL.
- `src/cuda_fast_sampling.cu` / `src/cuda_semantic_sampling.cu`: algorithms are reusable, NVRTC kernels are not CubeCL implementations.

## 3. Architecture targets

```text
                       FishModel<SemanticGenerator>
                         TTS / audio / streaming
                                  |
                 DualAr / Naive generation state machines
                                  |
              backend-neutral ModelGraph / ExecutablePlan
                                  |
         +------------------------+------------------------+
         |                        |                        |
    CPU reference            TensorRT-RTX                CubeCL
   CpuIrInterpreter       TrtBlueprint lowering        Graph lowering
                              TrtEngine          CubeClGraphExecutor
                                 |               / CUDA / CPU / wgpu
                       TensorRT runtime           CubeCL client
```

Canonical source for model behavior is **Graph IR**. Lowering is a strictly validated backend operation. The CubeCL side must be capable of compiling once per shape/profile, uploading checkpoint weights once per model, maintaining device-resident activations and KV across steps, and letting the generic generator choose when to retire or reset a request.

### Contracts to introduce

1. `GraphCompiler`: `compile(graph: &Graph) -> Result<CompiledGraph>`; reject unsupported ops/dtypes/layouts **before** enqueuing anything. No silent CPU or TensorRT fallback.
2. `GraphDevice`: `upload`, `allocate`, `execute(&CompiledGraph, &bindings)`, `download`, `synchronize`, `retire`, `reset_request`; owns its buffer handles. Backend-specific buffers must not cross backend boundaries accidentally.
3. `KvDevice`: tensor storage with model-defined BF16 write semantics, capacities, offset checks, and poison/recovery behavior; stream-ordered with graph execution.
4. `SemanticGenerator` adapter: shared DualAR/Naive sequencing and EOS. Selecting `cubecl` must not create a TensorRT runtime.
5. `AcousticDecoder` / `ReferenceEncoder`: independent selection of CPU or GPU Firefly. CubeCL AR and CPU Firefly should work together as the first full TTS route; CubeCL Firefly follows separately.

Do **not** make `TrtBlueprint` the public portable IR. Migrate shared sequencing in `ArStreamPlan` to a backend-neutral `ExecutionSegment` once CubeCL's minimum execution can be exercised, leaving a compatibility adapter for existing TensorRT tests. Keep `Graph IR` as the semantic source of truth; do not duplicate model configurations.

## 4. Operator coverage

| Priority | Graph IR operation | First correct CubeCL implementation | Optimization later |
|---|---|---|---|
| P0 | `Identity`, `Add`, `Mul`, `SiLu` | F32 independent elementwise kernels; no host arithmetic | vectorization / fusion |
| P0 | `Linear`, `LinearBias` | F32 correctness GEMM; weights `[out,in]`, output `[B,T,out]` | tiled / Tensor Core / optimized CubeCL matmul |
| P0 | `RmsNorm` | F32 reduction and affine weight | fused warp/block reduction |
| P1 | `TokenEmbedding`, `FishEmbedding` | checked embedding gather and codebook sharing | merged load / vectorized layout |
| P1 | `SplitQkv`, `RoPe`, `LastToken` | exact layout and positional math | fused QKV / RoPE |
| P1 | `KvUpdate`, `KvPrefix`, GQA attention | BF16 K/V cache and masked causal attention | persistent KV, specialized decode kernels |
| P1 | DualAR Fast / Naive outputs | same tensor shapes and codebook ordering | fused sampling |
| P2 | semantic / acoustic sampling | native CubeCL top-p, repetition penalty, exponential race, EOS | remove sync per token |
| P2 | Firefly ConvNeXt / HiFiGAN / FSQ | complete native audio encoder and decoder | convolution-specific kernels |

All tensors must be validated for dtype, shape, strides and maximum allocation size. Initially restrict to dense contiguous F32/I32 where implemented; BF16 conversion must be explicit (do not silently treat BF16 as F32). Invalid token IDs must fail deterministically, without out-of-bounds GPU reads. `f32` numeric parity must be tested within operator-specific tolerance, not by exact bit equality unless required. Never claim numerical parity without actual output comparisons.

## 5. Runtime, feature flags and isolation

- `default`: existing CPU reference; **no CubeCL compilation**.
- `cubecl-backend`: CubeCL core kernels / generic graph executor; compiled only when requested.
- `cubecl-cpu`: CubeCL CPU runtime used for validation without CUDA.
- `cubecl-cuda`: CubeCL CUDA runtime; no TensorRT dependency; native NVIDIA driver integration via CubeCL.
- Later `cubecl-wgpu`: wgpu runtime and feature portability matrix.
- `cuda-driver + tensorrt-api`: existing TensorRT path, unchanged.

The backend selection belongs in the model factory / service configuration. Exactly one semantic backend is active per running model. Separate acoustic backend selection is allowed, e.g. CubeCL AR + CPU Firefly. Existing TensorRT keys and engine cache must not be reused for CubeCL kernel cache.

## 6. Memory ownership and scheduling

CubeCL buffer handles live until all queued device work is retired; input/output tensors and KV are request-scoped or model-scoped appropriately. Allow same-stream reuse without a CPU barrier. Transfer between streams requires explicit synchronization or event dependencies. GPU memory accounting includes weights, scratch, KV and compilation caches; a cleared model must release all owned allocations. `max_inflight_bytes`, scratch limits and rollback on failed enqueue should preserve existing semantic guarantees. GPU-to-CPU readback is only for returned results or necessary host decisions; internal attention and sampling should remain device-resident.

First executor may synchronously return host tensors for operator parity, but **that is a harness, not the final generation API**. It must be documented as such. No implicit CPU reference execution inside a CubeCL-backed method.

## 7. Staged implementation and acceptance gates

1. **C01 / bootstrap (this cycle):** add optional CubeCL feature(s), independent graph validation, real CubeCL kernels for a subset (`Identity`, F32 elementwise, linear and normalization as capacity permits) and isolated tests against `CpuIrInterpreter`. The default and TensorRT feature builds must remain unchanged. Unsupported model ops explicitly error before dispatch. No extra Git Bridge exports if the current local tree already matches GitHub.
2. **C02 / full Transformer op coverage:** all P1 ops, BF16 KV semantics, attention, RoPE, embeddings; negative and numerical reference tests. Bucketed decode must be position-correct. A fabricated all-zeros checkpoint is not proof of parity; validate actual tensor outputs.
3. **C03 / stream-ordered CubeCL AR:** backend-neutral execution plan, model-scoped weights, request-scoped KV and buffers, DualAR + Naive adapter; EOS, cancellation, reset/retry and memory budget tests.
4. **C04 / complete generation:** semantic and Fast/Naive acoustic sampling in CubeCL, device-resident codebooks and stream retirement; CPU Firefly end-to-end WAV output, parity against CPU trace.
5. **C05 / CubeCL Firefly:** native CPU/GPU parity for FSQ, ConvNeXt, HiFiGAN, spectrogram and resampling path; no TensorRT dependency.
6. **C06 / Windows NVIDIA acceptance:** use provided real checkpoint/audio, compare against original Python and TensorRT with fixed seeds; run Windows CUDA/NVRTC/CubeCL environment checks; benchmark load, first token, tokens/s, peak VRAM, end-to-end RTF. Profile top bottlenecks with Nsight Systems and Nsight Compute. No production-readiness claim before this gate.

### Acceptance evidence

For each operator: (1) graph and shape validation; (2) deterministic numerical parity with CPU reference; (3) negative cases and resource/lifetime tests; (4) CubeCL CPU compilation/execution where available; (5) on the RTX target, actual GPU execution and latency. Functional progress and machine operability are separate. A function counts as fully ported only when the real CubeCL executor produces it; stubs, test-only host emulation and TensorRT delegation do not qualify.

## 8. Risks and architectural decisions

- **Numerical semantics:** TensorRT F32/BF16/TF32 reduction order may differ from CubeCL; set tolerances explicitly per operation, with hard failure on NaN/Inf unless reference expects them.
- **Convolution performance:** the expensive Firefly portion will need dedicated CubeCL kernels; naive im2col or scalar convolution is only a reference, not an optimized inference path.
- **Dual ownership:** never share raw TensorRT device pointers with CubeCL without a demonstrated, stream-safe interop contract. Backend switching unloads/reloads runtime state.
- **Engine cache:** TensorRT compiled engines and CubeCL kernel compilation caches are not interchangeable.
- **No GPU in this environment:** compilation is not evidence of working Windows CUDA; provide reproducible target-side test instructions later.
- **API version:** target vendored `cubecl 0.11.3`, Burn `0.22.1` optional in future. Do not fetch new dependencies without need.

## 9. Progress tracking

Use a **new CubeCL-specific** tracker; do not subtract progress from completed TensorRT work or pretend CubeCL is included in the earlier 97.7% feature counter. Track architecture, kernel coverage, graph execution, generation, audio, parity, and Windows operability separately. Do not change the old Fish Speech F01–F22 denominator for adding a new backend.
