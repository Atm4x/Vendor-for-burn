# Stage 41 — Pipeline/layout R&D beyond Conv1d temporal tiling

Date: 2026-11-10. Scope: **experimental CubeCL CPU JIT only**, NSF-shaped deterministic FP32 data. The goal is to profile work **around** Conv1d: residual result aggregation, tensor layout materialization, and constant-weight preparation. No production RVC dispatch is modified; no RTX 4060 execution is claimed.

## Why a different direction

Previous stages optimized arithmetic inside Conv1d. Profiling the caller shows additional possible costs:

1. `models/tts-with-rvc-rs/src/decoder.rs` constructs a zero result, adds three ResBlock outputs, and divides by three. Eagerly executed as four elementwise kernels, this reads/writes full tensors repeatedly.
2. `vendor/burn-cubecl-0.22.1/src/kernel/conv/base.rs` permutes NCL to NLC via metadata; `.../kernel/conv/direct.rs` invokes `into_contiguous_aligned` when the channel stride is not 1. **The `permute` itself is not a physical copy**. Data movement depends on the actual storage strides and backend's layout capabilities.
3. Experimental packed-weight creation previously used `ci` instead of `ci * source.stride(2)`, so the pack kernel could silently misread the original `[Cout,Cin,K]` Burn weight after the view becomes `[Cout,K,Cin]`. Fixing this is both a correctness requirement and an opportunity to bypass an intermediate layout materialization.

## Reproduction and machine

- Machine: virtualized AMD EPYC 9V74, 5 logical CPUs exposed, KVM. **Single pinned logical CPU** (`taskset -c 1`) for the reproducible measurements below. These are not representative of a consumer desktop CPU or any CUDA device.
- Rust 1.98.1; LLVM 23.1.0; CubeCL 0.11.3; offline Cargo vendor; release mode.
- Run: `cargo build --locked --offline --release --no-default-features --features cpu --bin bench_pipeline_stage41` from `research/stage37-windows-gpu-bench`, with `LLVM_SYS_231_PREFIX` set to the supplied LLVM root.
- Measure: `taskset -c 1 target/release/bench_pipeline_stage41 --warmup 6 --iterations 20 --json result.json`; two separate processes. Results below are the **arithmetic mean of two reported warmed medians** (not a cross-device speed comparison). Explicit `client.sync()` before stopping timer. Repeated forward runs reuse already allocated device tensors; source transfers and initial compilation excluded except where specifically stated.
- Input shapes: K3 `[1,256,240]`, `[256,256,3]`, D1; K11/D5 `[1,128,2400]`, `[128,128,11]`, D5; K11 `[1,32,9600]`, `[32,32,11]`, D1. Synthetic deterministic FP32 only.

## A. Residual output processing: 4 eager operations versus 1 fusion

Four-operation emulation: copy branch A -> accumulator, add B, add C, divide by 3. A fused CubeCL operation computes `((a+b)+c)/3` into the output directly. The comparison uses the same preallocated buffers and checks the entire result.

| Shape | 4 eager operations (ms) | 1 fused (ms) | Speedup | Full output max abs difference |
|---|---:|---:|---:|---:|
| K3 | 0.06435 | 0.04559 | 1.41× | 0 |
| K11/D5 | 0.15378 | 0.08853 | 1.74× | 0 |
| K11 | 0.15320 | 0.08737 | 1.75× | 0 |

**Interpretation:** eagerly materializing the whole residual aggregation is wasteful. This is *not yet a measured speedup for Burn's actual ResBlock graph*: Burn may use eager in-place operations or fusion, and the exact production path needs instrumenting before replacing the aggregation implementation. Our 4-operation baseline also starts by copying A instead of allocating zero and adding A, so it is an optimistic lower-cost surrogate for the current source-level pattern.

## B. Layout and Conv caller overhead

Physical conversion `[1,C,L]` NCL -> `[1,L,C]` NLC -> NCL is separate from metadata-only `permute`. A two-pass physical round-trip on CPU JIT cost:

| Shape | NCL↔NLC physical round trip (ms) | Exact round-trip |
|---|---:|---|
| K3 | 0.6520 | PASS |
| K11/D5 | 3.1371 | PASS |
| K11 | 3.1170 | PASS |

A more realistic *explicit materialization + Conv* A/B compared:

- **Baseline:** transform original NCL input -> NLC, transform `[Cout,Cin,K]` weight -> `[Cout,K,Cin]`, execute CubeCL Direct. This includes weight materialization on **every** measured call, mirroring an uncached preparation path.
- **Resident plan:** transform original input NCL -> NLC, execute QuadInterior with packed weights prepared **once** before the repeated loop. This intentionally moves immutable weight preparation out of inference; it is not a pure kernel-to-kernel comparison.

| Shape | Baseline materialize + Direct (ms) | Input materialize + resident QuadInterior (ms) | Ratio |
|---|---:|---:|---:|
| K3 | 3.4358 | 1.6023 | 2.14× |
| K11/D5 | 20.7678 | 12.3376 | 1.68× |
| K11 | 7.0030 | 4.2032 | 1.67× |

For context, with both input and weights resident in their required layouts, Direct vs QuadInterior timings were (ms): K3 **2.1418 / 1.2916**, K11/D5 **18.2864 / 10.8749**, K11 **5.4091 / 2.5556**. Therefore the pipeline gain combines the already-known kernel advantage and the changed preparation lifecycle. It **cannot be attributed entirely to the layout work**.

Direct vs QuadInterior entire output max absolute FP32 error: K3 `3.4571e-6`, K11/D5 `3.5763e-6`, K11 `1.6689e-6`; all full-array parity checks PASS.

**Important limitation:** This is an isolated synthetic convolution callpath with explicit NCL physical source and NLC materialization. Burn's actual `into_contiguous_aligned` may choose different pathways for some layouts. No full NSF model or real checkpoint was timed; don't multiply these layer numbers to claim global RVC gains.

## C. One-pass weight preparation from strided metadata

The original weight is physically `[Cout,Cin,K]`; a view `[Cout,K,Cin]` exposes strides `[Cin*K, 1, K]`. The updated common CubeCL pack kernel indexes **all three source strides**, so the view can be packed to `[Groups,K,CinGroup,CoutGroup]` without first materializing a contiguous `[Cout,K,Cin]` copy.

| Shape | Direct pack from strided view (ms) | Materialize then pack (ms) | Preparation acceleration | Full packed output |
|---|---:|---:|---:|---|
| K3 | 1.3473 | 2.3347 | 1.73× | bit-exact |
| K11/D5 | 1.2429 | 2.1460 | 1.73× | bit-exact |
| K11 | 0.1055 | 0.1636 | 1.55× | bit-exact |

This is a **model-load / warmup cost**, not a per-inference reduction when weights are truly resident. Group and mixed-dtype paths require broader validation; current checks use G=1 FP32 synthetic data.

## Validation and integration boundaries

- Whole-tensor residual fusion parity: PASS, exact on three shapes.
- Whole-tensor layout roundtrip parity: PASS, exact on three shapes.
- Packed weights against independent CPU reference: PASS, bit-exact on three shapes.
- Synthetic Direct vs QuadInterior full-tensor parity: PASS within stated floating point error.
- CubeCL CPU `release` build + execution: PASS.
- Existing GPU benchmark `cargo check --offline --locked` (WGPU and CUDA): PASS. **No GPU timings for Stage 41.**
- No modification to production `decoder.rs`, Burn conv routing or model checkpoints.
- New CPU-only runner is gated behind the `cpu` feature; Windows CUDA/WGPU artifact still tests the Stage39 arithmetic kernels and does **not** include this CPU-only pipeline benchmark.

## Architectural conclusion

The next genuine performance target is the **inference execution plan and tensor lifetime**, not more temporal tiling:

1. **Immutable model initialization** prepares/cache aligned or packed constant weights and verifies their version/dtype/backend. `forward` must never re-pack unchanged weights.
2. **Layout contract per edge** declares logical shape plus actual physical strides; materialize only when the chosen operator cannot consume the view. Prefer producer output directly in the next consumer's preferred physical layout. Do not globally assume NLC for every op: the whole decode graph must be inspected for layout crossings.
3. **Residual epilogue** combines all ResBlock outputs in one supported fused kernel, ideally without an initial zero accumulator, only after confirming Burn's actual existing optimizer does not already fuse these ops.
4. **Buffer liveness** should reuse device-resident scratch/outputs across sequential inference without unnecessary CPU round-trips. Preserve concurrency isolation and cancellation semantics.
5. Compare actual `NCL→materialize→Conv→result` against `NCL view→operator with strides` (or persistent NLC pipeline) in a full NSF-stage trace with per-operator/transfer timings. Validate total model output and steady-state latency; only then select a production dispatch policy.

These are research proposals, not deployed or validated end-to-end speedups.

## Evidence

`pipeline_pack_cpu_pinned_1.json`, `pipeline_pack_cpu_pinned_2.json` contain raw per-process results. New runner: `research/stage37-windows-gpu-bench/src/bin/bench_pipeline_stage41.rs`. Changed kernel: `vendor/cubek-convolution-0.3.1/src/kernels/forward/pack_weights.rs` plus checksum manifest. No private weights, samples, or index included.
