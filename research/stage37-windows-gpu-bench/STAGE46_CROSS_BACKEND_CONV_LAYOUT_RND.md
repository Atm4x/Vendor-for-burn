# Stage 46 — Cross-backend convolution dataflow and layout algorithm R&D

Date: 2026-11-10. Project: `Atm4x/Vendor-for-burn`, experimental PR #2. Status: experimental shared CubeCL kernel + CPU A/B; **do not switch production**.

## Decision: CUDA Graphs are an orthogonal optimization, not the primary algorithm

CUDA Graph capture/replay reduces repeated CPU-to-GPU command launch overhead and can optimize dependency submission. It **does not, by itself, change the convolution's arithmetic complexity, fuse all operations, improve memory locality, or remove repeated weight/layout conversions**. Graph capture can become valuable after fixing dataflow and repeated small-kernel launch costs. Its CUDA-only nature makes it a poor foundation for a platform-independent convolution engine.

Instead, assess the entire pipeline:

`Model tensor / layout → Burn conv NCHW↔NHWC view → potential physical contiguous copy → cubek direct kernel → output metadata/layout → activation/residual/next convolution → CubeCL scheduler → CPU/WGPU/CUDA backend`.

### Source-level bottleneck

The shared `cubek-convolution-0.3.1` direct forward kernel already supports spatial strides, but its inner channel loop assumes `stride(C)==1`. Consequently, the Burn CubeCL direct path (`burn-cubecl-0.22.1/src/kernel/conv/direct.rs`) materializes a contiguous representation when the permuted input or weight has a non-unit channel stride.

This is a **model/data-layout concern shared by Conv1d, Conv2d, Conv3d**; it is not specific to CUDA, CPU, or NSF. The cost of copying is not automatically more expensive than reading strided channels in the convolution kernel.

## Hypothesis A — avoid the pre-convolution copy and compute directly on strided views

A reversible research patch to `vendor/cubek-convolution-0.3.1/src/kernels/forward/direct.rs`:

- Propagates `stride_ic` and `stride_wc` through the existing dimension-generic (1/2/3-D) direct kernel loop, for grouped and dilated spatial convolutions.
- Computes channel locations with the correct logical strides instead of assuming a unit channel stride.
- Forces a scalar input vector size for non-contiguous channels, preventing invalid vectorized loads.
- Leaves the caller/production Burn route unchanged; the existing Burn direct path continues to materialize a contiguous input/weight as before. Experimental callers can directly use `launch_direct` with strided tensor bindings.

**This is not a selected production optimization.** It is an algorithmic counterfactual for measuring the cost of skipping data rearrangement.

## CPU measurement protocol

- CubeCL CPU JIT / tracel LLVM 23.1.0-4; Rust 1.98.1; virtualized AMD EPYC; one pinned logical CPU.
- Actual shared CubeCL direct Conv1d kernel; deterministic synthetic weights/inputs matching three NSF tensor geometries, plus two small tensors.
- Each trial: 12 warmed measurements after 5 warmups, wall-clock including CubeCL launch and completed CPU synchronization. Three independent controlled runs (A/B/C) with no concurrent compiler jobs.
- Compared `pre-materialized contiguous → Direct` (kernel cost only), `materialize input & weight each call → Direct`, and `strided view → Direct`. For the final table, the second and third modes are compared; the contiguous pre-materialized mode is available in the JSON.
- The materialization uses `cubecl::std::tensor::into_contiguous`; this is a low-level approximation of Burn's materialization, not a measured full production Burn/RVC graph.

| Shape | Materialize + Direct, median ms | Strided Direct, median ms | Ratio (copy / strided) | Finding |
|---|---:|---:|---:|---|
| Small K3, Cin 4 Cout 8 T 15 | 0.0445 | 0.0349 | **1.28×** | Strided view helps |
| Small K11 D3, Cin 4 Cout 8 T 63 | 0.0534 | 0.0433 | **1.23×** | Strided view helps |
| NSF K3, Cin 256 Cout 256 T 240 | 4.398 | 19.016 | **0.23×** | Strided view ~4.32× slower |
| NSF K11 D5, Cin 128 Cout 128 T 2400 | 22.724 | 199.544 | **0.11×** | Strided view ~8.78× slower |
| NSF K11, Cin 32 Cout 32 T 9600 | 5.977 | 43.076 | **0.14×** | Strided view ~7.21× slower |

All values above are **medians of three independent per-run medians**; raw per-run JSON files are retained, so one can inspect variance. One earlier exploratory run under concurrent compiler load was excluded from this table.

### Correctness

- Full deterministic FP32 output comparisons were performed against the same direct kernel with physically contiguous input and weights, plus an independent scalar reference for all five Conv1d shapes.
- Maximum measured absolute difference on the large NSF shapes: `1.22e-4`. With different vectorization / accumulation order, exact bitwise equality is not expected. This was below the test threshold of `4e-4`; the independent scalar reference threshold was `8e-4`.
- Separate non-contiguous-channel **Conv2d** and **Conv3d** kernel parity tests both gave `max_abs_error=0` on their small deterministic tensors.
- Separate **grouped Conv1d** test (`groups=2`) gave `max_abs_error=4.77e-7`.
- Cargo `check --release` passed on CPU, WGPU and CUDA feature configurations (Rust 1.98.1). There is **no WGPU or CUDA runtime A/B** for this candidate.
- Testing is not comprehensive for arbitrary shapes, vector widths, quantized types, f16/bf16, every padding/stride/group case, or ConvTranspose. Do not promote based on these tests alone.

## Interpretation

For small kernels, copies and allocations can dominate; reading strided data directly can help. For large NSF layers, the cost of repeated strided/scalar loads dwarfs one-time materialization. A universal "eliminate copies" policy is therefore the wrong optimization.

This experiment supports a higher-level cross-backend algorithm **without forcing a backend-specific solution**:

1. **Layout residency:** represent the logical NCHW/NCL view separately from the physical channels-last storage. Preserve preferred physical layouts across consecutive convolutions and compatible pointwise/residual operators, so downstream operations do not keep rematerializing intermediates.
2. **Weight lifecycle:** convert convolution weights once during model initialization into a layout appropriate for the shared kernel and retain them across inference calls. Account for checkpoint load time separately from steady-state latency.
3. **Cost-aware planning:** measure/estimate the one-time layout conversion, amortization over `Cout × kernel_volume` MAC reuse, and memory access patterns. Choose between direct-strided for small/lightweight shapes, prepack+direct, and shared implicit-GEMM when supported. This is a strategy decision on a common mathematical operation, not different model behavior per CPU/GPU.
4. **Operator-compatible epilogues:** consider bias, activation and residual aggregation at operator boundaries to avoid intermediate reads/writes, while preserving FP32 order and reference parity. This is independent of CUDA Graph capture.
5. **Backends only as lowerings:** CPU/WGPU/CUDA can tune launch geometry, vector width and register tiling, but the layout/packing lifetime and graph-level mathematical plan should be defined above them.

Work estimates for direct ConvNd at fixed tensor geometry: approximately `B × spatial_output_volume × Cout × (Cin/groups) × kernel_volume` MACs. Materialization is O(input + weight size) data movement. For repeated inference and high channel fan-out, paying the layout cost once dominates attempting to use strided data for every MAC.

## Next experiment (Stage 47)

Build a two-/three-convolution **persistent-layout** pipeline benchmark with the same intermediate activation and identical FP32 output:

- Baseline: repeated NCL↔NLC physical conversions and weight packing at each invocation.
- Candidate: prepack static weights once; keep intermediate tensors in NLC physical storage across Conv → activation → Conv and residual edges, materializing only at true external boundaries.
- Measure separately initial model preparation, warm forward, all intermediate allocations and GPU/CPU synchronization. Compare full-model NSF and low-level microbenchmark.
- Validate CPU first and then reuse exactly the same CubeCL source on WGPU and CUDA. Only after the algorithm is stable should CUDA Graph replay be measured as an **additional**, separately attributed benefit.

## Reproduction

In the experiment source tree, run the `bench_conv_stride_stage46` target from `research/stage37-windows-gpu-bench` with Rust 1.98.1, LLVM 23.1.0 and the CPU feature. The benchmark's small deterministic Conv2d/3d/grouped cases are run automatically. CPU results are written to stdout, progress/parity logs to stderr.

Source files: `vendor/cubek-convolution-0.3.1/src/kernels/forward/direct.rs`, its `.cargo-checksum.json`, `research/stage37-windows-gpu-bench/src/bin/bench_conv_stride_stage46.rs`, and `evidence/stage46-cpu/controlled_{a,b,c}.json`.

**Production/default Burn strategy is unchanged.** This stage tests a cross-backend algorithmic hypothesis and records why it is rejected for large NSF shapes. Do not advertise overall inference acceleration based on the small-shape wins.
