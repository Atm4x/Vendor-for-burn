# Stage 35 — shared CubeCL Conv1d temporal-pair experiment

## Objective

Investigate a shared CPU/GPU CubeCL convolution kernel rather than porting CPU AVX-512 intrinsics or hiding slow operations behind more Rayon threads. The primary workloads are NSF ResBlock Conv1d in RVC v2: K3/C256/L240, K11/D5/C128/L2400, and K11/C32/L9600. This experiment does not change any production inference dispatch.

## Relevant source inspection

- `cubek-convolution-0.3.1/src/kernels/forward/direct.rs`: current direct forward kernel assigns one output position/channel vector to a work item, loops over spatial kernel positions, then input-channel vectors. It reuses an input vector across output-channel lanes, but distinct temporal positions issue independent weight-vector loads and accumulations.
- `burn-cubecl-0.22.1/src/kernel/conv/base.rs`: `ConvStrategy` can dispatch Direct/ImplicitGemm/Autotune (autotune behind feature); the default without autotune is Direct.
- `cubecl-cpu-0.11.3`: CPU runtime JIT requires LLVM 23.1.0 via `tracel-llvm-bundler-23.1.0-4`.

## Candidate: temporal-pair output-stationary kernel

For a 1D cross-correlation, per output channel and group:

\[
 y[b,t,o] = \beta_o + \sum_{k=0}^{K-1}\sum_{c=0}^{C_g-1}
   \mathbf{1}_{0\le ts+kd-p<L}
   W[o,k,c]\,X[b,ts+kd-p,gC_g+c].
\]

For each work item, let `t0=2q`, `t1=2q+1`. Keep **two** output accumulators in registers, and load the weight vector `W[o,k,c..c+V]` once for the two positions. Each output uses its own bounds test and its own input vector, but shares the filter vector.

**Equivalence lemma:** For each legal `t`, there exists exactly one pair `q=floor(t/2)` and one lane `t mod 2`. The transformed kernel traverses the same `(k,c)` pairs in the same order for each lane as the direct kernel; therefore their finite-valued scalar reference outputs are bitwise identical, including grouped/dilated/padded and odd output lengths. The final odd lane is skipped. Production CubeCL backend floating-point reproducibility still requires execution tests because compilers can fuse or reassociate operations.

**Work bound:** For a fixed output-channel vector, old direct computes `L_out` temporal work items and logically loads `L_out*K*ceil(C_g/V)` filter vectors; paired direct uses `ceil(L_out/2)` work items and logically loads `ceil(L_out/2)*K*ceil(C_g/V)` filter vectors. For even output length, this halves filter load instructions and halves output work-item count **without reducing the arithmetic count**. Real DRAM traffic may change less due to caches, and larger accumulators can reduce occupancy.

## Implementation

- Added `vendor/cubek-convolution-0.3.1/src/kernels/forward/direct_temporal_pair.rs` and `launch_temporal_pair`, with the same CubeCL tensor and launch abstractions as `launch_direct`.
- Validates nonzero grouping, vector alignment, zero-sized output.
- Uses output-channel vectorization and per-lane pair accumulators. Preserves existing direct semantics of skipping out-of-bounds products.
- Does not enable the kernel in `burn-cubecl` production dispatch. This is a research candidate, not a proven speedup.

## Evidence

- `cargo check` of the shared `cubek-convolution` crate in an isolated manifest **PASS**, offline, Rust 1.98.1, with no CPU JIT linking.
- Independent Rust reference comparison: **2592** combinations across batch, groups, input/output channels, strides, dilations, padding, kernel sizes, odd/even output lengths: **PASS, bit-identical** in f32 scalar evaluation.
- Weight-load identity and odd-pair count: **PASS**.
- LLVM 23 CubeCL CPU: **BLOCKED**. Background Cargo process failed in `tracel-llvm-bundler` because it could not resolve the GitHub release download host; LLVM 23 artifact was not in the offline cache. LLVM 19 on this host is not a binary substitute. No JIT kernel was executed.
- GPU performance and parity: **NOT TESTED**. There is no accessible CUDA/ROCm/Vulkan device in this container. Avoid asserting a runtime speedup.

## Next gates

1. Supply/offline-cache the exact `tracel-llvm v23.1.0-4` Linux bundle and checksums, verify vendor integrity, and run CubeCL CPU JIT parity versus the existing direct kernel.
2. Benchmark shared CubeCL direct vs paired kernel on the three actual NSF shapes in a single process with identical packed inputs/weights, including JIT cold/warm timing.
3. Inspect LLVM-IR/PTX/spir-v register allocation and occupancy. Compare temporal tiles 1, 2, 4; test whether weight-load reuse beats doubled register pressure.
4. Validate f16/bf16, non-finite padding behavior, vectorized groups, extreme dimensions, and the exact model checkpoint before any production dispatch.
5. Repeat against CUDA/cuDNN and WGPU on target hardware. The same CubeCL implementation can compile for these runtimes but must be tuned and tested independently.
