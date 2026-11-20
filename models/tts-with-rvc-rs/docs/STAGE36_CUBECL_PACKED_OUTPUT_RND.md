# Stage 36 — cross-runtime packed-output-channel CubeCL Conv1d

Date: 2026-11-10. Experimental research; production dispatch is unchanged.

## Goal

Improve RVC NSF Conv1d computation on all CubeCL runtimes by changing the **common kernel arithmetic and memory layout**, rather than adding CPU-only SIMD intrinsics or a new parallelism layer. CubeCL CPU JIT was used as the available executable validation environment. No NVIDIA/ROCm/WGPU hardware was present; their performance is **unmeasured**.

## Design and equivalence

Original weight shape in CubeCL Direct: `[Cout, K, CinGroup]`. The inference plan creates device-resident weights of shape `[Groups, K, CinGroup, CoutGroup]` with

`Wpacked[g,k,ci,oc_local] = W[g*CoutGroup+oc_local,k,ci]`.

This is a permutation (a bijection) of weight entries. Each output position remains the same sum over `(k,ci)` for its group, with possibly different floating-point accumulation order. No weights are dropped, no low-rank approximation is introduced, and the number of mathematical multiplications is unchanged. The kernel reads `Vector<E,NOut>` weights contiguous across output channels and broadcasts one input value per output vector. Output accumulators remain vector-valued throughout the entire reduction, with no per-input-channel extract/insert, im2col or GEMM. Two adjacent output times share a loaded weight vector.

The packing step is itself a **CubeCL kernel** (`pack_weights.rs`) and therefore runs on the same device as Conv1d; a model may cache the packed tensor after initial load. Inference runtime measurements exclude the one-time packing cost, as appropriate for a resident model. Changes to weights require re-packing.

The shared source is backend-independent CubeCL DSL. Compilation and runtime correctness on CubeCL CPU **do not prove CUDA, ROCm, or WGPU compatibility or speed**. GPU backend compilation, numerical verification and performance autotuning remain required.

## Measured performance (CubeCL CPU JIT / LLVM 23, CPU release, FP32, 2 threads)

All variants receive numerically equivalent deterministic inputs and weights. `B=1`, `groups=1`, stride=1, zero padding chosen for same-length output. Measured times include forward + host readback; original weight input and packed tensor are resident before timings.

| NSF-like shape | Direct | Output-packed | Ratio Direct/Packed |
|---|---:|---:|---:|
| Cin=256 Cout=256 K=3 L=240 | 1.064 ms | 0.721 ms | 1.48x |
| Cin=128 Cout=128 K=11 D=5 L=2400 | 8.737 ms | 5.440 ms | 1.61x |
| Cin=32 Cout=32 K=11 L=9600 | 2.531 ms | 1.705 ms | 1.48x |

Other runs with the same benchmark showed variation, including Direct 5.631 ms vs Packed 4.632 ms for K11/D5; these results are not device-independent speedups. Packed-vs-Direct maximum difference across the three shapes: 8.1e-6.

## Correctness and runtime coverage

The common CubeCL packing kernel was compiled and **executed** by LLVM CPU JIT. Its output was exactly equal to an independent CPU permutation, including grouped/depthwise configurations. The packed convolution and CubeCL Direct were numerically compared on the following additional geometries (max difference across checks: 6e-8):

- `B2 groups2 CinGroup3 CoutGroup2 K3 stride2 dilation2 padding3 Lin19 Lout11`;
- `B1 groups8 CinGroup1 CoutGroup1 K5 stride1 dilation1 padding4 Lin17 Lout21`;
- `B2 groups2 CinGroup4 CoutGroup4 K3 stride3 dilation1 padding2 Lin21 Lout8`.

These checks include group boundaries, odd output lengths, depthwise, stride, dilation, and padding. The standalone harness currently tests FP32 with no bias. Full dtype, bias, non-finite, output-padding, model-level parity, and GPU runtime coverage are **not complete**.

## Limitations and further R&D

- General convolutions need performance plans by geometry/device; a universal arithmetic kernel is not automatically the fastest on all backend architectures. No silent fallback is introduced: the new launcher is an explicit experimental operation.
- Weight packing costs one extra device allocation of `num_weights * dtype_size` bytes if both original and packed tensors are kept. It can replace the original tensor in inference-only models.
- The packed kernel cannot use matrix/Tensor Core acceleration in its present form. On CUDA it must be compared with cuDNN/CubeCL matmul-based algorithms; there is no guarantee of superiority.
- `u32` shapes/indices and strides need overflow validation; special cases with empty channels and non-finite padding need expanded tests before production activation.
- Benchmark CPU results do not establish performance on Apple ARM/NEON or NVIDIA GPUs.

## Reproduce

Standalone benchmark: `research/stage36-cubecl-packed` with Rust 1.98.1, offline vendored dependencies, and Tracel LLVM 23.1.0-4 for CubeCL CPU. Set `CARGO_TARGET_DIR` to an existing cached target path and run `cargo run --manifest-path research/stage36-cubecl-packed/Cargo.toml --release --offline`. CPU output is deliberately marked as CPU-only evidence. Keep all GPU tests separate.
