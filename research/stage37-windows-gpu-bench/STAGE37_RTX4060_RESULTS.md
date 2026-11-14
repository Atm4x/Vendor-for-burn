# RVC NSF Conv1d — Stage 37 GPU benchmark report

Date: 2026-11-10. Target: RTX 4060 (Windows x64). Status: **real-device operator tests PASS; full RVC/GPU-only measurements not yet performed**.

## Purpose and provenance

The experimental common CubeCL implementation is compared with stock CubeCL Direct and PyTorch 2.11.0+cu128 (cuDNN 9.19). The results below were supplied by the operator running the Stage 37 Windows artifact on an NVIDIA GeForce RTX 4060 and reproduced from the two generated `torch-cudnn.json` variants and console output. These numbers are **not CI GPU measurements**. CubeCL CUDA and WGPU each returned status `pass` and exit code 0.

The three cases use deterministic synthetic FP32 inputs and weights shaped like NSF ResBlocks; they do **not** use CrazyMita checkpoint activations. The operation measured is dispatch + allocation + synchronous CPU readback, warmup=3, iterations=10 (median). Packing is performed once outside the repeated forward loop and its measurement includes verification readback and potentially first-time JIT/initialization. PyTorch uses a similar synchronous CPU readback, but execution harness overhead differs; the precise cuDNN algorithm is not verified by profiling. CUDA and WGPU adapter identity may differ if the machine has multiple GPUs.

## Implemented optimizations (Stages 28–37)

1. **Burn Flex native Conv1d prototype (Stage 28).** Eliminated pseudo-3D expansion in the experimental path, with grouped/depthwise and GEMM-per-kernel. Addressed `f16` partial-sum overflow, NaN-in-padding behavior, and zero-input-channel safety. Microbenchmarks were positive; no production-wide promotion.
2. **Burn Flex Conv2d R&D (Stages 29–31).** NCHW tiled implicit-GEMM without legacy Conv3d fallback, analytical spatial bounds, tiled Rayon scheduling, and reduced depthwise scheduler overhead. On selected release RMVPE shapes improved up to ~3x against the legacy pseudo-3D path; 10.1s full RVC produced matching PCM. This does not establish GPU speedup.
3. **NSF residual-branch parallelism (Stage 32).** Parallel evaluation of independent ResBlock branches preserved output in the tested release path. Showed ~1.29x for NSF at 128 frames / 4 threads, but no stable improvement at 128 frames / 2 threads; fundamentally a CPU scheduling optimization.
4. **Arithmetic/AVX-512 Conv1d research (Stages 33–34).** Tested Winograd-like transformed K3 (benefit on selected short layers), segmented K11 (not consistently beneficial), low-rank factorization (insufficient compression), and output-stationary direct AVX-512 kernel (11–18% gains on three real model shapes in isolated tests). These CPU-specific experiments were not promoted as cross-device kernels.
5. **Common CubeCL TemporalPair (Stage 35).** Shared Conv1d kernel processing adjacent temporal positions without GEMM. Compiled and mathematically tested; initial runtime on CPU JIT was 2.8–3.5x slower than stock Direct, so it was not adopted.
6. **Common CubeCL packed-channel Conv1d (Stage 36).** A device-side weight-packing kernel rearranges weights for contiguous output channels. The forward kernel accumulates across output channels in a vector with device-resident packed weights, avoiding im2col and separate GEMM. The weights are intended to be packed once per model load. This is shared CubeCL source used by CUDA and WGPU, not AVX-512 code.
7. **Windows GPU benchmark tool (Stage 37).** Built separate CubeCL CUDA/WGPU executables, a Python/cuDNN reference, diagnostics, and JSON reports. Windows PowerShell 5.1 runner incorrectly treated normal native stderr progress as a terminating error; direct EXE invocations succeeded. Fix the runner before its next release.

## Real GPU results (milliseconds; lower is better)

| Shape (Cin→Cout, L, K, dilation) | CubeCL CUDA Direct | CubeCL CUDA Packed | CUDA Direct/Packed | CubeCL WGPU Direct | CubeCL WGPU Packed | WGPU Direct/Packed |
|---|---:|---:|---:|---:|---:|---:|
| K3 (256→256, 240, K3, D1) | 0.3267 | **0.2016** | **1.62x** | 0.3558 | **0.2951** | **1.21x** |
| K11/D5 (128→128, 2400, K11, D5) | 2.4674 | **0.8525** | **2.89x** | 2.5929 | **1.1398** | **2.27x** |
| K11 (32→32, 9600, K11, D1) | 0.8541 | **0.5254** | **1.63x** | 0.8924 | **0.6198** | **1.44x** |

Across all six Direct/Packed comparisons the benchmark reported `max_abs_error = 0`, `max_normalized_error = 0`, `passed = true`. These checks compare **CubeCL Direct and Packed** with entire output arrays, not the entire output against PyTorch. All numerical results are specific to the supplied deterministic inputs.

One-off weight-pack timings (CUDA/WGPU, milliseconds): K3 290.0665/25.4745; K11/D5 2.0555/0.8642; K11 0.4173/1.1607. The extreme K3 cost may contain first-use compilation and initialization; do not treat it as steady-state pack latency.

## PyTorch/cuDNN comparison, FP32 with TF32 explicitly disabled

PyTorch: `torch.backends.cudnn.allow_tf32 = False`, `torch.backends.cuda.matmul.allow_tf32 = False`; cuDNN **enabled**, v9.19.0; PyTorch 2.11.0+cu128. These were separate repeated-process measurements from CubeCL. The earlier TF32-permitted run is retained only as context, not as strict-FP32 baseline.

| Shape | PyTorch TF32 allowed | PyTorch strict FP32 | CubeCL CUDA Packed | Packed/PyTorch strict latency |
|---|---:|---:|---:|---:|
| K3 | 0.11915 | **0.18395** | 0.20160 | **1.10x** |
| K11/D5 | 0.51700 | **0.43870** | 0.85250 | **1.94x** |
| K11 | 0.39550 | **0.28370** | 0.52540 | **1.85x** |

Strict FP32 PyTorch probes are now much closer to CubeCL's probes. Max absolute deviation across three probes per shape: K3 4.16e-7, K11/D5 1.43e-6, K11 3.76e-7. **This is not full-tensor PyTorch numerical parity**. The reported PyTorch output sums also differ slightly from CubeCL; the comparison requires exporting complete arrays for robust RMSE/ULP checks.

Note: TF32 enabled timings are not uniformly lower (K11 measurements were slower). Variation in cuDNN algorithm selection, execution conditions and short sample counts must be investigated; TF32 cannot be declared the sole cause from these runs.

## What these results prove

- A *single shared CubeCL algorithm* accelerated Conv1d against stock CubeCL Direct on both CUDA and WGPU for all three tested shapes.
- The new algorithm does **not** yet beat PyTorch strict FP32 on this GPU. K3 is close (~10% slower measured latency); K11 remains ~85–94% slower.
- No inference-level performance claim is justified: no native CUDA RVC end-to-end, no full NSF on the GPU, no kernel-only GPU timing, no confirmed same-device WGPU adapter identity, no cuDNN algorithm trace.
- No production RVC dispatch has been switched to the experimental CubeCL kernel.

## Next research gates

1. Fix Windows runner's `stderr`/PowerShell exception bug and record driver, adapter name and TF32/precision toggles in all JSON.
2. Measure **GPU kernel-only** latency with CUDA events or device timestamps, separate from allocations and CPU copies; measure pack cold, warm and amortization across resident inference calls.
3. Export and compare full FP32 output tensor against PyTorch with TF32 disabled, including max error, RMS error and relative tolerance.
4. R&D the **shared CubeCL K11 kernel** (output-position tiling, lane/coalescing and reuse of weights/input) without CPU-only intrinsics; preserve CUDA, WGPU, ROCm and CubeCL CPU implementation path.
5. Benchmark multiple grids/block sizes and B>1 on actual checkpoint tensors; only then integrate a resident packed-weight inference plan in NSF and check end-to-end speed and memory.

Sources: user's Windows CUDA/WGPU console/JSON output and `torch-cudnn.json` with and without TF32, plus implemented source and previous stage technical notes in this PR. No checkpoint, index or audio uploaded.
