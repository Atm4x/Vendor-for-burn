# Stage 102 — profile CUDA kernel behavior before changing convolution algorithms

## Purpose

Stage 97 identifies CUDA convolution and GEMM as expensive, but mixed precision, unequal model stages, unequal capture windows and un-attributed launches invalidate a direct total-kernel-time ratio. The goal of Stage 102 is **evidence**, not an unmeasured NHWC/Winograd optimization.

The Windows CI builds `tts-with-rvc-rs-stage102.exe` and `stage102-conv2d-probe.exe` and runs static tests. CI does **not** execute CUDA work, Nsight Systems or Nsight Compute.

### Matched Conv2d (recommended first)

With NVIDIA Nsight Compute, Nsight Systems, CUDA PyTorch and `nvToolsExt64_1.dll` installed and visible on PATH, unpack the CI artifact and run:

```powershell
.\Run-RVC-Stage102-Matched-Conv2d.ps1 -PythonExe 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe' -Runs 12 -NcuLaunchCount 3 -Profiler both -Precision fp32 -Layout nchw
```

The workloads use the **same synthetic FP32 input and filter values**, group count, stride/padding, channels and spatial dimensions. Each implementation synchronizes after the operation. The JSON comparison validates the input contract, four output sample probes and mean value before reporting per-call elapsed-time ratios. This is the correct first attempt at an **apples-to-apples FP32 NCHW** benchmark. It still measures Python/Burn submission overhead as well as GPU execution, so use the `.nsys-rep` and `.ncu-rep` to determine GPU time and hardware bottlenecks.

The probe includes four cases: dense 3×3 32→32 64×64 batch 2; dense 3×3 64→64 48×48 batch 1; pointwise 1×1 16→32 80×64 batch 2; and depthwise 3×3 64 groups. These are deliberately synthetic, **not verified RMVPE production shapes**. Refine shapes from the resulting NVTX/NSYS trace before extrapolating performance to RVC.

Additional PyTorch `-Precision fp16` or `-Layout nhwc` sweeps are *not* numerically matched to the current Rust FP32 NCHW probe and will be marked as unmatched. They may still show the speed contribution of Tensor Cores and channels-last.

The matched runner writes `RVC_STAGE102_MATCHED_CONV2D_RESULTS.zip` containing JSON, NCU CSV/reports and NSYS reports; it writes **no WAV**.

### Full inference CUDA profiler (separate comparison)

```powershell
.\Run-RVC-Stage102-Nsight.ps1 -PythonExe 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe' -OriginalRoot 'D:\Projects\tts-with-rvc' -InputWav '.\embed\MitaKind.wav' -Profiler both -Implementation both -TorchPrecision fp16 -Runs 2 -NcuLaunchCount 3 -CoreFrames 300 -TensorBatchSize 2
```

The script runs the unchanged upstream `tts-with-rvc` pipeline through a small Stage 102 adapter and the Rust Stage 102 binary. NVTX ranges are named `stage102_torch_measured/` and `stage102_rust_measured/`. Rust additionally marks `stage102_rust_hubert`, `stage102_rust_rmvpe` and `stage102_rust_nsf` inside its warmed pass. `ncu` collects only a small number of selected convolution launches; do not run a full set on all 10k+ launches.

This full pipeline experiment is **not matched**: PyTorch defaults to FP16 whereas Rust defaults to FP32; chunk construction, resampling, normalization, index retrieval and generated sample count may differ. Resulting GPU timing ratios are diagnostic, not acceptance evidence. Record actual batch/shape/dtype/layout/kernel dispatch before claiming a 1.2× target. The full-script PyTorch reference is derived from the upstream Stage 79 harness; its model and pipeline sources are not modified.

The full runner generates a temporary output WAV only because the existing Rust CLI requires one. It does **not** include the WAV in `RVC_STAGE102_NSYS_NCU_RESULTS.zip`. The PyTorch Stage 102 adapter hashes its output samples in memory and never writes WAVs.

### Prerequisites and caveats

- `ncu --version`, `nsys --version` and `nvidia-smi` must work. Nsight Compute GPU performance counters require the NVIDIA driver to permit access; if `ERR_NVGPUCTRPERM`, enable performance counters or use an appropriately authorized environment. Do not silently skip metrics.
- Rust NVTX uses optional `nvToolsExt64_1.dll` dynamically loaded when `RVC_NVTX=1`. `nvtx-status` must report `nvtx_available=true` before the run. DLL is not bundled into the executable; install NVIDIA's supplied runtime rather than copying an arbitrary binary.
- All NVTX labels bracket *host launch submissions*; sub-stage NVTX ranges should not be treated as synchronized per-stage GPU latency. For Nsight Compute, kernel replay perturbs timing; use Nsys and warm uninstrumented elapsed time for overall throughput.
- Kernel-name regexes are intentionally conservative (`direct_conv2d_kernel` on Rust and cuDNN/CUTLASS on PyTorch). If no kernels match, the runner fails rather than reporting 0 ms. Inspect kernel lists to select representative launch/shape pairs.
- Do not interpret mixed-FP16-vs-FP32 performance as a backend regression without a precision-matched A/B. No performance claim can be made without RTX 4060 results.

Reference for NCU CLI filters, NVTX, report export and sections: https://docs.nvidia.com/nsight-compute/NsightComputeCli/

### Next experiment after results

Map the largest kernel groups to their tensor shapes and compute/memory metrics. If reduced memory transactions and Tensor Core utilization can be demonstrated at equal precision, test persistent NHWC layout and separate FP16 execution islands. Keep Winograd experimental until GPU data shows it wins on actual RMVPE shapes and maintains F0 parity.
