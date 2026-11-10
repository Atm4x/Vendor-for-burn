# RVC CubeCL Windows GPU Benchmark — Stage 37

This is a **standalone Conv1d kernel benchmark**, not full voice conversion. No private weights or WAV required. It exercises the three RVC NSF layer geometries with deterministic FP32 tensors and compares stock CubeCL Direct with the experimental device-packed output-channel SIMD kernel (two temporal positions) and the independent four-position TemporalQuad variant. It optionally compares against `torch.nn.functional.conv1d` with cuDNN enabled.

## Running

Unpack the Windows Actions artifact onto a local Windows x64 PC with a working GPU driver. In PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-Benchmark.ps1
```

The script discovers CUDA and WGPU adapters through actual kernel execution, records `nvidia-smi -q`, `nvcc --version` and Python/PyTorch information, then creates `RVC_GPU_BENCH_RESULTS.zip` to send back. If Python/PyTorch is missing, the CubeCL tests still run. The wrapper keeps native stdout and stderr in separate log files to avoid PowerShell 5.1 NativeCommandError on ordinary progress messages. No CUDA Toolkit installation is attempted automatically. `nvcc` absence **does not necessarily prevent CUDA runtime execution**: the NVIDIA driver and the compiled program's runtime dependencies matter more. CUDA 12.8 Toolkit and the PyTorch wheel's bundled CUDA version are separate from the CUDA driver API version.

Optional cuDNN reference (install a supported pinned wheel **only on explicit request**):

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-Benchmark.ps1 -InstallTorchCu128
```

To test a single backend: `-Backend cuda` or `-Backend wgpu`; use `-Warmup 5 -Iterations 20` for longer measurements.

## Methodology and limitations

- Each Rust repetition includes kernel dispatch, synchronous CPU readback and output buffer allocation. First three iterations are warmup. Direct and packed order alternates. This measures a **synchronous end-to-end operation**, not isolated GPU kernel latency.
- Device-side weight packing is measured separately and excluded from repeated inference; weights are packed once and read back to verify correctness. This matches the model-resident use case, but `pack_ms` includes a verification readback.
- PyTorch uses the same integer-generated tensors; `conv1d` is run with cuDNN enabled when CUDA is available, and the output is copied back to CPU for an analogous latency comparison. Whether a given convolution actually used a cuDNN kernel depends on PyTorch's internal dispatch; confirm with a profiler before claiming a specific cuDNN algorithm.
- Numerical parity: Direct vs Packed and Direct vs Quad over entire output arrays. With the optional `--parity-dir` input, each runtime saves little-endian full FP32 arrays and `compare_tensor_dumps.py` compares every element against strict-FP32 PyTorch. The JSON probes alone are not full PyTorch parity.
- Performance on Windows GitHub Actions is **not** representative of user GPU: the hosted runner is only used to compile and check `--help`, not to assert CUDA availability.
- Different backends may choose different physical adapters. WGPU might choose a different device than CUDA on multi-GPU machines; include the diagnostic files.
- No conclusion about the CUDA/WGPU GPU performance is possible before a real-device run.

## Artifacts

`rvc-cubecl-wgpu.exe` (Windows WGPU), and `rvc-cubecl-cuda.exe` when the CUDA build succeeded; `Run-Benchmark.ps1`, `torch_cudnn_bench.py`, `merge_reports.py`, and `build-status.txt`. If the CUDA binary is absent, use the logs/build status and report it as a build blocker, not a benchmark regression.

## Stage 38 experiments

The CLI now reports `quad_ms` and `speedup_pair_over_quad` in addition to the existing Direct/Packed fields. Pass `--parity-dir <directory>` to export full outputs for all three variants. The PowerShell runner enables these dumps and runs PyTorch with TF32 **disabled** by default; use the standalone PyTorch script `--tf32 allow` only for a separate TF32 comparison. Quad is **not** a claimed improvement until actual CUDA/WGPU measurements confirm its benefit. See [STAGE38_TEMPORAL_QUAD_RND.md](STAGE38_TEMPORAL_QUAD_RND.md).
