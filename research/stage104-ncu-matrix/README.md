# Stage 104 — complete NCU RVC profiling matrix

This is a **profiling stage**, not a new numerical optimization. It extends the Stage 103 2×2 register-reuse experiment with a bounded, reproducible **all-block** GPU capture plan.

## Quick start on Windows RTX 4060

The single Windows CI artifact `rvc-stage104-ncu-matrix-windows-x64` includes `stage103-conv2d-probe.exe`, `tts-with-rvc-rs-stage104.exe`, Python helpers and `Run-RVC-Stage104-NCU-Matrix.ps1`. The runner adds `nvToolsExt64_1.dll` from the supplied PyTorch virtual environment to `PATH` if found. Run PowerShell **as administrator** when NCU counters require it (`ERR_NVGPUCTRPERM`). Existing Nsight Compute 2025.1.0 is sufficient.

First run a **small micro-kernel matrix** requiring no model files or WAV:

```powershell
.\Run-RVC-Stage104-NCU-Matrix.ps1 -Scope micro `
  -PythonExe "D:\Projects\tts-with-rvc\.venv\Scripts\python.exe" `
  -NcuLaunchCount 3
```

This captures four matched Conv2d shapes for Direct, register-reuse and PyTorch FP32 (12 NCU reports). Full output parity is checked for dense 3×3 reuse. The zero-asset dry-run is `-Scope all -DryRun`.

Next **profile all blocks** with a short temporary WAV prefix; copy existing model files and original WAV to `embed/`, as for prior stages:

```powershell
.\Run-RVC-Stage104-NCU-Matrix.ps1 -Scope pipeline `
  -PythonExe "D:\Projects\tts-with-rvc\.venv\Scripts\python.exe" `
  -ProfileSeconds 12 -PipelineStrategies direct -NcuLaunchCount 3
```

For one combined archive, use `-Scope all`. To compare production alternatives, set `-PipelineStrategies 'direct,reuse,quad,gemm,tf32_batched,winograd'` with `-Blocks rmvpe`; to diagnose different kernel families, set `-Families 'all,conv,gemm,recurrent,elementwise'`. The matrix builder refuses over `-MaxTargets` (default 60). Many full-RVC captures with `-MetricSet full` can take hours; start with the default balanced sections, then use full for only identified outliers.

### NVTX blocks

`audio_h2d`, `hubert`, `hubert_readback`, `rmvpe`, `retrieval_pitch`, `encoder`, `latent`, `flow`, `nsf_prepare`, `nsf`, `nsf_readback`, plus `rmvpe_unet`, `rmvpe_gru`, `rmvpe_head`, `nsf_upsample`, `nsf_noise_conv`, `nsf_residuals`.

A subset can contain **zero GPU kernels** (for example CPU retrieval or readback); the runner records `no_kernels` rather than inventing profiling data. `--nvtx-push-pop-scope process` handles CubeCL's cross-thread dispatch. Pipeline captures additionally restrict to the warmed `stage102_rust_measured` parent range. Micro captures restrict to a single shape using an optional case argument in Stage 103's probe.

### Sections and limitations

Default: `SpeedOfLight`, `LaunchStats`, `Occupancy`, `MemoryWorkloadAnalysis`, `ComputeWorkloadAnalysis`, `WarpStateStats`, `SchedulerStats`. `-MetricSet full` invokes NCU's `--set full`. No fixed kernel-name filter is applied to `all`: early helper copies can consume a small launch quota. Use `-Families conv` and focused blocks when measuring specific compute kernels. Device kernel replay overhead invalidates ordinary latency comparisons; use `micro-parity.json` and the uninstrumented `micro-{direct,reuse,torch}.json` for that purpose.

`RVC_STAGE104_NCU_MATRIX_RESULTS.zip` contains **only** `.ncu-rep`, optional `.nsys-rep`, JSON metadata, and logs. It excludes all WAVs, checkpoints, Hubert/RMVPE weights and FAISS indexes. There is no CLI CSV-export step because NCU 2025.1.0 previously returned `bad conversion` despite a valid GUI report; inspect `.ncu-rep` in Nsight Compute GUI and export CSV there if needed.

The Windows CI **compiles and statically validates** the profiler; it has no RTX 4060 and cannot validate the GPU timings or all NCU NVTX expressions. Full Rust/PyTorch inference comparisons also differ in preprocessing/chunking, so only the matched Conv2d microbench is a shape- and FP32-matched test.
