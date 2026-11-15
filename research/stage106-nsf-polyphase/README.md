# Stage 106 — CubeCL-native NSF dataflow

## Purpose

Replace the two measured slow generic convolution paths in NSF with specialized operators:

- **Upsample**: output-gather polyphase `ConvTranspose1d` supporting groups and unit dilation. A thread computes two temporal positions with the same stride phase; every weight vector updates both accumulators. The model caches its immutable channel-last weight representation.
- **Harmonic noise injection**: `Conv1d` specialized for `Cin=1,groups=1`. Each thread computes two neighboring output positions with contiguous vectors of output-channel weights, cached at model loading.

Both compute FP32 multiply/add without TF32 approximation and allocate only the output tensor (weight prepacking is one-time). Neither creates an `im2col`, `col2im`, or zero-inserted input buffer. They are not Winograd and should not be generalized to arbitrary convolutions without constraints.

## Benchmarking on RTX 4060

Obtain the Windows artifact `rvc-stage106-nsf-polyphase-windows-x64` from the single active GitHub workflow. Extract it and run PowerShell as administrator if GPU counters are restricted:

```powershell
.\Run-RVC-Stage106-NSF-Dataflow.ps1 -Profiler none -Backend cuda -Runs 12
.\Run-RVC-Stage106-NSF-Dataflow.ps1 -Profiler ncu -Backend cuda -Runs 12 -NcuLaunchCount 3 -PythonExe "D:\Projects\tts-with-rvc\.venv\Scripts\python.exe"
```

By default, this tests `upsample8`, `upsample10`, `noise8`, `noise40`. Select other cases through `-TransposeCases` and `-NoiseCases`. Every variant checks every output element against the same Burn reference, with `max_abs <= 1e-4`. `RVC_STAGE106_NSF_DATAFLOW_RESULTS.zip` contains all generated reports, logs, JSON and optional raw `.ncu-rep`; it contains no WAV, checkpoint or model assets.

## End-to-end opt-in

Set `RVC_NSF_POLYPHASE=1` and `RVC_NSF_NOISE_TILED=1` before starting RVC inference. These flags use both specialized kernels on NSF. Leaving the flags unset preserves the verified old path. Weight prepacking happens once when the relevant model is loaded. Run the existing 12-second WAV/PCM parity and profile before changing the default behavior.

## Gates

1. Local reference scatter-versus-gather and single-input convolution: PASS.
2. Rust 1.100.0 CPU-mode `cargo check` main EXE and both probe binaries: PASS.
3. CPU-generated CubeCL kernel parity: Windows CI required, due missing LLVM package in local offline environment.
4. Windows release compilation with CPU/CUDA/WGPU enabled: Windows CI required.
5. Actual RTX 4060 speedup, FP32 error, NCU L1/TEX, L2/DRAM, scheduler stalls, register count, spills and SASS FFMA: require Windows RTX 4060 output.
6. End-to-end 12-second WAV/PCM equivalence: required before switching production defaults.

A successful mathematical prototype does **not** imply faster CUDA execution. If register pressure increases or output-channel vectorization does not help, inspect NCU raw counters before further tuning.
