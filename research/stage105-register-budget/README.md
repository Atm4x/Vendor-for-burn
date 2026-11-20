# Stage 105 — register-budget alternatives for dense FP32 Conv2d

Source-only experiment; FP32 dense 3×3 is measured on RTX 4060 in Stage 104 Direct/Reuse and PyTorch; Pair is newly added and not yet GPU benchmarked.

`RVC_STAGE103_STRATEGY=pair` selects Pair in `stage103-conv2d-probe.exe cuda 8 dense3x3`. Full-RVC opt-in is `RVC_RMVPE_UNET_SPATIAL_PAIR=1` (never default).

To profile selected GPU shapes after Windows CI succeeds:

```powershell
.\Run-RVC-Stage104-NCU-Matrix.ps1 -Scope micro -Cases dense3x3,dense3x3_wide -MicroStrategies direct,reuse,pair -PythonExe "D:\Projects\tts-with-rvc\.venv\Scripts\python.exe" -NcuLaunchCount 3
```

The runner requires NVIDIA NCU performance-counter permission and NVTX DLL found in the PyTorch installation. Results contain logs, JSON and `.ncu-rep`, never WAVs, models or indexes.
