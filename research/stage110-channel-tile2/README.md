# Stage 110 — NSF ConvTranspose1d channel tile2

GPU R&D only. Two CUDA strategies: `tile2` (two output-channel vectors share each input load) and `tile2_split` (four independent FP32 accumulators). Default production inference remains unchanged.

From the Windows CI artifact, run `Run-RVC-Stage110-Channel-Tile2.ps1 -Backend cuda -Profiler ncu -Runs 12 -NcuLaunchCount 2 -PythonExe 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe'`. NCU requires NVTX DLL from PyTorch and GPU counter permissions. `-DryRun` checks the plan without GPU. `-Profiler none` runs timings and parity only. The output ZIP excludes audio and models.

Report: `STAGE110_REPORT_2026-11-15.md` in source archive and Library.
