# Stage121 — ResBlock Plane Broadcast experiment

This is an experimental opt-in for C128/K11 Burn/CubeCL on RTX 4060, based on real Stage120 NCU/NSYS evidence. **No speedup is claimed.** Stage120 weight prefetch was rejected because PCM diverged by up to 25966 LSB and actual ResBlock GPU kernels slowed by 26.2%.

Stage121 shares input scalar loads across the 32 output-channel lanes using CubeCL `plane_broadcast`, preserving the accumulation order. It applies only with ResBlock transport, NTC, temporal tile4, vector4; other shapes keep baseline. Flag: `RVC_NSF_RESBLOCK_PLANE_BROADCAST=1`.

The CI artifact contains a newly built `tts-with-rvc-rs.exe`, `stage121_plane_broadcast_probe.exe`, `Run-Stage121-ResBlock-Plane-Broadcast.ps1`, `Export-Stage121-NCU-Metrics.py`, the report and workflow contract. Put existing `embed/` with CrazyMita.pth, hubert_base.pt, rmvpe.pt, CrazyMita.index and MitaKind.wav alongside. Do not use old Stage120 binaries.

From an unpacked artifact with `embed` present:

```powershell
.\Run-Stage121-ResBlock-Plane-Broadcast.ps1 -CheckOnly
.\stage121_plane_broadcast_probe.exe cuda
.\Run-Stage121-ResBlock-Plane-Broadcast.ps1 -Cycles 2 -Runs 5 -Profiler all
```

The first command checks paths. The dedicated CUDA probe is a small real CubeCL parity test and is also executed automatically by the full runner. The full runner runs baseline/control/broadcast in alternating order, compares 119200-sample PCM16 output and **requires bit-exact broadcast vs control** before running expensive NCU/NSYS. If parity fails it archives the results and stops with REJECT_PCM. The resulting archive contains full NCU reports, optional exported CSV, complete NSYS timelines in SQLite, JSON, logs and no WAV/model assets. Upload that one ZIP.

Tool paths already configured in runner: NCU 2025.1, NSYS 2024.6.2, NVTX DLL under miniconda3 torch2.6.0/torch/lib. **Only Python for execution**: D:\Projects\tts-with-rvc\.venv\Scripts\python.exe (optional Nsight SDK exporter). Do not install anything manually, and do not compare NCU replay times against warm end-to-end medians.
