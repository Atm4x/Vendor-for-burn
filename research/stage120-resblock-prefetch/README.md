# Stage120 — NCU-guided ResBlock weight prefetch

This is an opt-in Rust Burn/CubeCL experiment for RVC40 GeneratorNSF on RTX 4060. NCU C128/K11 reported Long Scoreboard 4.355/4.811 average warps per issued instruction, DRAM throughput below 9%, and achieved occupancy above 97%. The hypothesis is that issuing a future weight read before the current channel's FP32 FMAs can reduce dependent-load latency. It may also worsen occupancy; no speedup is claimed.

The new path is enabled by RVC_NSF_RESBLOCK_WEIGHT_PREFETCH=1 only when RVC_NSF_RESBLOCK_TRANSPORT=1 and Cout=128, K=11. All other cases use the existing algorithm. The compile-time kernel preserves FP32 accumulation order; no CUDA-only implementation is introduced.

The Windows artifact contains the Rust 1.100.0 EXE, one complete PowerShell runner, this README, a technical report, a workflow contract, and a build SHA. No WAV/model files are distributed. Place the existing embed folder next to the extracted EXE.

Installed tool paths are included in the runner: NCU 2025.1, NSYS 2024.6.2, CUDA 12.8, and the installed NVTX DLL in the conda torch2.6.0 library (DLL only). The sole Python for separate PyTorch reference work is D:\Projects\tts-with-rvc\.venv\Scripts\python.exe. The runner itself does not call Python.

Preflight:
.Run-Stage120-ResBlock-Prefetch.ps1 -CheckOnly

Full experiment:
.Run-Stage120-ResBlock-Prefetch.ps1 -Cycles 2 -Runs 5 -Profiler all

The runner measures independent processes for baseline, control (transport, shared NTC, tile4), and prefetch (identical except for new prefetch). Persistent GRU is held equal for all three. It records warm medians, PCM16 parity, targeted NCU reports, full NSYS timeline and SQLite exports. One ZIP includes logs/JSON/profiles/SQLite but excludes WAV, checkpoints, indexes and audio.

Accept the prefetch only if real RTX 4060 kernel and full-RVC performance improve with acceptable parity and register/occupancy effects. NCU replay wall time is not a production latency benchmark.
