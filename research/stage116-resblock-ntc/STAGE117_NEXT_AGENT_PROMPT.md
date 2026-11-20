# Stage117 — GPU acceptance, not another unmeasured ResBlock kernel

Start from local Git Bridge PR #2 at `e41ce4b2b1d9c5c89b8c4f5cdef7e059817a03bc` plus Stage116 cumulative patch. Do not fetch repository source by individual GitHub files; use a refreshed CI Git bundle if HEAD changed. Source changes and local tests only, publication via GitHub connector after validation.

1. Verify Stage116 patch against the updated PR #2 head; validate `git apply --check` and checksum before use. Do not edit unrelated legacy code.
2. Windows CI: publish this patch through the GitHub connector and run the new `rvc-stage116-resblock-ntc.yml` workflow. Confirm PowerShell syntax, Rust executable build, and startup `doctor`. CI lacks an RTX4060, so it is a compilation gate only.
3. On RTX4060: `Run-Stage116-ResBlock.ps1 -Profiler measure -Modes baseline,baseline_ntc,gemm,gemm_ntc,transport,transport_ntc -Runs 8` with identical complete inputs; inspect `*.benchmark.json`, compare PCM16 and retain WAVs privately.
4. Use NCU for `C128/K11/D5 first` and `C128/K11/D1 second`, across at least baseline, gemm, transport, transport_ntc; obtain Nsight Systems timeline for all ResBlocks and full generator, not just launch-count samples.
5. Inspect `gpu__time_duration.avg`, registers, occupancy, load/store issue, L1TEX/L2/DRAM traffic, long scoreboard, FFMA/MMA utilization. Verify source-layout copies and CUDA kernel names from traces. Compare warm GPU events and end-to-end wall time independently.
6. Gate on full RVC PCM quality and measured GPU/warm inference gains; stop this transport optimization if not competitive. If compute-bound, research CubeCL Tensor Core-friendly implicit Conv1d with proper prepack and graph-aware residency (no explicit Stage111 phase GEMM resurrection).

Stage116 current status: Rust 1.100 CPU/CUDA/WGPU typecheck PASS, std-only reference 4/4 PASS, no GPU runtime measurements. Opt-ins `RVC_NSF_RESBLOCK_TRANSPORT` and `RVC_NSF_RESBLOCK_SHARED_NTC` are independent. Stage115 `into_contiguous_aligned` residual bug fixed using strict `into_contiguous`.
