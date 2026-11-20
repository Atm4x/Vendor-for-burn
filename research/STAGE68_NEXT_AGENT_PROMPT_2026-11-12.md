# Stage 68 — validate Stage67 on RTX4060

Retrieve latest Git Bridge PR #2 and compare commit SHA; read `research/STAGE67_ELEPHANT_KERNELS_REPORT_2026-11-12.md`. Inspect user's `RVC_STAGE67_FULL_WAV_RESULTS.zip`.

Stage67 has two experimental GPU paths: `RVC_RMVPE_GRU_PERSISTENT=1` uses one CubeCL recurrent kernel launch for both GRU directions per tensor batch, and `RVC_NSF_IMPLICIT_GEMM=1` uses implicit GEMM Conv1d plus GEMM/col2im ConvTranspose1d. Stage65 Sequential production unchanged; Stage66 global F0 slower and off in Stage67 A/B.

Compare warmed full-WAV time and GPU load for baseline, nsf-only, gru-only and both; check output duration, PCM error, deterministic passes, audio and CUDA JIT. Verify correct gate order and barriers if GRU drifts, correct shape/dilation/padding and VRAM workspace if GEMM fails. No silent fallback or optional backend switch. Merge only proven improvements. Continue algorithmic work on observed bottleneck, not more certificates. Produce measurements and next report.
