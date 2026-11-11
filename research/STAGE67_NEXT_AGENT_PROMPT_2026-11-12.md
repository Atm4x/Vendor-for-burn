# Stage 67 — from Stage66

Retrieve the newest Git Bridge bundle for PR #2. Read `research/STAGE66_SHARED_F0_REPORT_2026-11-12.md`, Stage66 Windows CI and user-supplied `RVC_STAGE66_FULL_WAV_RESULTS.zip`.

Validate exact output length, stable warmed results, production Sequential, no fallback, and legacy Stage65 PCM bitwise compatibility. Analyze real shared-vs-legacy latency and audio difference (including window joins near 9s, 18s, ...). Only enable `RVC_GLOBAL_F0` by default if both performance and pitch quality pass.

Follow up with real-device full-route GPU profiling to quantify NSF Conv1d/ConvTranspose and RMVPE GRU bottlenecks. Do not claim synthetic CPU speedups transfer to full RVC. Plan and implement a truly fused native recurrent GRU with FP32 correctness tests if the end-to-end result warrants it. Do not change unused legacy scripts. Produce a source-only progress report, tests and next prompt.
