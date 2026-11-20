# Stage 76 — Investigate full-chunk F0 numerical parity and CUDA replay

Continue PR #2 from the latest Stage75 Git Bridge, not obsolete branches. Read `research/STAGE75_F0_PRODUCTION_CHUNK_AUDIT_2026-11-12.md` and analyze user-uploaded `RVC_STAGE75_EXACT_F0_RESULTS.zip` when provided.

Stage74 actual GPU full-RVC 89.94s: NSF all-GEMM 12.369s warm, QuadInterior residual Conv1d + GEMM transpose 18.569s, Quad 22.101s, Packed 28.315s, Prepacked Direct 68.143s; all other modes slower. Preserve NSF all-GEMM and persistent BiGRU production baseline.

Stage73 isolated RMVPE 4s cut: U-Net Conv2d GEMM saves ~2x RMVPE subroutine latency, but final FP32 F0 difference is small (max 0.025Hz), zero voiced and zero coarse pitch changes on that independent segment. Stage72 full-WAV with U-Net GEMM had large PCM16 RMSE ~991; root cause still unknown. Stage75 records the REAL full-RVC chunk F0 after filtering, reflection, batching, pitch shift. Compare all F0 and coarse pitch bins per 32 chunks, locate the exact frames responsible for any downstream divergence, separate F0 changes from NSF phase sensitivity, and require listening/quality checks on the audio. Never classify small F0 difference as a math bug without proof, nor promote a 991 PCM RMSE route without diagnosis.

If exact full-chunk F0 remains numerically stable, develop an isolated NSF source amplification study with identical noise/seed and controlled F0 injection, and compare by audio perception/fundamental frequency metrics. If actual pitch bins or voicing differ, identify input/shape-sensitive Conv2d layers using per-layer FP32 comparisons.

For further performance, plan proper fixed-shape CUDA Graph capture/replay in real RVC with persistent H2D input buffers, stable tensor addresses/noise, conservative shape-bounded graph cache, stream ownership, and replay/eager full-WAV parity. Stage51-53 only tested synthetic NSF graphs; the production pipeline still reports graph_replay=false. Do not imply replay is integrated until a real RTX test passes.
