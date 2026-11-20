# Stage 106 handoff

Continue `Atm4x/Vendor-for-burn`, draft PR #2, experimental branch `experiment/conv-kernel-rnd-stage33-20261110`. Use a Git bundle or the existing local checkout; do not repeatedly call GitHub. Rust 1.100.0, no `dotnet build` for Unity. Do not modify unused Legacy/Obsolete scripts.

Stage 105 investigated six user-provided NCU CSVs; see `STAGE105_REPORT_2026-11-15.md`, `/mnt/data/stage105/analysis_summary.json`, or the saved diagnostic archive. Reuse 2×2 gains ~3.1× GPU kernel speed over Direct but uses 118 registers/thread and still loses ~5–6.6× to cuDNN Winograd's main compute kernel. Exported compact CSV has no per-kernel Tensor Core details; do not infer zero utilization for PyTorch.

Stage 105 adds an opt-in FP32 3×3 1×2 spatial pair kernel intended to reduce register demand at the cost of more input loads. Rust reference tests 2/2, Python tests 23/23. Full CubeCL Windows compile and CUDA results must be confirmed via the single active Stage 105 Windows CI. If CI fails, inspect exact job log, repair in a minimal commit through GitHub connector, rerun only the necessary workflow. No historical workflows should be automatically activated.

On RTX 4060 compare direct/reuse/pair for dense3x3 and dense3x3_wide with NCU and unprofiled medians. Verify full tensor parity, actual register count, achieved occupancy, L1TEX LSU, eligible warps, duration. Stage104 runner is reused, with `-MicroStrategies direct,reuse,pair`, no WAVs in archives. Investigate streamed 2×2, FP32 warp tiling, and fused Winograd based strictly on measurements, then integrate the fastest and numerically verified candidate into RMVPE opt-in testing. Goal Rust/PyTorch ≤1.2× for genuinely comparable workloads, not an NCU-only synthetic speed claim.
