# Stage 105 continuation prompt

Work locally against the latest full history Git bundle/PR #2, using Rust 1.100.0. Stage104 added an all-block Nsight Compute matrix to the existing Windows workflow. The local source is in `research/stage104-ncu-matrix/`. Investigate Stage104 artifacts before writing new kernels.

1. Verify Windows CI produced artifact `rvc-stage104-ncu-matrix-windows-x64`; if failed, fix the *single* active workflow via the GitHub connector; do not reactivate historical automatic actions.
2. User runs `Run-RVC-Stage104-NCU-Matrix.ps1 -Scope micro -PythonExe ...` for 12 captures or `-Scope all` for 29 captures (17 RVC blocks + 12 Conv2d micro). Requires administrator-enabled NCU counters on RTX 4060, and existing PyTorch `nvToolsExt64_1.dll`. Full RVC requires models plus WAV in `embed/`. Do not put WAV in the results ZIP.
3. Check NVTX process-scoped nested expressions (`stage102_rust_measured/*/stage104_*`) on actual RTX. Zero-kernel pipeline ranges are recorded, not considered measured; do not infer performance from them. If not captured, debug expression first, preserve model behavior.
4. Analyze `direct` vs `reuse`: GPU Duration, L1/TEX wavefront throughput, DRAM/L2, registers, spills, occupancy, achieved warps, scheduler stalls. Profile named `rmvpe_unet` block with `-Families conv` to distinguish model-shape effects.
5. Compare matched Conv2d FP32 with PyTorch FP32; FP16/NHWC are separate experiments. Move to channel/output spatial tiling, shared-memory tile loads or CMMA/GEMM *only when* the NCU evidence warrants.
6. Keep source-only archives and GitHub commits reproducible, do not touch obsolete scripts, no dotnet/Unity builds, and keep historical workflows dispatch-only.

Acceptance: numerical full tensor parity, real RTX benchmark median including warmed production pipeline, valid NCU reports for targeted kernels, target Rust/PyTorch <=1.2x not yet achieved.
