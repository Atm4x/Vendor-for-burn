# Stage 103 — FP32 dense 3×3 spatial 2×2 register-footprint reuse

## Hypothesis

NCU Stage 102 measured 95.32% `l1tex__data_pipe_lsu_wavefronts` and 4.66% DRAM utilization for one Direct Conv2d kernel on RTX 4060. A 2×2 output tile covers 16 distinct input spatial sites for a 3×3 stride-1 filter. Direct computation reads 4×9=36 input vectors for one channel chunk; register-footprint reuse loads 4×4=16, then performs all 4 outputs from the resident footprint. Mathematical operation count is unchanged. Loop interchange may alter floating-point accumulation order.

## Safety

- Opt-in RMVPE: `RVC_RMVPE_UNET_SPATIAL_REUSE=1`, eligible only for dense 3×3 padding=1 stride=1 dilation=1 FP32.
- No default routing changes; kernels for other shapes remain Direct (or any separately selected experimental algorithm).
- GPU speed and parity **not measured** in the authoring environment. Compile-only is not validation of CUDA execution.
- CPU mathematical reference: `rustc --edition 2024 --test reference.rs -O -o ref_test && ./ref_test --nocapture`.
- Two GPU variants compiled into the *same* Stage 103 binary: `RVC_STAGE103_STRATEGY=direct|reuse`, with identical weights and input generation. Reuse mode calculates full-output parity against Direct after benchmark timing.
- On RTX 4060: `Run-RVC-Stage103-Conv2d-Reuse.ps1 -PythonExe <python> -Runs 12 -Profiler ncu`. Run PowerShell as Administrator if NCU counters are restricted; NVTX runtime is auto-added from the provided PyTorch venv if available.
- `-NcuSkip 0` profiles first dense 3×3; `-NcuSkip 1` profiles next dense 3×3. NCU kernel-name filter prevents the copy kernels from consuming the launch quota. NCU CSV CLI import is intentionally omitted (user's Nsight Compute 2025.1.0 returned `bad conversion`); open `.ncu-rep` in NCU GUI, export Raw CSV.
- Zip includes JSON, logs, NCU reports, no WAV.

## Acceptance

1. CPU reference: all tests PASS, maximum absolute difference vs Direct < 2e−5.
2. Windows build: CI compile and static runner parse PASS.
3. CUDA run: `full_parity.passed=true` for every dense3x3; additionally compare full RMVPE F0 / PCM before considering production.
4. Performance: compare *kernel-only* times and `l1tex__data_pipe_lsu_wavefronts` for both strategies on identical shapes; monitor spills, registers, occupancy, grid/tail effect. Reject if speed worsens, even if theoretical load count improves.
5. Later variants: output-channel tiling, shared-memory loads, CMMA-friendly implicit GEMM. Do not prematurely select these based only on CPU or operation counts.
