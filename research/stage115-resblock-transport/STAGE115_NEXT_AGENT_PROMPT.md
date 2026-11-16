# Stage116 — Verify Stage115 ResBlock transport on hardware

Source is a patch against PR #2 Git Bridge SHA `e41ce4b2b1d9c5c89b8c4f5cdef7e059817a03bc`, branch `rnd/stage115-resblock-transport`. Apply only locally; use GitHub connector for commits and CI.

The experiment is opt-in `RVC_NSF_RESBLOCK_TRANSPORT=1`. New generic CubeCL kernels fuse LeakyReLU/Conv1d and optional residual epilogue while keeping packed `[K,Cin,Cout]` model weights and NTC outputs. Production default unchanged. Theoretical ResBlock MAC share 97.23% is not runtime attribution. Existing NCU Stage102 is Conv2d; Stage109–112 are ConvTranspose1d, so capture REAL ResBlock Conv1d first.

1. Acquire offline Tracel LLVM 23.1.0-4 bundle or use GPU-capable Windows local environment. Rust `cargo check --features cpu` PASS and std-only Rust reference 2/2 PASS. CPU link failed without LLVM. Confirm complete Burn CubeCL runtime tests including full paired nonlinear ResBlock graph, not just math reference.
2. CUDA-only typecheck: PASS (Rust 1.100.0, offline, DOCS_RS=1), runtime NOT TESTED. Next, finish WGPU typechecking, then actual GPU JIT and CUDA kernel correctness. Guard numerical tolerance for mixed signs and boundary padding. Test C32/64/128/256, K3/K7/K11, dilation 1/3/5, B>1.
3. Build Windows artifact, run `Run-Stage115-ResBlock.ps1 -Profiler measure` then `-Profiler nsys` and `-Profiler ncu -Modes gemm,transport` on RTX4060. Compare warm medians, per-kernel GPU time, launch counts, registers, SM efficiency, L1TEX, L2, DRAM, long-scoreboard stalls. Save original ncu-rep/nsys-rep and logs.
4. Compare full WAV against original PyTorch CUDA and existing Rust baseline at the exact same source, precision, checkpoint, F0 and index. Require full PCM accuracy and end-to-end improvement.
5. If transport remains memory-throughput limited, investigate full graph resident data reuse and implicit GEMM scheduling. Do NOT claim improvement from operator counts alone or recycle Stage111 explicit TF32x3 GEMM / Stage112 fused2tap without evidence.

No production switch until full GPU acceptance. Do not change unused Legacy/Obsolete files. No dotnet in Unity.
