# Stage 111: phase-domain batched ConvTranspose1d

## Measured cause

Stage 110 RTX4060 FP32 polyphase CUDA captures report that channel-pair tiling increases L1/TEX LSU pressure and Long Scoreboard stalls while reducing CTA count. Stage 97 PyTorch RVC NSYS trace contains FP16 cuDNN CUTLASS Tensor Core kernels. The Stage 97 trace is **not a matched ConvTranspose1d microbenchmark**; Stage 111 adds a matched Torch control for each exact shape.

## Algorithm

For output phase `p = t % stride`, eligible transposed-filter taps satisfy `k = (p+padding)%stride + h*stride`. The input index is `i = floor((p+padding)/stride) + floor(t/stride) - h`. Thus each phase is a dense `[time, Cin*taps] @ [Cin*taps, Cout]` matmul. Stage 111 precomputes a 4D packed phase-weight tensor once, gathers all phase input matrices in one kernel per convolution, calls CubeCL's batched matmul and scatters all phases and bias in one final kernel. No per-output-channel direct gather and no per-phase kernel launch.

- `RVC_NSF_PHASE_GEMM=f32` (FP32 input/weight/output, full precision).
- `RVC_NSF_PHASE_GEMM=f16` (FP16 input/weight and CubeCL matmul, output converted back to FP32; precision differs).
- unset: retains the current production path.

The FP16 numerical gate on synthetic tensors uses `max_abs <= 5e-3`, SNR >= 50 dB against FP32; it does **not** certify end-to-end voice quality or exact parity.

## Offline validation

`rustc --edition=2024 --test research/stage111-phase-gemm/reference.rs -o reference && ./reference`

`DOCS_RS=1 cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu,cuda,wgpu --bin tts-with-rvc-rs --bin stage111_phase_gemm_probe`

The `DOCS_RS=1` setting is strictly for offline type checking when the Tracel LLVM bundle isn't installed. It doesn't validate CUDA JIT execution.

## RTX4060 control

Run the CI artifact's `Run-RVC-Stage111-Phase-GEMM.ps1 -Backend cuda -Profiler ncu -Runs 12 -NcuLaunchCount 8 -PythonExe <venv-python>`.

The runner compares fully synchronized Burn Direct, Phase-GEMM F32, Phase-GEMM F16, and matching PyTorch/cuDNN F32, native F16 and F16-with-input/output-casts calls on the same exact tensor values. NCU records include the cast/gather/matmul/scatter kernels. All outputs are checked against Burn Direct FP32. Output ZIP excludes WAV/model weights.
