# Stage 67 — structural GPU R&D: NSF GEMM and persistent RMVPE BiGRU

PR #2, branch `experiment/conv-kernel-rnd-stage33-20261110`. Parent Stage66 fix: `0045aa919ed590b04e076ad37726b7c06536db49`.

## Evidence and motivation
Stage65 89.94s WAV warmed CUDA ~44.47s; Stage66 same WAV warmed legacy 39.17s, shared-F0 39.81s (1.6% slower and substantially different PCM). Stage64 intrusive sync profile: NSF 34.17/62.06s, RMVPE 14.24s incl. BiGRU 9.53s. These numbers are **not direct GPU kernel timings**. Production NSF used Direct convolution; RMVPE BiGRU used Burn matmul/sigmoid/tanh and sliced inputs for every frame.

## Code and algorithms
- `rmvpe_persistent.rs`: CubeCL kernel with 256 threads per cube, one cube per (batch,direction), both GRU directions in **one kernel dispatch per tensor batch** using `CubeCount::Static(B,2,1)`. Hidden vector resides in shared memory with step barriers. Gates use original [reset,update,new] ordering, forward/reverse temporal indexing, and [B,T,512] output layout. Input projection uses one combined Burn matmul and combined immutable weights prepared at model load. Enable via `RVC_RMVPE_GRU_PERSISTENT=1`. No per-timestep host/kernel dispatch. This is not a single full 90-second RMVPE inference: original audio chunks remain, while Stage66 global F0 remains opt-in and **disabled** for Stage67.
- `nsf_gemm.rs`: direct CubeCL backend for NSF `Conv1d` via `ConvStrategy::ImplicitGemm` and `ConvTranspose1d` via an equivalent width-1 2D transposed convolution with `ConvTranspose2dStrategy::Gemm`. Applies to all actual NSF upsampling, residual and noise convolution sites without changing stored weights or synthesis geometry. Enable via `RVC_NSF_IMPLICIT_GEMM=1`. This chooses GPU GEMM/col2im algorithms rather than Direct; additional temporary GPU workspace is possible.
- Baseline unchanged: Sequential NSF residual execution, CPU-decoded F0 and FAISS, no CUDA Graph, prepacked Conv1d weights. An explicitly selected unsupported accelerated route raises an error; it does not silently fall back to CPU or old kernels.
- Benchmark report now records `nsf_conv_execution` and `rmvpe_gru_execution`; Stage67 compares baseline, NSF-only, GRU-only, both on full WAV, two passes per mode, PCM16 differences and RTF.

## Verification
PASS: Rust 1.98.1 `cargo check --offline --locked --no-default-features --features cuda -p tts-with-rvc-rs --lib --bin tts-with-rvc-rs` using `DOCS_RS=1` to avoid the Tracel LLVM SDK download in an isolated container. Rustfmt syntax and git diff whitespace checks PASS. **CUDA JIT execution, real RTX4060 throughput, FPGA/VRAM workspace, and sound parity NOT tested locally**. Windows CI produces a unified CPU/CUDA/WGPU executable and checks structural tests; its runner does not expose physical RTX 4060 GPU.

## Windows A/B from CMD
Download artifact `rvc-stage67-elephant-kernels-windows-x64` from Stage67 workflow, extract, place private CrazyMita.pth, hubert_base.pt, rmvpe.pt, CrazyMita.index and source MitaKind.wav into `.\embed\`. Run from the extracted folder:

```cmd
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage67.ps1 -Device cuda -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -WarmRuns 2 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -Modes both
```

`-Modes both` runs baseline then both accelerated, each first+warm entire-WAV. `-Modes all` runs baseline/NSF-only/GRU-only/both for isolating effects (8 whole-WAV passes). Output ZIP `RVC_STAGE67_FULL_WAV_RESULTS.zip` includes WAVs, logs, full-WAV benchmark JSON, RTF and pairwise PCM16 deltas. Other GPU applications (e.g. Unity) may distort absolute performance; keep load comparable between modes.

## Acceptance and risks
- Kernel compile is **not** numerical or GPU JIT proof. On RTX, validate no errors, output length, finite values, deterministic behavior, audible quality, and warmed latency.
- The persistent GRU launches only 2*B CUDA workgroups; limited SM occupancy may outweigh the savings from deleting thousands of launches. Its result can differ slightly from torch/Burn due to FP32 gate accumulation.
- GEMM NSF may have large workspaces, dtype/shape support constraints, and different numeric accumulation from Direct. Explicit failure instead of fallback on unsupported configurations.
- Promote either path to CUDA default only after a successful real-device A/B without significant quality drift. Stage66 global F0 stays off: previously slower with significant PCM change.

No unrelated legacy scripts or private checkpoint assets were modified.
