# Stage 78 — packed-batch compensated TF32×2 R&D (2026-11-13)

PR #2, parent `9860ab7adae3959d0a846584f3cda2ae54105340`. Experiment on real RMVPE with original checkpoint; no permanent production behavior change.

## Stage77 verified RTX4060 evidence

Read user-uploaded `RVC_STAGE77_FULL_WAV_RESULTS.zip` (14 files). ZIP integrity PASS; all mode logs and WAV binary data analyzed independently, 89.93925s source, 3,597,570 PCM16 output samples at 40kHz, 32 real chunks, 2 runs per mode, NSF implicit GEMM + prepacked weights + persistent BiGRU, no graph replay.

| RMVPE Conv2d route | Cold full WAV (s) | Warm (s) | Warm RTF | PCM16 RMSE vs Direct | max_abs PCM16 |
|---|---:|---:|---:|---:|---:|
| Original Direct | 19.2267 | 12.4442 | 0.13836 | 0 | 0 |
| Uncompensated ImplicitGemm/TF32 | 17.6617 | **10.5856** | 0.11770 | **985.6076** | 31490 |
| **Compensated TF32×2, separate GEMMs** | 21.3408 | **11.0838** | **0.12324** | **1.37679** | **46** |

SNR relative to Direct: ordinary GEMM **6.04 dB**, TF32×2 **63.13 dB**; TF32×2 Pearson 0.999999757, PCM16 p99 abs difference 5. Three huge F0 voiced switches from ordinary TF32 no longer cause waveform damage. Stage77 TF32×2 accelerated full WAV by **1.1227×** (1.3603 s saved) relative to Direct, at a 0.4982s overhead to uncompensated GEMM. Exact output WAV SHA256: Direct `62dda39430725fd5014e1e2df65abc140e9be3c30572c1bbcbd45664b7558d7d`; GEMM `53de583b2ea897b4f8e44b429b35b2c76884a5139422ff4786b663778940399b`; compensated `8d9d9288eb94e2a30b0c38110c05ec4144a92490b52e4460949303d91078d97e`. Each mode deterministic between repeated passes. One voice/WAV only; NOT sufficient to promote production.

## Algorithmic Stage78 experiment: one Conv2d dispatch for two-component accuracy

Current compensated implementation performs `Conv2d(high,W,b) + Conv2d(low,W,0)`, with TF32 rounding `high=round_tf32(x)`, `low=x-high`; weights were FP16 in original checkpoint and are exactly representable as TF32.

New, distinct opt-in flag `RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32=1`:

1. One CubeCL GPU kernel writes high and low to **one contiguous [2B,C,H,W] buffer**, high first, low second, handling noncontiguous input strides.
2. One implicit GEMM Conv2d uses the same W for all 2B images, bias omitted.
3. Slice output into two B-image tensors, add bias to high result, then accumulate high + low.

Mathematical identity is exactly the same as Stage77, aside from permitted FP32 accumulation order/tiling. The two components still entail roughly **2× multiplication work**. Gains, if any, come from fewer dispatch/launches and potentially better GPU occupancy, not from reduced arithmetic. There are **no case-specific F0 fixes, smoothing, threshold changes, reference files, or changes to the model**. Existing separate-Conv TF32×2 remains untouched as the control route. Default RMVPE Direct remains unchanged.

### CPU numerical harness

Tested equivalent high/low batch packing with PyTorch CPU Conv2d for small FP32 shapes including bias. The max elementwise error of batched versus separate Conv2d ranged from 0 to `5.72e-6` (different GEMM internal summation order). This only validates the grouping identity, **not real CUDA/JIT correctness**. Another pure Rust unit checks packed memory layout and compensated reconstruction for B=1,2.

### RTX test gate

Workflow `.github/workflows/rvc-stage78-tf32-batch.yml` builds unified Rust 1.98.1 Windows executable and includes the Stage78 runner. With original five private assets in `embed`:

```cmd
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage78.ps1 -Device cuda -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -WarmRuns 3 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -Modes all
```

It runs Direct / original ImplicitGemm / verified separate TF32×2 / experimental batch TF32×2 and writes `RVC_STAGE78_FULL_WAV_RESULTS.zip` with execution metadata, WAV, SHA256, PCM16 parity and error logs. Stage78 performance, sample accuracy and CUDA kernel launch success are **not claimed** before the user returns RTX results. Fallback to existing route must never be silent. Check that the packed mode produces RMSE comparable to the working TF32×2 (1.377), not the unsafe 985.6 of ordinary GEMM. Measure warm full-WAV A/B fairly; changing GEMM batch geometry may affect individual kernel schedules and JIT.

No CUDA Graph capture/replay integrated in full pipeline (`graph_replay=false`). Private weights and user WAV not committed.
