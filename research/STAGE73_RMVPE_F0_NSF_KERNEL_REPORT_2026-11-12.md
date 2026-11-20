# Stage 73 — RMVPE U-Net F0 parity and real NSF Quad A/B

Parent: `1f30d592bc96c6b780dd3ff7f9d2e1140d3888db`. PR #2. Stage72 result from user-supplied full 89.93925s WAV archive, RTX 4060, 32 chunks, 40kHz, 2 iterations, no CUDA Graph.

## Stage72 verified full-WAV times and numeric parity

| Mode | Warm sec | RTF | PCM16 RMSE vs baseline | PCM16 max abs |
|---|---:|---:|---:|---:|
| Stage67 proven NSF ImplicitGemm + persistent GRU | 12.3620 | 0.13745 | 0 | 0 |
| HuBERT 7 frontend Conv1d GEMM | 10.4977 | 0.11672 | 21.007 | 2341 |
| RMVPE UNet 2D Conv + transpose GEMM | 10.5860 | 0.11770 | **991.129** | **31490** |
| Both new GEMM paths | **8.5348** | 0.09490 | **991.043** | **31510** |

Every mode produced the expected 3,597,570 PCM16 samples and repeated byte-identically within the same mode. Independent waveform comparison identified large UNet-induced differences particularly in portions near 40–50s, 70–90s. Correlation only ~0.874 for UNet GEMM; SNR only ~5.99dB. **Do not promote UNet GEMM to production.** HuBERT GEMM is faster but not proven audio-equivalent (SNR ~39.46dB). Test listening and direct feature-map parity.

## Stage73 implementation

- Preserves legacy `RVC_RMVPE_UNET_GEMM=1` to reproduce Stage72.
- Adds separate `RVC_RMVPE_UNET_CONV2D_GEMM=1` and `RVC_RMVPE_UNET_TRANSPOSE2D_GEMM=1`, allowing precise identification of which algorithm creates F0 drift.
- Adds real RVC diagnostic metadata for `nsf_temporal_conv_kernel` and the split UNet execution mode.
- Adds `profile_rmvpe_f0_stage73` binary. With real private `rmvpe.pt` and `MitaKind.wav`, extracts a 4s segment (default 80–84s), computes original RMVPE FP32 salience and decoded F0 for Direct, Conv2d-only GEMM, transpose-only GEMM and both. Exports raw full FP32 salience, F0 and timing; no synthetic network stand-in.
- Adds `Run-RVC-RMVPE-F0-Stage73.ps1`: four independent processes, checksum + full-array F0/salience RMSE, max abs, changed/voiced frames.
- Adds `Run-RVC-Full-Stage73.ps1`: full actual WAV, baseline (NSF ImplicitGemm), NSF Prepacked Direct, Temporal Quad, Temporal QuadInterior (`-Suite nsf`), or baseline plus UNet split and HuBERT routes (`-Suite unet`). For specialized NSF routes, **disables** `RVC_NSF_IMPLICIT_GEMM` so Quad is actually selected by Burn/CubeCL Direct dispatcher. All full modes retain persistent GRU and fixed RVC settings; tags, raw WAV/PCM16 error, logs, failed cases and warm timing in ZIP.

## Historical CUDA Graph, Prepacked and Quad status

Production full RVC always reports `graph_replay=false`. Hardware capture/replay was previously tested with persistent changing inputs in fixed-shape *synthetic NSF* Stage51–53. No end-to-end CUDA graph is integrated. A safe capture needs persistent input/seed/condition buffers, bounded shape-keyed cache, stream ordering, invalidation and exact replay-vs-eager comparisons.

`RVC_CONV_WEIGHT_PREPACK=1` was active in Stage72. Stage50 synthetic full NSF compared Prepacked, Packed, Quad, QuadInterior and found Sequential+Prepacked best of those tested, but **no** direct comparison against current real-RVC NSF ImplicitGemm has been made. Stage73 provides this missing test.

## Windows RTX4060 instructions

From unpacked CI artifact `rvc-stage73-quality-gpu-ab-windows-x64`, place private model assets under `.\embed\`.

First run targeted real F0 diagnostic:

```cmd
powershell -ExecutionPolicy Bypass -File .\Run-RVC-RMVPE-F0-Stage73.ps1 -Rmvpe ".\embed\rmvpe.pt" -InputWav ".\embed\MitaKind.wav" -StartSeconds 80 -DurationSeconds 4
```

Upload `RVC_STAGE73_RMVPE_F0_RESULTS.zip`.

Then run actual Quad/Prepacked versus NSF GEMM A/B:

```cmd
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage73.ps1 -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -WarmRuns 2 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -Suite nsf
```

Upload `RVC_STAGE73_FULL_WAV_RESULTS.zip`. Use `-Suite unet` later if the raw F0 report isolates a correct accelerated half.

## Validation and constraints

Rustfmt parsed the new Rust executable; logic is branched off the exact Stage72 source. Windows CI compiles with Rust1.98.1 unified CUDA/CPU/WGPU and validates PowerShell parsing, CPU tests and doctor, but has no physical NVIDIA card. CUDA JIT, raw F0 accuracy and Quad throughput are not yet verified. PCM16 numerical equality is required alongside listening and multiple recordings before production promotion. No weights or private audio are committed.
