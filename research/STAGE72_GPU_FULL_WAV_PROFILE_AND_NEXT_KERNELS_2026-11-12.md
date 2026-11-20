# Stage 72 — four-mode GPU full-WAV profiling and next convolution experiments

Date: 2026-11-12. Parent: `81633489b88276c06fef247ba65121bbce717088`; PR: https://github.com/Atm4x/Vendor-for-burn/pull/2

## Real RTX4060 evidence

Directly inspected user-supplied `RVC_STAGE67_FULL_WAV_RESULTS.zip` and `RVC_STAGE68_PROFILE_RESULTS.zip`. All four Stage67+68 modes cover the same 89.939297 s input, 32 chunks, 3,597,570 output samples, 40 kHz, `core_frames=300`, batch 2, CPU FAISS/CPU F0, no CUDA Graph. Every Stage68 WAV hash matches its corresponding Stage67 mode.

Stage67 **uninstrumented warmed** entire-WAV CUDA times:

| Mode | First (s) | Warm (s) | Warm RTF | vs baseline |
|---|---:|---:|---:|---:|
| Direct NSF + stepwise GRU | 43.2814 | 39.9087 | 0.44373 | 1x |
| NSF ImplicitGemm only | 22.8114 | 15.9475 | 0.17731 | 2.502x |
| persistent GRU only | 38.7943 | 35.8983 | 0.39914 | 1.112x |
| NSF ImplicitGemm + persistent GRU | 18.2800 | **12.3673** | **0.13751** | **3.227x** |

NSF-only saved 23.9612 s, GRU-only saved 4.0103 s; combined saved 27.5414 s. Bitwise determinism across the two iterations of each mode. Output numeric differences versus baseline: combined PCM16 RMSE 1.5519, max absolute 86; NSF-only RMSE 1.5276/max88; GRU-only RMSE 0.3521/max10. This validates no gross numerical drift, **not listening quality**.

Stage68 synchronized diagnostic (17 measured batches, 884 spans per mode) for **combined**:

| Non-overlapping top-level phase | Inclusive time (s) |
|---|---:|
| NSF | 5.1644 |
| HuBERT | 3.1095 |
| RMVPE | 3.0684 |
| TextEncoder | 1.8301 |
| Flow | 0.5278 |
| FAISS + pitch protection | 0.1451 |

Nested hot spans: NSF residual aggregation **4.4727s**, HuBERT convolution frontend **2.6124s**, RMVPE UNet **2.4385s**, TextEncoder attention/FFN **1.7443s**, persistent GRU **0.5063s**. Disjoint NSF residual stage totals 0/1/2/3: **0.2872 / 1.1880 / 1.5457 / 1.4519 s**; actual NSF upsample is only 0.2188s. Stage68 combined diagnostic wall 14.0142s; nested GPU sync changes timings, so **do not add child+parent or directly compare diagnostic RTF to Stage67 warm RTF**.

## Implemented opt-in Stage72 source changes

- Generalized the previously active `nsf_gemm.rs` to shared `cubecl_conv.rs` without changing its 1D algorithm or default scheduling.
- `RVC_HUBERT_FRONTEND_GEMM=1` routes all seven HuBERT frontend Conv1d layers to `ConvStrategy::ImplicitGemm` instead of the default Direct. Preserves existing weights, padding/stride, GELU, GroupNorm and post projection.
- `RVC_RMVPE_UNET_GEMM=1` routes real RMVPE UNet Conv2d to `ImplicitGemm` and ConvTranspose2d to GEMM/col2im. Preserves existing normalization, residual and concatenation semantics.
- Both features are optional **GPU experiments**, never enabled in standard production or CPU paths. Unsupported kernels fail explicitly instead of silent CPU fallback.
- Full-WAV report records both algorithms in additional independent metadata fields.
- New Windows A/B runner compares `baseline` (proven Stage67 NSF+GRU accelerated), `hubert` (only new HuBERT option), `unet` (only new UNet option), and `both` (both new options). All cases keep NSF GEMM, persistent GRU and Prepacked weights. Two full-WAV passes, PCM16 parity, per-mode stdout/stderr; failed cases are preserved in `failures` and included in results ZIP.

## Validation status and execution

Offline Rust 1.98.1 `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cuda --lib --bin tts-with-rvc-rs` **PASS**, twice; `git diff --check` PASS. No local physical CUDA device; performance, NVIDIA JIT shape support and numeric parity of the new candidate kernels **remain unverified**. CPU-only check exceeded the command budget and is not represented as a PASS; Windows CI builds unified CPU+CUDA+WGPU EXE and unit tests.

After CI success download `rvc-stage72-hubert-rmvpe-gemm-windows-x64`, extract and place original 5 private assets into `embed`. From Windows CMD:

```cmd
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage72.ps1 -Device cuda -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -WarmRuns 2 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -Modes all
```

Output `RVC_STAGE72_FULL_WAV_RESULTS.zip`. Compare successful modes against its baseline and inspect any failures before considering promotion. Private model/index/WAV files are not committed.
