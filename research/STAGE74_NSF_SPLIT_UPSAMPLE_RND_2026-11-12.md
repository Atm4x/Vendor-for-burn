# Stage 74 — isolate NSF Conv1d vs ConvTranspose1d CUDA performance

Parent PR #2: `e8b6eb0465237978362bf6ccbf725e73f421bd61`. Source: uploaded `RVC_STAGE73_FULL_WAV_RESULTS.zip`. Hardware: RTX 4060; input 89.93925 seconds; 32 RVC chunks; 2 passes per mode; production Sequential residuals + persistent RMVPE GRU, FAISS/F0 CPU, no CUDA Graph.

## Measured Stage73 (whole real RVC, warmed)

| NSF mode | Warm seconds | RTF | WAV SHA prefix |
|---|---:|---:|---|
| ImplicitGemm all Conv1d+all ConvTranspose1d | **12.0247265** | 0.13370 | 62dda394 |
| Prepacked Direct Conv1d+Direct ConvTranspose1d | 37.9636642 | 0.42210 | db4d1e68 |
| Quad Conv1d+Direct ConvTranspose1d | 20.9659564 | 0.23311 | db4d1e68 |
| QuadInterior Conv1d+Direct ConvTranspose1d | 18.3331227 | 0.20384 | db4d1e68 |

All four cases succeeded, output 3,597,570 samples at 40 kHz, each repeated byte-identically. Prepacked/Quad/QuadInterior WAVs are **identical byte for byte** to each other. All-GEMM vs Direct-group PCM16 RMSE 1.52636, max absolute 86, Pearson 0.999999704, SNR 62.24 dB. Output length and range valid, not a substitute for listening. All four runs report `graph_replay=false`.

## Methodological discovery

Stage73 selected the entire NSF conv family through `RVC_NSF_IMPLICIT_GEMM=1`. In production source `decoder.rs`, this switches **both** Conv1d residual/noise kernels **and** ConvTranspose1d upsampling from Direct to GEMM. Hence Stage73 demonstrates which **combined route** is faster, not that Quad or Prepacked **Conv1d individually** are slower. A high cost in Direct ConvTranspose1d can contaminate the prior comparison. Quad and QuadInterior require temporal-packed weight strides and are only selected on the Direct Conv1d dispatcher. `RVC_CONV_WEIGHT_PREPACK=1` remains active.

## Stage74 independent algorithm paths

- Keep existing `RVC_NSF_IMPLICIT_GEMM=1` semantics unchanged for older scripts.
- Add `RVC_NSF_CONV1D_GEMM=1` for residual/noise 1D convolution only.
- Add `RVC_NSF_TRANSPOSE_GEMM=1` for the four upsampling transpose convolutions only.
- Add explicit runtime metadata `nsf_conv1d_execution` and `nsf_transpose_execution`; existing `nsf_conv_execution` reports `mixed` for partial GEMM.
- Package Stage74 Windows real-WAV runner with isolated `baseline` (both GEMM), `prepacked_gemm_transpose`, `packed_gemm_transpose`, `quad_gemm_transpose`, `quad_interior_gemm_transpose`. Optional `-Suite all` adds `gemm_conv_direct_transpose` and `direct_both`.
- Each mode runs full real WAV twice, verifies actual execution metadata, deterministic output, PCM16 RMSE/max error, saves WAV and logs in `RVC_STAGE74_FULL_WAV_RESULTS.zip`.

## Validations and remaining gates

Local Rust 1.98.1 `rustfmt --emit stdout` parser PASS on changed decoder/main. `git diff --check` PASS. **No physical NVIDIA GPU available in this environment**, so no Stage74 hybrid speedup, quality or CUDA JIT success is claimed. Windows CI must compile CPU+CUDA+WGPU EXE and validate PowerShell; RTX 4060 A/B must confirm all claims. Production default and global RMVPE F0 remain unchanged; unsafe Stage72 UNet GEMM still disabled. No CUDA Graph replay integrated.

Windows CMD after downloading `rvc-stage74-nsf-hybrid-windows-x64`:

```cmd
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage74.ps1 -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -WarmRuns 2 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -Suite hybrid
```

Once actual hybrids are measured, select the correct per-family algorithm; next investigate NSF residual hot kernels and bounded CUDA Graph capture/replay with stable buffers, shapes, lifecycle and replay vs eager waveform parity.
