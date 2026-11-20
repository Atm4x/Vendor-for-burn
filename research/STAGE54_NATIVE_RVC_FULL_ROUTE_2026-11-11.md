# Stage 54 — Full RVC native CUDA wiring and cross-stage Conv1d layout

Date: 2026-11-11. Repository: `Atm4x/Vendor-for-burn`, experimental PR #2. All changes are experimental and source-only.

## Objective

Move beyond synthetic NSF into a native full-route **HuBERT → RMVPE → FAISS retrieval → RVC TextEncoder → reverse Flow → NSF waveform**. Keep RVC weights, input WAVs and retrieval index local, never in Git/GitHub artifacts. Python and PyTorch are **not needed for native inference**.

## Real asset validation (local Rust reader)

`tts-with-rvc-rs inspect-assets` successfully opened all four user-supplied files and validated compatibility:

| Asset | Verified properties |
|---|---|
| `CrazyMita.pth` | 457 tensors, RVC v2, 768 content channels, 109 speaker embeddings, 40 kHz/F0 from safe metadata inspection |
| `hubert_base.pt` | 214 tensors, first convolution `[512,1,10]` |
| `rmvpe.pt` | 741 tensors, first convolution `[16,1,3,3]` |
| `CrazyMita.index` | native IVF Flat L2, dimension 768, 61,690 vectors, 1,581 inverted lists, nprobe=1 |
| `MitaKind.wav` | source mono PCM16, 44,100 Hz, approx. 89.94 s; resampling required at runtime |

Real-file validation currently confirms the native loaders can read required file formats and selected weight shapes. It **does not** prove all model layers are numerically correct, CUDA runtime works, complete model inference succeeds, or audio sounds acceptable.

## Native changes

1. `--device cpu|cuda|wgpu` selector in the active `user_cli.rs`, enabled only when the corresponding Rust feature is compiled. Explicit unsupported backend returns error. The Windows Stage54 binary is built **with default Flex CPU plus `cuda` feature**, keeping one EXE able to choose CPU or CUDA.
2. Existing research `infer-wav-segment` and `infer-wav-full`/`infer-wav-batch` accept an optional trailing backend; the primary `convert` and `convert-batch` use `--device`.
3. One `conv1d_layout` module owns the Standard / Prepacked / Temporal weight transforms and loads directly from **host-side checkpoint data**. It eliminates the previous upload→GPU readback→repack→upload path for NSF weights and makes Prepacked available for HuBERT frontend and Flow. Grouped HuBERT positional Conv1d is deliberately not altered without dedicated group-parity proof. The temporal Packed/Quad modes are still selected **only for NSF**; they require matching CubeCL kernel/layout geometry.
4. FAISS native IVF retrieval uses Rayon across independent HuBERT frames for batches of ≥32, preserving existing search order, per-frame normalization and output. The small input path remains serial.
5. `inspect-assets` command checks native checkpoint/index compatibility without loading all model tensors to GPU.
6. Windows workflow builds `tts-with-rvc-rs.exe` with `--features cuda` and runs CLI tests. It packages the `Run-RVC-Full-Stage54.ps1` hardware runner (without any private model files). That runner converts a 0.2–9 s segment, records per-stage RVC profile and generates a result ZIP.

## Existing vs. new CUDA Graph scope

The Stage51–53 **synthetic NSF profiler** implements dynamic hardware CUDA Graph replay with persistent input buffers. This optimization **has not been wired into production `RvcV2Inferencer`**, which still executes normal Burn tensor operations. The real-model Stage54 runner reports `graph_replay=false` to prevent confusing GPU-native execution with CUDA Graph acceleration. A safe production migration must respect model and audio length-dependent shapes, input buffer lifetime, stream ordering, source and conditioning buffer updates, and bounded graph cache; no silent fallback may count as graph acceleration.

`RVC_CONV_WEIGHT_PREPACK=1` is now a cross-stage Conv1d optimization (NSF, HuBERT frontend, reverse Flow). `RVC_NSF_CONV_KERNEL=packed|quad|quad_interior` uses specialized CubeCL temporal kernels for NSF only. Neither setting changes ConvTranspose1d or RMVPE Conv2d automatically.

## Local tests and hardware limits

- Actual file/header and native reader validation: **PASS**.
- Rust 1.98.1, `cargo check --no-default-features --features cuda`: **PASS**.
- Rust 1.98.1, CUDA-only CLI tests: **2 PASS, 0 FAIL**.
- Rust 1.98.1, Flex CPU library regression suite: **17 PASS, 0 FAIL**, including 9 Conv1d geometry combinations and sequential-vs-parallel FAISS parity.
- Combined Flex + CUDA (`--features cuda`) compilation on Rust 1.98.1: **PASS**.
- Full RVC CUDA execution on user RTX 4060: **not yet measured**; GitHub Actions Windows runners do not provide the user's GPU.
- End-to-end PyTorch numerical/audio comparison: **not established**; Stage53 covers only synthetic NSF.
- Runtime installation, persistent IPC lifecycle and NeuroMita Core orchestration: **not implemented in Stage54**.

## User run (after Windows Stage54 CI artifact is built)

Extract artifact `rvc-stage54-native-full-cuda-windows-x64` and run in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage54.ps1 -Voice 'D:\models\CrazyMita.pth' -Hubert 'D:\models\hubert_base.pt' -Rmvpe 'D:\models\rmvpe.pt' -Index 'D:\models\CrazyMita.index' -InputWav 'D:\models\MitaKind.wav' -Start 0 -Duration 2 -Device cuda -Variant prepacked
```

Replace paths with actual paths. The runner creates `RVC_STAGE54_RESULTS.zip` with stage timings, logs and output 40-kHz WAV. It accepts baseline/packed/quad/quad_interior alternatives for controlled comparison. **If the GPU runtime fails, do not infer success from successful CI compilation.**

## Next engineering tasks

1. Run real-model Stage54 hardware test on RTX 4060 with baseline then prepacked; inspect failures before applying additional optimizations.
2. Add per-stage GPU timing with CUDA events and explicit synchronization. Separate one-time weight load/warmup from steady-state requests. Audit expensive mid-graph CPU transfers (HuBERT outputs, RMVPE F0, FAISS retrieval).
3. Design per-shape persistent CUDA Graph only where deterministic model inputs/outputs, buffers and stream lifetime can be proven safe. Handle cache invalidation and differing segment lengths.
4. Optimize RMVPE Conv2d and HuBERT grouped positional conv only after representative profiler measurements and parity. Do not reuse NSF temporal kernels for incompatible convolution geometries.
5. Expose a versioned, long-lived model EXE IPC protocol (`load`,`warmup`,`infer`,`status`,`cancel`,`unload`,`shutdown`) and integrate it via a top-level NeuroMita orchestrator. Reuse compatible external NVIDIA runtime DLLs rather than adding a required Python installation.
