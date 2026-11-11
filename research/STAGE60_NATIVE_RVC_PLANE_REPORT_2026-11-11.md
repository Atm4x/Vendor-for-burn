# Stage 60 — CUDA warp-per-frame RMVPE F0 readback (experimental)

Date: 2026-11-11. Repository: `Atm4x/Vendor-for-burn`, experimental PR #2.
Baseline: published Stage 59. Rust 1.98.1; Burn 0.22.1 / CubeCL 0.11.3.

## Hardware evidence received: Stage 59 (RTX 4060)

The user's `RVC_STAGE59_RMVPE_FUSED_RESULTS.zip` contains six successful real-model RVC CUDA runs, two for each mode. Every output WAV has SHA-256 `4F54A58F5C0910DBA8863812BAC5782689BC438B2A18C77B0318AE6A5FEF9793`. Each has 70 F0 frames; index_rate=0.75, input segment=0.7 s, CUDA Graph disabled.

| Mode | RMVPE runs, ms | Median, ms | Total runs, ms | Logical RMVPE host bytes |
|---|---|---:|---|---:|
| Dense | 855 / 872 | 863.5 | 4071.4148 / 4015.6918 | 100800 |
| Burn Compact | 977 / 907 | 942.0 | 4051.0676 / 4036.2638 | 2800 |
| Stage59 Fused | 872 / 864 | 868.0 | 4042.8439 / 4048.5987 | 2800 |

Fused vs Burn Compact: median RMVPE time ~7.85% lower (942 / 868 ≈1.085×, or (942-868)/942≈7.86% shorter). Fused vs Dense: ~0.52% higher RMVPE time. Full end-to-end timings around 4.04 s; this two-round study does not establish an end-to-end improvement. RMVPE times include neural processing and therefore are not an isolated measurement of the readback kernel. The old Dense implementation must remain the default.

## Experimental Stage60 implementation

`models/tts-with-rvc-rs/src/rmvpe_fused.rs` adds `plane_salience_kernel`, separate from the Stage59 sequential fused kernel. On CUDA one 32-lane plane processes each 360-bin F0 frame: each lane scans every 32nd bin, `plane_max` reduces the peak, `plane_min` chooses the earliest index among ties and `plane_any` detects non-finite values. The same ten-FP32 format (peak index + nine neighbours) is returned; the existing FP64 CPU F0 conversion remains unchanged. The kernel explicitly handles negative-only and `-f32::MAX` inputs and preserves tensor strides.

`RVC_RMVPE_PLANE_READBACK=1` selects this experiment. It is mutually exclusive with `RVC_RMVPE_FUSED_READBACK=1` and `RVC_RMVPE_COMPACT_READBACK=1`. Normal operation keeps Dense. The GPU pathway never falls back silently to CPU. CUDA `CubeDim::new_1d(32)` supplies one plane per frame. A separate Windows runner compares Dense, Burn Compact, Stage59 Sequential Fused and Stage60 Plane in order-reversed rounds, checking full WAV SHA and reported host bytes.

### CPU/LLVM limitation — verified, not concealed

CubeCL 0.11.3's CPU/LLVM JIT **cannot lower plane operations**: it rejected `plane_any` with `plane operations need a plane of more than one unit`. Consequently the Stage60 plane kernel is explicitly **CUDA-only**; selecting it with CPU returns an error. CPU tests validate the mathematics using a 32-lane simulator, independently comparing exact ten-FP32 output against the executable Stage59 sequential kernel and bitwise F0 against Dense. They do not execute the actual plane kernel.

## Checks

- `cargo test --release --offline --locked -p tts-with-rvc-rs --features flex,cpu --lib`: **35 pass / 0 fail / 4 intentionally ignored**.
- Opt-in real RMVPE test with private `rmvpe.pt` and `MitaKind.wav`: **PASS**, 36 and 71 frames, FP64 F0 matches Dense bitwise; the host-simulated 32-lane compact result matches the executed Stage59 CubeCL CPU output.
- `cargo check --offline --locked -p tts-with-rvc-rs --features flex,cuda --all-targets`: **PASS**.
- `git diff --check`: **PASS**.
- CUDA kernel **runtime** and latency on RTX 4060: **NOT TESTED**. NVIDIA hardware not present in the local Linux environment; do not claim accelerated inference or full WAV parity for the new plane path.

## Local Windows A/B validation

Workflow `.github/workflows/rvc-rmvpe-plane-stage60.yml` packages artifact `rvc-stage60-rmvpe-plane-windows-x64`. CI validates the PowerShell parser and CUDA build, but cannot validate runtime arithmetic without RTX hardware and the private models. The standalone runner consumes the local operator's private files, does not upload them, and creates `RVC_STAGE60_RMVPE_PLANE_RESULTS.zip`.

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-RVC-RMVPE-Plane-Stage60.ps1 -Voice "D:\Models\CrazyMita.pth" -Hubert "D:\Models\hubert_base.pt" -Rmvpe "D:\Models\rmvpe.pt" -Index "D:\Models\CrazyMita.index" -InputWav "D:\Models\MitaKind.wav" -Duration 0.7 -Device cuda -Rounds 2
```

Also test 0.35 and 2 seconds, index_rate=0 and 0.75, and reversed order. If even a single WAV hash differs or NVIDIA compilation fails, leave the new mode disabled and investigate the trace without enabling a fallback.

## Boundaries / next steps

1. Compare plane and sequential fused kernel with CUDA Events or Nsight Systems/Compute; full RMVPE stage latency is dominated by many other operators. Measure warm/resident inference rather than cold model loading if evaluating deployable latency.
2. If parity and a stable improvement are proven, consider moving F0 decode to the new mode; otherwise retain Dense. Do not compromise first-maximum ties, NaN/Inf rejection, FP64 centroid math, or `nprobe` FAISS semantics.
3. Return to original objective: a native RVC inference model integrated through NeuroMita's **single shared TTS service worker**, rather than creating per-model workers or Python/native switching. Do not merge experimental PR #2 into `main` without end-to-end GPU proof.

No private model weights, index or audio are included in this source commit.
