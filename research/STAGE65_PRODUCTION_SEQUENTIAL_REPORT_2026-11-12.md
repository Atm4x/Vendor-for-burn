# Stage 65 — Sequential NSF is the production policy

Branch: `experiment/conv-kernel-rnd-stage33-20261110`. Parent commit: `4510c0e1b079cb13ac028523f21fe82bfb9c08a0`.

## Corrected production routing

- One source of truth: `ResidualExecution::PRODUCTION = ResidualExecution::Sequential`.
- Default `DecoderStage::forward`, `NsfWaveformDecoder::forward`, and `NsfWaveformDecoder::forward_from_f0` now use this policy.
- The real `RvcV2Inferencer` explicitly invokes `forward_from_f0_with_execution(..., ResidualExecution::PRODUCTION)`, so the old implicit Parallel full-RVC path is removed.
- Parallel remains available only for explicit `forward_with_execution` research / A-B comparisons, not for production inference.
- Added `production_residuals_are_sequential` regression test.

## Evidence, not an unmeasured performance promise

- Stage44 CubeCL CPU synthetic NSF: Parallel/batch8 to Sequential/batch32 accelerated 64-frame forward 6.07x in one measured run (107.942 ms to 17.770 ms). Both scheduling and batch policy changed.
- Stage50 RTX4060 synthetic NSF: Sequential+Prepacked over Parallel+Direct 2.80x at 64 frames, 2.00x at 256, 1.85x at 1024. Relative gains on full real RVC are unmeasured.
- Stage64 62.06-second single-WAV diagnostic inserted GPU sync and is not an uncontended speed reference. Earlier Stage63 production WAV was approximately 45 seconds without diagnostic trace, but with background Unity load.
- Synthetic CUDA Graph was never enabled in production. No new Graph / GRU / Conv kernels were introduced by Stage65.

## Stage65 Windows gate

`Run-RVC-Full-Stage65.ps1` uses the original entire-WAV benchmark API, `RVC_CONV_WEIGHT_PREPACK=1`, CPU FAISS and CPU FP64 F0 in default baseline mode, no diagnostic trace and no graph replay. It records first and warmed full-waveform inference separately.

Unlike Stage64 and Stage63 scripts, it reads process stdout/stderr asynchronously using `System.Diagnostics.Process` and checks a real ExitCode, avoiding the Start-Process null ExitCode failure. Experimental NSF and RMVPE environment overrides are reset then restored.

The internal raw JSON `stage=63` is the unchanged full-WAV benchmark schema; Stage65 `summary.json` identifies `stage=65` and `production_residual_execution=sequential`.

## Validation at authoring time

- Downloaded and SHA256-verified latest available successful `rvc-experiment-git-bridge` Git bundle from workflow run `42926555249`, stage52 commit `e9134c27`. The source was checked out with `git clone`.
- Read and patched the **current PR #2 head** `4510c0e1b079cb13ac028523f21fe82bfb9c08a0` through GitHub Connector; the Git bridge artifact's older HEAD is not used as the remote commit parent.
- Local Rust 1.98.1 rustfmt parser accepted the changed decoder and inference on restored source.
- Local offline `cargo check` reached dependency compilation but exceeded execution timeout; does NOT count as PASS.
- Stage65 CI checks PowerShell syntax, compiles unified CPU/CUDA/WGPU release executable, runs production policy test and packages the EXE. CUDA hardware is not available in CI.
- No private model, index, or source WAV files are committed. No real RTX 4060 Stage65 run yet; full-WAV throughput and parity remain to be measured.

## Windows RTX 4060 run

From the extracted GitHub Actions `rvc-stage65-full-wav-unified-windows-x64` artifact, place all private assets in `.\embed\` and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage65.ps1 -Device cuda -Voice '.\embed\CrazyMita.pth' -Hubert '.\embed\hubert_base.pt' -Rmvpe '.\embed\rmvpe.pt' -Index '.\embed\CrazyMita.index' -InputWav '.\embed\MitaKind.wav' -WarmRuns 2 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -Modes baseline
```

Output: `RVC_STAGE65_FULL_WAV_RESULTS.zip`. Benchmark includes one first and one warmed entire-WAV pass, exact output sample count, logs, summary and WAV. Do not confuse `WarmRuns=2` with two warmup calls.

Next step: measure Stage65 without Unity, compare Stage63 on the same idle RTX4060, then profile any remaining NSF direct conv/ConvTranspose and RMVPE BiGRU performance bottlenecks before further structural changes.
