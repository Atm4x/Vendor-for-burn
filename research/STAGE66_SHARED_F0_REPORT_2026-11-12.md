# Stage 66 — shared RMVPE F0 timeline A/B

PR: https://github.com/Atm4x/Vendor-for-burn/pull/2
Parent Stage65: `c23ef42266eca363d12b286b93f3c4549ba4ce9d`.

## Why

The Python `tts_with_rvc/infer/vc/pipeline.py` extracts F0 once on the padded entire recording, then passes pitch slices to voice-conversion segments. Prior Rust production `infer_long_filtered_16khz_batched` instead invoked RMVPE on every overlapping chunk batch. Stage64 traced RMVPE 14.24 s of the 62.06 s diagnostic full-WAV pass, including 9.53 s in its manually unrolled bidirectional GRU. These diagnostic measurements contain GPU sync and do not predict the uninstrumented Stage66 speedup.

## Code changes

- Opt-in `RVC_GLOBAL_F0=1` builds a continuous 100 Hz F0 timeline with original RMVPE weights before chunked HuBERT/NSF. The global timeline is calculated using consecutive 900-frame cores each surrounded by 40 frames of context on each side, with each window never exceeding 9.8 seconds of audio input. The bidirectional GRU still recomputes each window's halo; this is not a true full-length single recurrent pass.
- A shared core of `RvcV2Inferencer` accepts aligned precomputed F0 rows and **skips redundant per-chunk RMVPE execution** while retaining pitch shift, voiced/unvoiced protection, coarse pitch quantization, FAISS, the regular TextEncoder/Flow/NSF, and seeded source noise.
- Source PCM chunk intervals and overlap-add remain unchanged. F0 slice extraction uses frame indices reflected at timeline boundaries. Added Rust tests for boundary frame alignment, overlap and partial end segments.
- `RVC_GPU_F0=1` combined with shared F0 fails explicitly. Direct short-audio calls are unchanged.
- Benchmark JSON includes `global_f0` and `nsf_execution`; Stage66 compares legacy and shared paths, with warm full-WAV latency/RTF and PCM16 RMSE vs legacy.

## Safety and limits

The default remains the **validated legacy per-chunk F0** until the new route passes real audio-quality and latency A/B. A bidirectional GRU's prediction changes when its temporal context changes, so Stage66 shared F0 may not yield byte-identical PCM. Compare carefully near 9-second window boundaries and listen for F0 discontinuities. No claim of full-WAV speedup or numeric equivalence is justified before a real RTX4060 run.

Stage65 warmed 89.94s full WAV was ~44.47s (RTF 0.4945) under concurrent GPU load. NSF is still the majority of diagnostic time, and Stage66 does not change the RVC synthesis mathematics, use CUDA Graph, fuse GRU, enable FP16 or eliminate overlap in HuBERT/NSF. This stage isolates redundant RMVPE and tests whether removing it helps end-to-end. If shared F0 is slower or degrades audio, leave it opt-in and prioritize fused GPU GRU and device-timed NSF kernels.

## Windows Stage66 artifact

Workflow `.github/workflows/rvc-stage66-shared-f0.yml` builds single CPU/CUDA/WGPU release EXE, checks PowerShell syntax, runs Rust scheduling/global-F0 unit tests and packages `Run-RVC-Full-Stage66.ps1` without private weights or recordings.

Copy the private checkpoint, HuBERT, RMVPE, index and original WAV into `.\embed\` inside the extracted artifact. From Windows CMD (double quotes required for paths):

```cmd
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage66.ps1 -Device cuda -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -WarmRuns 2 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -F0Modes both
```

This will perform TWO complete full-WAV passes in each of TWO separate modes: `legacy` and `shared`. The result `RVC_STAGE66_FULL_WAV_RESULTS.zip` contains each mode's WAV, first and warmed inference time, source sample count, logs, RTF and PCM difference. The internal per-mode benchmark uses unchanged stage=63 schema; runner summary identifies Stage 66. No GPU benchmarking is possible in GitHub Windows CI runners.

## Validation and next gates

The locally modified source was parsed by Rust 1.98.1 rustfmt and `git diff --check` passed. Full offline CPU dependency build repeatedly exceeded the local command time budget; **no local compilation or test PASS is claimed**. Windows CI is mandatory before asking the user to run the new artifact.

Compare Stage66 `legacy` WAV SHA256 with Stage65 for the same source, options, device and deterministic settings; verify shared F0 finite length and waveform stability; compare quality and warmed performance on an idle RTX4060 before promoting global F0 into production.
