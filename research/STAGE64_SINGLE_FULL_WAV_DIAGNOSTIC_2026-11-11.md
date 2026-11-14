# Stage 64 — Single full-WAV diagnostic (CUDA baseline)

## Purpose

Find the actual end-to-end bottleneck behind Stage 63's ~0.45 RTF on RTX 4060 (versus the user's previous PyTorch RVC baseline of >15x realtime). This stage makes **no performance claims** and adds no model optimization. It captures a full diagnostic trace for exactly **one measured entire-WAV inference** in exactly **one configuration**.

## Fixed configuration

- Backend: CUDA, single unified CPU/CUDA/WGPU binary (runtime selection).
- Model: CrazyMita.pth, hubert_base.pt, rmvpe.pt, CrazyMita.index.
- Full input: MitaKind.wav (~89.94 s); no input trimming.
- FAISS: original CPU IVF, `index_rate=0.75`; not experimental GPU IVF.
- F0: original CPU FP64, dense RMVPE salience readback.
- Prepacked Conv1d: enabled. CUDA Graph: not wired into full RVC, disabled.
- Chunking: core 300 frames, overlap 12, context 40, trailing 4; tensor batch 2.
- One representative-shape warmup batch (same tensor batch and chunk shape), excluded from the single measured full pass.
- **Exactly one measured full-file inference.** No mode sweep, no multi-round comparison.

## Trace

The EXE emits `RVC_TRACE` newline-delimited JSON to stderr, including phase (`setup`, `warmup`, `measured`, `postprocess`), millisecond timestamp, batch ID, tensor shapes and per-stage durations. The Windows script exports:

- `benchmark.stderr.log` / `benchmark.stdout.log`: complete untouched executable logs.
- `stages.jsonl`: machine-readable structured records, including per-batch nested stages.
- `stage-time-summary.csv`: grouped times **from measured phase only**; nested stages must NOT be added to their parent totals.
- `gpu_process_telemetry.csv`: process RSS and CPU time, GPU utilization/memory utilization, used VRAM, power, SM clock, temperature, sampled every ~750 ms.
- `nvidia-driver-and-device.txt`: full device and driver snapshot.
- `rvc-doctor.txt`: compiled backends.
- `file-sha256.json`: SHA-256 for EXE, four model/index files and full input WAV (no files uploaded to CI).
- `run-configuration.json`: exact arguments and environment.
- `single-full-wav-summary.json` + `converted-full.wav.diagnostic.json`: wall-clock timing, chunk count, output length, RTF, warmup/load/prepare/postprocess.
- `converted-full.wav`: actual converted output, which can be audited or listened to.

Main measured spans:

- Audio H2D, HuBERT frontend Conv stack, HuBERT positional Conv and Transformer stack, HuBERT D2H.
- RMVPE CPU mel/FFT, mel H2D, RMVPE U-Net, bidirectional GRU, output head, salience D2H and FP64 F0 decode.
- CPU FAISS, pitch/protect, phone and pitch H2D, TextEncoder front, per-layer attention/FFN, stats projection.
- Latent and speaker preparation, Flow, NSF random/source generation, decoder conv_pre, each of four upsample/resblocks, conv_post and waveform D2H.
- Batch starts/ends and chunk completion with exact lengths.

Every traced GPU span synchronizes the selected device before recording the time, so attribution does not simply move deferred work to the following stage. Both enqueue+CPU time and synchronization time are recorded. **This intrusive tracing changes execution and RTF**; it is intended for bottleneck attribution only, and must not be compared directly against uninstrumented Stage 63 or PyTorch speed. The tool does not claim kernel-level GPU Nsight timing, detailed allocation counts or PCIe bus counters. GPU telemetry is sampling-based and may miss short bursts.

## Build/test evidence

- Rust 1.98.1 `cargo check --locked --offline -p tts-with-rvc-rs --no-default-features --features cpu,cuda,wgpu --bin tts-with-rvc-rs`: PASS.
- `cargo check --all-targets` without `flex`: fails in old, unused benchmark binaries expecting Flex; the production binary itself passes. Do not rewrite obsolete scripts simply to satisfy all-targets.
- Windows CI builds only the unified EXE, statically checks the PS runner, calls `doctor`, then packages EXE + runner. No real GPU test and no private model files in CI.
- Windows RTX 4060 run remains required before interpreting performance or claiming acceptance.

## Command

From the unpacked `rvc-stage64-single-full-diagnostic-windows-x64` artifact with a sibling `embed` folder containing the private assets:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Diagnostic-Stage64.ps1 -Voice '.\embed\CrazyMita.pth' -Hubert '.\embed\hubert_base.pt' -Rmvpe '.\embed\rmvpe.pt' -Index '.\embed\CrazyMita.index' -InputWav '.\embed\MitaKind.wav' -CoreFrames 300 -TensorBatchSize 2
```

Result: `RVC_STAGE64_FULL_DIAGNOSTIC_RESULTS.zip`.

## Decision gate

After inspecting the diagnostic ZIP, rank the actual full-route stages by measured time, split compute from transfer/sync and pre/postprocessing, and identify a single dominant optimization target. Only then develop the required fix. Client-side integration must remain separate from unresolved high-cost inference bottlenecks. No assertion that Stage 64 itself closes the performance gap.
