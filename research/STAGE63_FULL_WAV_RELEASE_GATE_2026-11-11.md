# Stage 63 — full-WAV RVC acceptance gate before NeuroMita client integration

## Motivation and corrected performance interpretation

Stage 62 compared warmed execution of a short 0.68 s fragment on the same RTX 4060 through two different GPU backends. The CUDA case did **not** use CUDA Graph (`graph_replay=false`); consequently the 248 vs 254 ms observation does not establish that WGPU beats CUDA Graph. Only seven warmed iterations were used for each mode, with no variance-controlled multi-length evidence. The CPU-FAISS configuration remained faster than the first GPU-IVF implementation for that workload. It is not defensible to extrapolate that result to a ~90 s recording or to claim a production throughput gain from avoiding the HuBERT readback.

## New end-to-end full-WAV benchmark

`tts-with-rvc-rs infer-wav-full-benchmark` uses the **production** `infer_long_filtered_16khz_batched` path and its overlap-add chunking with context, reading the entire input WAV, resampling it to 16 kHz and applying 48 Hz high-pass once. All RVC, HuBERT, RMVPE and optional FAISS weights are loaded **once** into a single `RvcV2Inferencer` in the process. Each iteration processes the **entire** PCM, synchronizes device execution, validates finite floats and exact output length, then reports cold-first and warmed timings, RTF, chunk counts, iteration-to-iteration equality, total source/output duration and postprocessing/WAV write time. First inference may include GPU shader JIT and allocation overhead. Warm RTF excludes model load, I/O and preparation. Output WAV is normalized/matched in the same manner as normal `infer-wav-full`.

The Stage63 Windows runner executes this mode from ONE unified CPU/CUDA/WGPU EXE, `-Device cuda` by default, with the user files inside `./embed/`. There is no implicit CPU fallback. The default `-Modes all` compares CPU FAISS+CPU F0 (baseline), resident IVF+CPU F0, resident IVF+GPU F0, all with the exact same FULL source WAV and chunk configuration. `-WarmRuns 2` is the minimally useful first/warm protocol; increase to `3` or `5` for statistics after initial validation. `-Modes baseline` isolates the trusted production reference. For WGPU use the **same artifact** and `-Device wgpu`.

Each full mode emits a WAV, independent benchmark JSON, stdout and stderr logs, and the PowerShell runner additionally emits summary with WAV SHA-256, global PCM16 comparison and optional NVIDIA driver metadata. The result archive contains private converted output audio but **never** gets uploaded into GitHub or distributed to other users. The model files are not packaged.

### Copy/paste command

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage63.ps1 -Voice '.\embed\CrazyMita.pth' -Hubert '.\embed\hubert_base.pt' -Rmvpe '.\embed\rmvpe.pt' -Index '.\embed\CrazyMita.index' -InputWav '.\embed\MitaKind.wav' -Device cuda -WarmRuns 2 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -Modes all
```

## Product acceptance criteria — not another microbenchmark roadmap

1. **Functionality:** end-to-end rendering from untrimmed source with exactly sized finite 40-kHz WAV, correct chunk count and no gaps in overlap-add; no silent switching between backends.
2. **Reproducibility:** model load once; first and subsequent full calls reported separately. Report whether repeated renders are bit-identical and capture if not. Different source WAVs and audio lengths remain additional release tests.
3. **Numerics:** baseline CPU-FAISS/F0 is reference. GPU-IVF and GPU-F0 remain opt-in until whole-recording PCM and listening review show acceptable differences. A low short-fragment RMSE does not prove perceptual equivalence over 90 s.
4. **Performance:** compare whole-recording warm RTF and load/JIT separately, on the same hardware. `graph_replay=false` remains explicit: earlier synthetic NSF CUDA Graph was never wired into full inference. Graph integration is a follow-up optimization, NOT included in the claimed timings.
5. **Packaging:** one Windows EXE with CubeCL CPU/CUDA/WGPU, explicit `-Device` and no GPU SDK installation requirement for typical WGPU/D3D12 use; exact driver/runtime requirements for CUDA remain dependent on the shipped package and target system.
6. **Client boundary:** typed long-lived `RvcV2Inferencer` with `load` + `infer_long_filtered_16khz_batched` is ready to be wrapped by NeuroMita's **TTS worker**. The worker should own the model and device lifetime and offer load/unload, infer, cancel-before-next-chunk, progress and shutdown. The existing CLI is not yet a robust IPC worker, and mid-kernel cancellation is not promised.

## Evidence and limitations

- Prior Stage62 real NVIDIA measurements: ~248–254 ms per 0.68-s warmed chunk with CPU FAISS; no graph replay, no 90-s measurement. CUDA versus WGPU winner not established.
- This Stage63 adds the full-WAV benchmark and CI packaging. CPU/CUDA/WGPU static compile checks can pass locally, but **the final complete 90-second CUDA/WGPU inference results depend on running the attached Windows EXE on the physical RTX 4060**. CI Windows build does not execute GPU benchmarks.
- The long CPU end-to-end attempt using the existing Stage62 executable is tracked separately; it is not a Stage63 CUDA result. The Rust release rebuild may exceed the local resource/time budget despite passing `cargo check`.

## Recommendation

If the baseline full-WAV run satisfies the functional gates and the full-CPU-FAISS path meets user-visible latency and audio quality requirements, proceed to the client/worker work. Keep opt-in IVF/F0 numeric gaps and optional real NSF CUDA Graph on an R&D backlog rather than indefinitely blocking an end-user prototype. Do not report that full RVC is "CUDA-Graph optimized" until the actual production graph capture is implemented and demonstrated.

## Stage63 local verification at authoring time

- Rust 1.98.1, unified default-feature `cargo check --offline --locked -p tts-with-rvc-rs --bin tts-with-rvc-rs`: PASS.
- Rust 1.98.1, CPU-only `cargo build --release --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu --bin tts-with-rvc-rs`: PASS.
- Unified **release** rebuild locally: timed out while compiling the GPU/LLVM dependency graph; not evidence of a runtime failure. Windows CI is the release gate.
- Complete source WAV: 3,966,323 frames at 44,100 Hz = 89.9393 seconds. Stage62 CPU-only full-WAV attempt ran to the 165-second resource limit without producing a final WAV; therefore full-route CPU success is NOT claimed.
- New Stage63 full-WAV CPU smoke tests also exceeded local execution limits before producing a final artifact. `cargo check` validates the source, not dynamic CPU/LLVM JIT or GPU hardware output. All hardware acceptance must use actual Stage63 CI-produced EXE and attached results.
