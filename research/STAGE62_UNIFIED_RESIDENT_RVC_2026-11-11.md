# Stage 62 — unified CubeCL runtimes, resident F0 and honest warmed inference

## Scope and evidence

Branch: `experiment/conv-kernel-rnd-stage33-20261110`, PR #2 in `Atm4x/Vendor-for-burn`. This stage fixes runtime/backend fragmentation and the Stage61 WGPU shader failure, adds a GPU-resident RMVPE F0 path, and supplies a genuine persistent-model inference benchmark. Do not confuse static compilation with hardware validation or CPU simulation with AMD GPU verification.

Private model assets (`CrazyMita.pth`, `CrazyMita.index`, `hubert_base.pt`, `rmvpe.pt`, `MitaKind.wav`) are NEVER committed or included in CI artifacts.

## Architecture

- Cargo defaults now include `cpu,cuda,wgpu`: one EXE includes all three CubeCL device runtimes. They share the same low-level `burn_cubecl::Cube` primitive, removing the old compile-time `Cuda`/`Cpu`/`Wgpu` alias precedence in `faiss_gpu.rs` and `rmvpe_fused.rs`.
- The CLI still requires explicit `-Device cpu|cuda|wgpu`; it never silently substitutes another backend. WGPU (D3D12/Vulkan depending on the platform) is the practical AMD route; native ROCm is not integrated or tested in this Windows package.
- GPU-IVF index data are allocated persistently in model lifetime, without HuBERT host readback for `RVC_GPU_FAISS=1`. The public index supports exact `nprobe=1` only; this experimental path is OFF by default.
- `rank_kernel` no longer generates `f32(NaN)` in WGSL. Its inverse-square-distance weighting is normalized relative to the smallest nonzero selected distance, preventing reciprocal overflow in the finite, valid-distance case. This is mathematically equivalent but can differ by FP32 rounding from historical CPU FAISS; exact PCM parity is NOT claimed.
- `RVC_GPU_F0=1` performs salience peak/9-bin F0 calculations on CubeCL and exponentiation + pitch/coarse pitch generation on the same Burn device, without transferring the RMVPE salience `[B,T,360]` or subsequent F0/pitch to the host. `RVC_GPU_FAISS=1` is required when index_rate>0; otherwise it errors instead of silently reverting to CPU. CPU FP64 remains the default trusted reference. The GPU path is **experimental** and does not claim exact FP64 or audio parity before real GPU validation. It emits a negative sentinel for invalid salience, whose handling will need stricter validation before production use.
- Remaining CPU-side components: source WAV decoding/resampling/high-pass and input chunk scheduling; random noise initialization; final waveform readback and WAV writing. Full asynchronous service queue, persistent graph capture, and native ROCm are not completed by this stage.

## Benchmark semantics

`infer-wav-segment-benchmark` loads RVC, HuBERT, RMVPE and FAISS exactly once; prepares PCM exactly once; calls the same inferencer repeatedly; synchronizes before and after each inference (the inference already reads back the final audio). It writes `<output.wav>.benchmark.json` with `load_ms`, `prepare_ms`, every `inference_ms`, `warm_median_ms` excluding the first inference, backend and GPU mode. There is no CUDA Graph replay in this mode. This is an honest warmed end-to-end-inference measurement, unlike Stage61's separate-process `cold_total_ms`.

`RVC_PROFILE=1` now times the actual GPU IVF blend with synchronization, not just the previous zero-work preparation step. This profiling synchronization is deliberately EXCLUDED from normal warmed benchmarking, since it changes execution scheduling. Do not equate an enqueue-only number to GPU execution time.

## Validation gates

- Unified Rust 1.98.1 `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu,cuda,wgpu --bin tts-with-rvc-rs`: **PASS**.
- Default-feature single EXE `cargo check --offline --locked -p tts-with-rvc-rs --bin tts-with-rvc-rs`: **PASS**.
- CubeCL CPU/LLVM library regression suite: **33 PASS, 0 FAIL, 4 ignored opt-in tests** (`cargo test --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu --lib`). The resident FP32 F0 vs FP64 synthetic comparison passed on variable batch/frame sizes, with max allowed absolute difference 0.03 Hz.
- Windows CI: requires successful unified release EXE build and an artifact containing one EXE and `Run-RVC-Stage62-Resident.ps1`.
- RTX 4060 warmed resident A/B: **NOT RUN IN THIS ENVIRONMENT**. WGPU shader runtime on NVIDIA/AMD: **NOT YET VERIFIED**. CUDA/FP32 F0 WAV parity: **NOT YET VERIFIED**.
- Stage61 WGPU crash fix specifically requires runtime WGSL compilation; `cargo check` alone cannot prove success.

## Windows operator command

From the Stage62 artifact folder, copy private models and input WAV under `.\embed\` and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Stage62-Resident.ps1 -Voice '.\embed\CrazyMita.pth' -Hubert '.\embed\hubert_base.pt' -Rmvpe '.\embed\rmvpe.pt' -Index '.\embed\CrazyMita.index' -InputWav '.\embed\MitaKind.wav' -Duration 0.7 -IndexRate 0.75 -Device cuda -WarmRuns 8
```

Change `-Device wgpu` for WGPU (NVIDIA/AMD hardware validation), or `-Device cpu` for direct CubeCL CPU. The script runs three modes (CPU index, resident IVF, resident IVF+F0) in distinct processes, with **all warm iterations happening within each process and its loaded model**. Reports the load time, first-inference time, warmed median and pairwise PCM16 differences. The result is `RVC_STAGE62_RESIDENT_RESULTS.zip`; it includes the per-mode warm JSON. No private weights are archived.

## Remaining blockers

1. Confirm the changed WGPU `rank_kernel` JIT compiles and executes. CPU/CUDA/WGPU compile success is NOT sufficient; errors can surface only at final tensor readback.
2. Compare intermediate IVF center IDs, top-eight IDs, weights and blended features to explain the observed Stage61 max PCM16 difference of 9. Do not enforce exact identity if the GPU arithmetic legitimately differs; use documented numeric tolerances.
3. Establish GPU F0 numeric and coarse pitch equality on the real RMVPE salience before enabling by default. FP64 is not a model requirement but the original host reference.
4. Instrument CUDA/WGPU device timings and VRAM, not just wall-clock timings. Analyze the 194.6 MB resident IVF allocation.
5. Make the inference worker a long-lived app-level service with input lifetime/cancellation contracts for eventual NeuroMita Rust integration. This stage includes a long-lived inferencer **within each benchmark process**, not yet a full IPC worker.
