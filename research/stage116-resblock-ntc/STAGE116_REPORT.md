# Stage116 — shared NTC transport for RVC ResBlocks (Burn / CubeCL)

**Repository:** `Atm4x/Vendor-for-burn`, PR #2. **Git-bundle base:** `e41ce4b2b1d9c5c89b8c4f5cdef7e059817a03bc`. **Rust:** 1.100.0. **Scope:** opt-in research, no production mode switch. All source analysis and changes were made against a **local clone of the verified Git bundle**. No private checkpoint/audio/index included.

## Finding: Stage115 had an invalid contiguity guarantee

`vendor/burn-cubecl-0.22.1/src/kernel/conv/preact_residual1d.rs` used `into_contiguous_aligned` to prepare the residual passed into the direct preactivation-convolution kernel. That function may return a non-contiguous tensor when the backend reports it as readable (`can_read_tensor`). The new Stage115 kernel, however, accesses residuals as a contiguous vector buffer and its launch validator rejects any residual with non-unit channel stride or non-standard NTC strides. In practice the first residual pair of each branch may receive a standard NCT tensor and fail at kernel launch. This is a correctness bug, independent of performance.

Stage116 replaces that call with `into_contiguous`, which **guarantees physical contiguous layout** before the residual is passed to the kernel. This is a precondition fix, not an optimization. The old Stage115 path remains available for comparison, now with valid materialization.

## New algorithm: one shared prepack per DecoderStage

Each of the 3 independent ResBlock1 branches consumes the exact same `prepared = upsampled + harmonic_source` activation. In Stage115, the first residual pair of **each** branch may separately materialize that NCT activation into NTC to feed the fused second convolution. Stage116 introduces an orthogonal switch:

`RVC_NSF_RESBLOCK_SHARED_NTC=1`

At the DecoderStage boundary, it permutes `prepared` to logical NTC, materializes **once** via a generic `burn-cubecl` contiguous operator, and permutes it back to logical NCT **without another copy**. All three branches see the same NTC-backed tensor. Later fused Conv1d operations already emit physically NTC outputs, so the subsequent paired convolutions can reuse their buffers. The optimization is independent of the Stage115 fused-convolution opt-in, so baseline direct, ImplicitGemm, and fused transport can be measured with or without shared NTC prepack.

The new helper is in `burn-cubecl` and uses only generic CubeCL primitives; it has no CUDA intrinsics or backend-specific source.

### Quantitative upper bound for transport-only benefit

For RVC 40 kHz, 128 latent frames, batch=1, stage inputs are C/T approximately `(256,1280), (128,12800), (64,25600), (32,51200)`. Four stage activation buffers total **21.0 MB** (decimal) FP32. If each of three branches would otherwise require one full NCT→NTC copy and Stage116 reduces this to one copy per stage, the **maximum reduction is 84 MB of input/output copy traffic** (two avoided copies × one read and one write), not a reduction of the 56.812 billion ResBlock MACs. If a stage already arrives contiguous in NTC, it needs zero copies and the benefit may be zero. Runtime/Nsight attribution is required to determine which case actually happens.

This is unlikely to close a large gap against highly tuned Tensor Core convolutions by itself. Stage116 exists to isolate and measure the transport penalty instead of hiding it inside an unrelated tiling or GEMM redesign.

## Profiling fixes

Previous `stage115_resblock_first/second` NVTX names selected all stages at once, so `ncu --launch-count` could capture only early 256-channel or K3 kernels. New labels specify exact geometry:

- `stage116_resblock_c128_k11_d5_first`
- `stage116_resblock_c128_k11_d1_second`

The label applies to both baseline and candidate. PowerShell profiling selects the exact geometry with `-ProfileChannels 128 -ProfileKernel 11 -ProfileDilation 5`; `ncu --nvtx --nvtx-push-pop-scope process --nvtx-include <label>/` captures NCU counters for that subgraph. This does **not** identify individual kernels automatically when other operations launch within the range: inspect the Nsight reports.

`Run-Stage116-ResBlock.ps1` supports six independent modes: `baseline`, `baseline_ntc`, `gemm`, `gemm_ntc`, `transport`, and `transport_ntc`, all from the same Rust CLI and checkpoints. `-Profiler measure|ncu|nsys|all`. Baseline and reference WAVs are compared sample-by-sample as **signed PCM16**, with max absolute PCM LSB, normalized RMSE, SNR and mismatch count. SHA equality alone is not the parity gate. The script accepts an externally produced `-ReferenceWav` and optional `-NvtxDllDir` (directory containing `nvToolsExt64_1.dll`).

A new Windows CI workflow, `.github/workflows/rvc-stage116-resblock-ntc.yml`, can build/package the patched Rust CUDA/WGPU executable and validate the PowerShell script and std-only reference. It has **not** been pushed or executed.

## Verification in the current environment

- Rust 1.100.0 `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cuda --lib` with `DOCS_RS=1`: **PASS** after vendored crate rebuild.
- Same `--features cpu --lib`: **PASS**.
- Same `--features wgpu --lib`: **PASS** (fresh WGPU compilation).
- Stage115 std-only Rust reference: **2/2 PASS**.
- Stage116 std-only Rust reference: **2/2 PASS**, including physically lossless NCT↔NTC layout over batch/channel/time shapes and **all three nonlinear K3/K7/K11 ResBlock branches** with 3 residual pairs each.
- `git diff --check`: **PASS**.
- Broad `cargo check --lib --tests --features cpu` is blocked by existing benchmark/fixture targets that require the separate `flex` feature (`burn_flex`, `Device::flex`). This is unrelated to the Stage116 library; do not count it as a complete Rust test pass.
- CPU runtime needs Tracel LLVM 23.x, unavailable here. CUDA/WGPU runtime, CUDA JIT, NCU/NSYS GPU traces, matched full WAV and Windows PowerShell parser: **NOT TESTED**.

## Decision and next experiment

Run matched warm full-WAV RTX4060 measurements first, then capture NCU on `C128/K11/D5` first and `C128/K11/D1` second, plus K11 at C64/C256. Compare baseline to baseline_ntc to quantify shared packing independently of fused transport, and transport to transport_ntc to quantify the same with the Stage115 kernel. Keep identical checkpoint/F0/index/precision/benchmark input and document CUDA clocks. Summarize GPU **device** durations, CTA/register occupancy, L1TEX requests, L2/DRAM traffic, FFMA/TC utilization, stalls and actual launch count. Do not equate Nsight-replay times with uninstrumented warm medians. The `NSF 97.23% MAC share` is not measured time.

If `Conv1d` kernels remain compute/instruction bound and shared packing is negligible, **stop optimizing NTC copies**. The next algorithmic candidate is an output-channel/input-channel shared-tile *implicit convolution with Tensor Core-capable CubeCL matmul*, in which NTC activation and fixed prepacked filters are loaded through the convolution im2col mapper rather than formed as an explicit im2col allocation. This is a different operator from the unsuccessful Stage111 deconvolution-phase GEMM; nevertheless it must beat the current CubeCL direct/ImplicitGemm and strict PyTorch cuDNN at matched numerical tolerance before integration.

No legacy/obsolete files, GitHub branches, or production defaults were modified.
