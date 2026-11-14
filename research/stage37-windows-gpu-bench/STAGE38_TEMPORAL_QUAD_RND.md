# Stage 38 — shared CubeCL TemporalQuad experiment

Date: 2026-11-10. Workstream: universal CubeCL NSF Conv1d CPU/CUDA/WGPU. **Experimental**, not production dispatch.

## Baseline and objective

The Stage 37 real RTX 4060 GPU measurements are preserved in [STAGE37_RTX4060_RESULTS.md](STAGE37_RTX4060_RESULTS.md). The existing weight-prepacked temporal-pair kernel is 1.10x slower than strict-FP32 PyTorch/cuDNN at K3, but 1.94x slower at K11/D5 and 1.85x slower at K11. This motivates research into K11 reuse and register-limited output tiling. The baseline includes allocation + synchronous host readback; no pure kernel times have been measured.

## Hypothesis

The existing CubeCL packed-channel kernel creates 2 temporal accumulators per output-channel vector. Its inner loop loads one weight vector for each (kernel offset, input channel) and reuses it across 2 output positions. The new `TemporalQuad` creates 4 accumulator vectors and reuses the weight vector across up to 4 consecutive positions.

For a full tile and ignoring cache effects, the **logical number of weight-vector loads per output time position** is reduced by a factor of 2 relative to TemporalPair. The number of scalar FMAs remains unchanged. More live accumulators may decrease GPU occupancy or cause spilling, so faster execution is **not** implied by this count.

No architecture-specific intrinsics are used. This kernel uses only CubeCL vector operations and the existing device-side packed-weight layout `[groups, K, CinGroup, CoutGroup]`; it is designed to compile across CPU JIT, CUDA, ROCm, and WGPU runtimes. Only CUDA and WGPU compilation have been checked here.

## Changes

- New `vendor/cubek-convolution-0.3.1/src/kernels/forward/direct_prepacked_quad.rs`: four adjacent temporal positions per unit, separate vector accumulators, group/stride/dilation/padding support, incomplete last tile handling. The old TemporalPair implementation remains available for a fair A/B comparison; no automatic dispatch changes.
- Export `launch_prepacked_temporal_quad` from the CubeCL kernel module.
- Extend Windows benchmark to measure Direct, Packed (two positions) and Quad (four positions), alternated in execution order, after the same warmup, including full-array Direct-vs-Quad parity.
- Optional `--parity-dir` writes `.f32le` output tensors for every variant, and Python/cuDNN writes the strict FP32 reference tensors. The new `compare_tensor_dumps.py` reports full-tensor max absolute error, normalized error, MAE, RMSE, and exact-value fraction.
- Fix Windows PowerShell 5.1 native stderr handling by using `Start-Process` and independent stdout/stderr logs. The script can no longer mistake the normal `[cuda] benchmarking...` progress message for `NativeCommandError`.
- Explicit PyTorch TF32 mode `--tf32 allow|disable` (default `disable`); the automated report uses strict FP32.

## Verification performed in this iteration

- `cargo check --offline --locked --no-default-features --features wgpu` from the Stage 37 benchmark manifest: **PASS**.
- The same check with `--features cuda`: **PASS**.
- Independently compiled Rust arithmetic reference: TemporalPair vs TemporalQuad, **3888/3888 configurations exactly identical** for finite deterministic FP32 data over batch, group counts, input/output channel counts, odd lengths, kernel sizes 1/3/5/11, stride 1/2 and dilation 1/2/5. This is **not runtime CubeCL GPU parity**.
- Python comparator: accepts 9/9 matching tensor files, correctly fails a deliberately modified tensor.
- Python script syntax and CLI `--help`: PASS.
- `git diff --check`: PASS before publishing.

## Pending acceptance gates

1. Produce Windows x64 binaries from CI and run on RTX 4060; check full-tensor Direct/Packed/Quad parity, then compare performance by K3, K11/D5 and K11. Do not claim acceleration before these GPU results.
2. Investigate GPU-only timing without CPU readback and repeat in multiple runs, including cold/warm packing.
3. Verify strict PyTorch full-tensor parity and actual GPU device identity for WGPU.
4. Evaluate tile sizes 1, 2, 4, 8 against register pressure and occupancy on CUDA/WGPU. Use measured performance, not a hard-coded assumption, to select an inference plan.
5. Test on actual CrazyMita weights and activation tensors, then integrate into an NSF model-resident packed-weight plan only after end-to-end correctness.

Private model weights/audio/index are not published. Production RVC remains unchanged.
