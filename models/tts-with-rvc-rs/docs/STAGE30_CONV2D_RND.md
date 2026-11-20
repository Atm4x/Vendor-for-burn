# Stage 30 — Universal Conv2d R&D and real RVC release validation

Date: 2026-11-09. Branch: `experiment/conv2d-universal-implicit-gemm-20261109`.

## Design

This remains a single public native `Conv2d` implementation with no legacy Conv3d fallback. Its spatial matrix contraction has two CPU microkernels, selected by matrix geometry: analytical row spans for `Cout/group <= 2`, `K <= 32` in f32/f64; bounded NCHW packing plus GEMM for larger matrix tiles and f16. The numerical `f16` reduction remains in full GEMM, rather than overflowing during kernel-position accumulation. Shared spatial indices are reused across batch/groups when bounded by 1,048,576 indices. Tile scheduling permits grouped tasks to run across Rayon workers; nested GEMM parallelism is disabled when tile tasks are parallel.

The numerical goal is preserved for stride, dilation, padding, groups, batch, partial final tiles, and non-finite zero-padding semantics. Choosing a matrix microkernel is not fallback to the old Conv3d implementation. This is still experimental because small depthwise shapes remain slower than the legacy path.

## Build and tests

The modified vendored Burn Flex was rebuilt for `release` with Rust 1.98.1, offline Cargo, two build jobs, without a full `cargo clean`. Because Cargo failed to notice vendor-file edits, only the `burn-flex` release fingerprint was invalidated; all other compiled dependencies were retained.

- `cargo test -p tts-with-rvc-rs --release --lib --offline --locked`: 15/15 PASS.
- `cargo test -p tts-with-rvc-rs --release --test parity --offline --locked`: 30/30 PASS.
- `bench_stage30_conv2d_matrix` release: 6 finite randomized-config cases PASS, plus non-finite padding test PASS. Maximum absolute deviation from legacy Conv3d in the six finite cases: 1.49e-7.
- `bench_conv2d_ab_stage29` release: same-process alternating A/B on actual compiled Burn; the detailed results are in the external evidence logs.

## Conv2d, release (2 Rayon threads, median of 20)

| Shape | Legacy Conv3d | Stage30 | old/new |
|---|---:|---:|---:|
| 256→256, 8×8, 3×3 | 1.34545 ms | 0.92009 ms | 1.46× |
| 512→512, 4×4, 3×3 | 3.14814 ms | 1.01454 ms | 3.10× |
| 16→16, 16×16, 3×3, depthwise groups=16 | 0.02545 ms | 0.04697 ms | 0.54× |

The depthwise regression remains open: the new kernel improved the Stage29 universal version (~0.118 ms in its original 2-thread run), but does not match the legacy depthwise path. Stronger universal small-microkernel / scheduler research is required; do not merge solely on the present benchmarks.

## Real RMVPE

The same 1.15-second audio sample was processed using `rmvpe.pt` with the final native Conv2d kernel. Seven measured forward passes after a warmup:

- 2 Rayon threads: median 158.232 ms (min 147.311; max 191.868).
- 4 Rayon threads: median 138.141 ms (min 135.355; max 146.704).
- 116 F0 frames, 88 voiced. Exact F0 match between this Stage30 build at 2 and 4 threads, and zero difference from the earlier Stage30 universal version.

The old Stage27 and Stage29 binaries were also rerun sequentially. Their observed timing varied notably with load, so these figures should **not** be interpreted as a controlled end-to-end speedup relative to PyTorch or previous stages.

## Real RVC WAV → WAV

- 10.1-second sample, `CrazyMita.pth` + `hubert_base.pt` + `rmvpe.pt` + `CrazyMita.index` + PCM, batch size 2, 2 Rayon threads: completed.
- Elapsed wall-clock including model load: 27.04 s; peak RSS: 1,396,404 KiB; output: 404,000 int16 samples at 40 kHz.
- Compared to the preserved Stage24 10.1-second output: RMS error 3.10e-6 (normalized PCM), MAE 3.14e-7, correlation 0.9999999989, no clipping or invalid values.
- Full 89.94-second input: stopped at the 245 s guard while processing chunk 28/32. No complete output was written. This is a **timeout**, not a completed benchmark and not an out-of-memory error. Do not claim this acceptance gate passed for Stage30.

## Remaining gates

1. Eliminate the depthwise regression without falling back to legacy Conv3d, and retain generic non-finite, grouped, and f16 semantics.
2. Compare against PyTorch at steady CPU load with controlled 2- and 4-thread sampling, rather than relying on cross-session figures.
3. Complete a real full-length WAV→WAV run on an uncontended machine; optionally implement resumable chunk checkpoints in the user-facing CLI to avoid losing work on a timeout.
4. Extend dtype/property-based tests (`f16`, `f64`, `bf16`, 0-channel, extreme padding, zero-width) before any production promotion.
5. CPU profiling of Rayon task overhead vs GEMM on small grouped contractions; then CUDA/WGPU testing independently.

Private checkpoints, indexes, F0 arrays, and audio remain local, outside Git.
