# Silero V5 RU — native Flex CPU inference optimization

Date: 2026-11-19. Environment: Rust 1.100.0, Burn/Flex 0.22.1, Linux x64 AMD EPYC virtual CPU, five visible vCPUs with a four-core cgroup quota. Real checkpoint: `v5_ru.pt`, converted to SafeTensors; `--release --no-default-features --features reference-flex`, `RAYON_NUM_THREADS=4`. The original PyTorch 2.10 CPU reference used four threads.

## Root causes

1. The generic first-port ISTFT performed two `[1201, 2400]` dense DFT basis matrix multiplications for every output frame. This is an `O(T·N²)` algorithm with subsequent tensor `slice_assign` overlap-add operations.
2. ConvNeXt uses eight 512-channel blocks with pointwise expansion to 1536 channels. A stage profile on 5.475 seconds of audio attributed approximately 133 ms to the two pointwise projections across eight blocks and 99 ms to exact GELU (`libm::erff`), while depthwise convolution took only about 2 ms.
3. The Flex backend was compiled with `burn/flex` alone. Because Burn is consumed with `default-features = false`, `burn-flex/simd` and `burn-flex/rayon` were **not** enabled. Thus large GEMM operations did not use Flex's intended parallel SIMD configuration.

## Changes

- Flex-only native mixed-radix FFT (RustFFT) performs the mathematically exact-size, non-power-of-two `N=2400` inverse transform: reconstruct Hermitian spectrum from `[B, 1201, T]`, inverse FFT, multiply periodic Hann window, accumulate overlap and normalize by squared-window envelope. FFT plan is created once at model load. The original Burn-tensor DFT remains available for CubeCL CPU/CUDA/WGPU. Device selection is runtime-aware even if both feature families are compiled.
- Flex-only fast GELU implements an Abramowitz–Stegun `erf` approximation and avoids expensive per-element exact `libm::erff`. The default Flex path uses the fast method; `SILERO_EXACT_GELU=1` selects original exact GELU for A/B. CubeCL continues to use exact GELU. A dense scalar comparison over [-10, 10] passes within `2e-6` absolute error.
- The optional `burn-flex` dependency explicitly activates `std`, `simd` and `rayon`, without pulling `burn/ndarray` multithreading dependencies absent from the local vendor archive. `RAYON_NUM_THREADS` controls parallelism; four threads worked best under the four-core cgroup quota.
- Optional `SILERO_TIMING=1` and `SILERO_VOCODER_TIMING=1` print stage/operator timings for future profiling. No debug output is produced by default.

## Results

All results use the same model weights, text, `aidar`, 8 kHz, and 5.475 seconds of audio (`43,800` PCM samples):

| Variant | Long-text inference | RTF | Notes |
|---|---:|---:|---|
| Original Flex release | 391–412 ms | 0.0714–0.0753 | Dense DFT, exact GELU, no SIMD/Rayon |
| Native FFT only | 336–363 ms | 0.0614–0.0663 | ISTFT falls from ~51 ms to ~3.4 ms |
| FFT + fast GELU | 278–299 ms | 0.0508–0.0546 | GELU falls from ~99 ms to ~19 ms |
| FFT + fast GELU + SIMD/Rayon, 4 threads | **168–209 ms** | **0.0307–0.0382** | Repeated local release runs, variable CPU contention |
| Original PyTorch CPU, 4 threads | 200–218 ms | 0.0365–0.0398 | Same checkpoint and input, warmed |

These are single-host measurements rather than a statistically controlled cross-machine performance claim. A new independent benchmark with three Rust repeats yielded 187, 207 and 196 ms, and the paired PyTorch warmed median was 218 ms. The prior baseline warmed median was 409.5 ms, an approximately **2.0–2.3×** improvement in end-to-end latency, depending on host noise.

Approximate bottleneck changes (long utterance): dense DFT 50.9 -> native FFT 3.4 ms (~15×); exact GELU 98.9 -> fast GELU 18.6 ms (~5.3×). SIMD/Rayon primarily accelerate GEMM projections; depthwise conv is not the central bottleneck.

Short text is still latency-dominated: `а` yields 2600 samples at 8 kHz, with typical optimized 46–51 ms (RTF ~0.14–0.16). Larger utterances amortize fixed overhead.

## Numerical parity

- Direct baseline Rust vs FFT-only WAV: <=1 int16 LSB on 43,800 and 2600 samples.
- Direct baseline Rust vs final optimized WAV: <=5 int16 LSB on the long sample, <=1 int16 LSB on one character.
- Real PyTorch reference, 8/24/48 kHz one character: max abs PCM errors 0.000151 / 0.000098 / 0.000148; exact sample counts.
- `Привет, мир!`, 8 kHz: 7900 samples, max abs 0.001426.
- `На двери замок.`, 8 kHz: 10100 samples, max abs 0.000456.
- Repeated 77-character phrase, 8 kHz: exact sample count (43,800), high correlation (>0.99993), RMSE ~0.00159, **but max abs error ~0.0699, failing the strict 0.003 criterion**. This discrepancy predates all optimizations; its origin has not been localized. Do not claim full strict numerical parity for longer texts.

All **38 Rust library tests passed** when original checkpoints were re-exported to the parity fixture directory, including actual vocoder/PQMF parity and new multi-batch FFT and scalar GELU tests. The optimizations do not use PyTorch at inference time.

## Reproduction

```bash
cargo build --offline --release --no-default-features --features reference-flex --bin silero-tts
RAYON_NUM_THREADS=4 python tools/benchmark_release_flex.py \
  --binary /path/to/target/release/silero-tts \
  --model-dir /path/to/silero/assets \
  --text 'Привет, мир! Привет, мир! Привет, мир! Привет, мир! Привет, мир! Привет, мир!' \
  --sample-rate 8000 --repeats 5 --threads 4 --timeout 40 \
  --reference-pt /path/to/v5_ru.pt --output-json flex-optimized.json
```

`--reference-pt` is optional and used only for development-side PyTorch parity. Without it, the tool requires only Python's standard library. Use `--exact-gelu` to benchmark exact GELU in the same optimized binary. No other backend features are required for this Flex binary.

## Follow-up R&D

If more acceleration is required, profile pointwise GEMM and tensor allocations in the eight ConvNeXt blocks rather than the depthwise convolutions. For CubeCL GPU, replace the old tensor DFT/overlap-add with a dedicated non-power-of-two FFT and a fused overlap-add kernel; the CPU RustFFT fallback cannot substitute for a GPU kernel. Separately investigate the pre-existing large-utterance PCM mismatch at the mel and phase output boundaries.
