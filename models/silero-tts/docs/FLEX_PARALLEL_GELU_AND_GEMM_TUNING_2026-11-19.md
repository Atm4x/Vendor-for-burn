# Silero Flex CPU — parallel GELU and GEMM tuning

Date: 2026-11-19. Real Silero v5_ru.pt checkpoint and exported SafeTensors; release Rust 1.100.0, Burn/Flex 0.22.1; AMD EPYC shared virtual CPU with a 4-core cgroup quota. `RUSTFLAGS='-C target-cpu=native'`, `RAYON_NUM_THREADS=4`, PyTorch CPU 2.10 reference.

## Accepted optimization

The existing Flex-only Abramowitz–Stegun GELU evaluation uses `f32::exp` per activation and previously traversed the activation vector serially. The new `apply_fast_gelu` dispatches to `rayon::par_iter_mut` on arrays with at least 32,768 elements; smaller arrays retain the serial path. `SILERO_SERIAL_GELU=1` disables parallelism as an A/B control. The exact GELU backend and CubeCL paths are not modified.

On the 5.475-second audio workload (`Привет, мир!` repeated six times, `aidar`, 8 kHz), four alternating release trials measured the total time for GELU in the eight ConvNeXt blocks:

| Trial | Mode | Warm median TTS latency | Median sum of eight GELU stages |
|---|---|---:|---:|
| 1 | Serial | 202 ms | 18.958 ms |
| 2 | Parallel | 190 ms | 8.046 ms |
| 3 | Serial | 190 ms | 18.033 ms |
| 4 | Parallel | 194 ms | 7.452 ms |

The GELU stage consistently improves by approximately 2.3–2.5x, corresponding to 10–11 ms saved per long utterance. Overall TTS differences are smaller than host timing fluctuations, so no statistically conclusive full-TTS speedup is asserted. Both implementations produce bit-identical 43,800-sample PCM (SHA-256 `be83493cbe230af4cb2bb4b11aa39bb0a107d16c7dc20d26fa078c71a603660e`). The new Rust test also compares serial and parallel GELU bitwise on 131,072 synthetic f32 values.

## Cross-voice PCM parity

These independent worker-to-PyTorch comparisons use the real original checkpoint, not synthetic fixtures. All passed `max_abs <= 0.003` with exact output lengths:

| Case | Speaker | Rate | Max absolute PCM delta | RMSE |
|---|---|---:|---:|---:|
| `Привет, мир!` | aidar | 8 kHz | 0.00004656 | 0.00000351 |
| `На двери замок.` | kseniya | 8 kHz | 0.00006343 | 0.00000581 |
| `Привет, мир!` × 6 | xenia | 8 kHz | 0.00004850 | 0.00000269 |
| `Привет, мир!` × 6 | baya | 24 kHz | 0.00002825 | 0.00000134 |
| `Привет, мир!` × 6 | eugene | 48 kHz | 0.00031400 | 0.00000983 |

The library's release suite passed 40/40 tests. Fixture-gated tests are distinct from the independent real-checkpoint tests above.

## Rejected experiment: GEMM RHS packing threshold

The Flex backend already delegates pointwise linear projections to the `gemm` crate with Rayon. An exploratory *uncommitted* tuning hook tested `gemm::set_rhs_packing_threshold` at 0, 4, 16, 32, 128 (default) and 256. Across runs with four threads, the sum of the eight `pwconv1` and eight `pwconv2` calls remained roughly 85–98 ms; neither forced RHS packing nor a higher threshold demonstrated stable improvement. The hook and extra direct `gemm` dependency were removed from the accepted changes.

The pointwise projections still dominate CPU vocoder time. Next work should investigate persistent weight packing or fused/mixed-precision GEMM in a separate opt-in implementation, always checking numeric parity on actual audio. Do not assume changing the matrix-packing threshold alone is productive.

## Build and measurements

```bash
cargo test --offline --release --no-default-features --features reference-flex --lib
cargo build --offline --release --no-default-features --features reference-flex --bin silero-tts
RAYON_NUM_THREADS=4 python tools/benchmark_release_flex.py \
  --binary target/release/silero-tts --model-dir /path/to/weights \
  --reference-pt /path/to/v5_ru.pt --threads 4 --repeats 6 \
  --max-pcm-error 0.003 --text 'Привет, мир! Привет, мир! Привет, мир! Привет, мир! Привет, мир! Привет, мир!'
```

Run the same command with `SILERO_SERIAL_GELU=1 SILERO_VOCODER_TIMING=1` for a serial stage-profile control. Local changes to vendored `.cargo-checksum.json` were environment-only, not part of the published source.
