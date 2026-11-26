# English HiFi-GAN: phase GEMM and runtime SIMD dispatch

Baseline: `7841f8e483670f408335194901ab72e5d2c68906`. Rust 1.100.0; Linux x86_64 Intel Xeon Platinum 8573C; four Rayon threads, official original v3_en/v5_ru. Release build, no Python or LibTorch in worker. PyTorch 2.10.0+cpu is an independent benchmark/test oracle only.

## Design

Replace each English residual-stage dense convolution on Flex with a channels-last phase GEMM. Weight layout is transformed once from OIK to KIO. For dilation d, time indices with the same residue modulo d form an independent undilated sequence. Pad each sequence by (K-1)/2 frames; a K*C-wide read-only GEMM operand with column stride C represents overlapping windows, without explicit im2col. Each phase writes disjoint output columns into the final activation. Leaky ReLU is fused into phase packing; bias and residual are fused in the output pass. The three branches preserve their original accumulation order. A single phase scratch allocation is reused; tensors cross the host/Burn boundary once per residual stage. GPU and other backends retain the Burn tensor graph. `SILERO_EN_BASELINE_CONV=1` is an explicit developer A/B override, not an error fallback.

Tiled im2col and concatenating all phases in one GEMM were tested but rejected: neither reliably improved 48 kHz relative to phase GEMM. Enable the existing `gemm/x86-v4` feature in reference-flex; GEMM detects AVX-512 at runtime, otherwise uses its normal FMA/scalar dispatch. No new crates, no changed weights, no reduced precision. No new global compiler target-feature flags. The non-AVX-512 feature configuration was tested before enabling dispatch; actual hardware without AVX-512 remains to be verified with the final package.

## Measurements

Text: `Hello, world! This is a test of the native English speech synthesis.` Speaker `en_0`, 5.3875 s PCM. Separate warmed IPC workers, alternating order, seven requests per candidate, four threads. Medians are from paired measurements; host variability prevents combining independently measured medians as if they were a single controlled comparison.

| Rate | Previous (ms) | Phase only (ms) | Phase+AVX512 (ms) | Previous vs phase+AVX512 |
|---|---:|---:|---:|---:|
| 8 kHz | 208.231 | 195.046 | 104.354 | 2.00x |
| 24 kHz | 537.858 | 482.476 | 274.964 | 1.96x |
| 48 kHz | 942.509 | 623.654 | 427.725 | 2.20x |

The phase-only column is from a separate 3-way experiment; corresponding baselines were 245.706, 668.576, 1005.349 ms. Final phase+AVX512 column was measured on the feature-equivalent experimental binary; final manifest build was separately checked against the oracle below.

| Rate | Final Flex median (ms) | PyTorch median (ms) | Flex/PyTorch | PCM16 SNR |
|---|---:|---:|---:|---:|
| 8 kHz | 145.732 | 118.150 | 1.23x | 76.73 dB |
| 24 kHz | 267.667 | 209.543 | 1.28x | 75.95 dB |
| 48 kHz | 426.294 | 427.144 | 1.00x | 76.06 dB |

Paired AVX512 vs phase-only on RU (`На двери замок. Мы проверяем скорость синтеза речи.`, aidar, 24 kHz): 182.784 -> 153.382 ms (1.19x); PCM16 max error 3 LSB. On EN 24/48 kHz additional AVX512 speedup was 1.40x/1.85x in five-request comparisons, affected by host variability.

## Validation

- Final locked release build, 56 unit tests passed, no failed tests, one real-checkpoint parity test separately executed. 216 convolution boundary cases; full three-branch stage with batch=2.
- Official v3_en FP32 oracle: 6/6 text-to-PCM and 6/6 isolated vocoder cases passed. Max PCM error 1.56732e-4, max isolated vocoder error 2.6628e-5. Same lengths and durations. With AVX512 disabled the corresponding max PCM error is 2.38402e-4.
- IPC real-checkpoint lifecycle and invalid-state/corrupt-checkpoint tests: 2/2 passed on final release.
- Previous vs feature-equivalent optimized binary PCM16 max error 1/6/4 LSB at 8/24/48 kHz, SNR 90.80/86.60/86.80 dB. Each worker output deterministic across repetitions.
- Previous Windows package (7841f8e4) completed CI and real-checkpoint smoke testing. The optimized Windows package must be rebuilt and verified; Linux results are not Windows verification.

Reproduce paired timings with `tools/benchmark_flex_ab.py BASELINE CANDIDATE CHECKPOINT --text ... --speaker en_0 --rate 24000 --threads 4 --repeats 7`. Use `tools/benchmark_native_ipc.py` with `--pytorch-oracle` for independent comparison. Further priority: profile remaining acoustic/upsampling/PQMF costs on the optimized graph, especially 8 and 24 kHz; validate Windows AVX2-only hardware.
