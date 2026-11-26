# Silero Flex: fused polyphase PQMF and transformer feed-forward

Base: `6a64a705eaf739616b31dc88899ecb8e01ce5f25`. Rust 1.100.0, AMD EPYC 9V74, Linux x86_64, four CPU threads (four-core cgroup). Fresh Git Bridge checkout and cold offline release build. Official checkpoint SHA-256 verified. PyTorch 2.10.0+cpu is only an independent test/benchmark oracle; the worker continues to load the original checkpoints and perform all inference natively.

## Architecture

- English PQMF: for input subband sequence `x[i,j]`, upsampler weights `U[i,o,q]` (including the original factor four), and synthesis filter `G[o,k]`, combine the two linear operations before inference. For output phase `p`, input displacement `d` in `[-8,8]`, and input band `i`, precompute `H[p,i,d] = sum_{o,q} G[o,4*d+q-p+31] * U[i,o,q]`, excluding indices outside `[0,62]`. The original length is exactly `4*T`; zero-padded `x` represents the finite support at both ends. One GEMM `[4,68] * [68,T]` reads overlapping windows of the padded channels-last input and returns chronological PCM without the intermediate upsampled tensor or per-tap GEMMs. This is valid even for dense `U`, not just the official sparse filter. FP32 accumulation order can differ.
- Shared acoustic transformer FFN: prepare KIO weights at load time. The K=9 projection uses a padded channels-last sliding window with stride equal to hidden width, followed by bias and ReLU. The K=1 projection uses a standard GEMM, then fuses bias and the residual. `FeedForward` selects exactly one weight representation at construction. Both EN and RU use the Flex representation; other devices retain the Burn representation. No reduced precision, runtime dependency, new crate, or change to checkpoint/IPC format.
- `SILERO_EN_PROFILE=1` emits per-phase latency to stderr on Flex only; stdout remains IPC-only. `SILERO_EN_BASELINE_PQMF=1` and `SILERO_BASELINE_TRANSFORMER_FFN=1` are explicit developer A/B controls selected at load, not automatic error fallbacks.

## Measurements

Text: `Hello, world! This is a test of the native English speech synthesis.` Speaker `en_0`, duration 5.3875 s. Different CPU from the previous cycle: do not compare independently measured Xeon and EPYC times as a speedup. All times are release worker `compute_ms`; no WAV serialization or process startup. Separate independent A/B runs can show substantial host variability.

### PQMF stage, before FFN change

Per-phase medians from five warmed requests, same binary with PQMF switched at load (separate runs):

| Rate | Old PQMF ms | Fused PQMF ms |
|---|---:|---:|
| 8 kHz | 1.380 | 0.318 |
| 24 kHz | 8.371 | 0.554 |
| 48 kHz | 15.242 | 1.282 |

The separate paired seven-request full-graph PQMF-only A/B gave 82.500 -> 81.591 ms, 196.980 -> 176.714 ms, and 302.460 -> 265.023 ms. PCM16 was bit-exact at every rate for this text. These are different runs from the phase medians.

### Combined optimization, paired old revision vs final candidate

Nine warmed requests per process, alternating order. Same checkpoint and text. The paired PCM comparison enforces identical sample count, per-worker deterministic output, and error <=16 LSB; observed max <=8 LSB.

| Rate | Base ms | Final ms | Speedup | Max PCM16 LSB | SNR dB |
|---|---:|---:|---:|---:|---:|
| 8 kHz | 83.338 | 56.556 | 1.47x | 3 | 86.82 |
| 24 kHz | 193.153 | 165.041 | 1.17x | 8 | 84.22 |
| 48 kHz | 384.052 | 357.302 | 1.07x | 7 | 84.38 |

FFN-only paired A/B (seven warmed requests): 78.243 -> 56.945 ms, 219.395 -> 213.814 ms, 391.930 -> 378.032 ms. RU `На двери замок. Мы проверяем скорость синтеза речи.`, aidar, 24 kHz, 3.9875 s: 120.792 -> 118.082 ms, max 3 LSB, SNR 79.62 dB. Small gains at 24/48 kHz and RU should not be overinterpreted given host variability.

### Independent PyTorch oracle, final candidate

| Rate | Flex ms | PyTorch ms | Flex/PyTorch | PCM16 SNR dB |
|---|---:|---:|---:|---:|
| EN 8 kHz | 72.002 | 55.826 | 1.29x | 76.62 |
| EN 24 kHz | 211.152 | 149.476 | 1.41x | 76.05 |
| EN 48 kHz | 296.500 | 259.815 | 1.14x | 76.15 |
| RU 24 kHz | 109.225 | 228.683 | 0.48x | 69.97 |

Seven warmed calls for EN, five for RU, four threads. Same sample counts. This is not a paired old/new benchmark; do not combine these medians with the table above. EN 8/24 kHz still exceed the 1.2x target on this host. Residual blocks remain the largest vocoder cost after PQMF; the pre-FFN acoustic stage also occupied roughly 30 ms at every rate.

## Validation

- Locked offline release build, 58 unit tests passed; separate real-checkpoint parity test passed. New synthetic PQMF covers dense filter coefficients, batch 1/2 and 10 lengths (20 cases). New FFN covers hidden/inner sizes (1,1), (3,7), (64,64), (128,512), batch 1/2 and seven lengths (56 cases).
- Independent official EN fixture: two texts, two speakers, three sample rates; exact duration and length, max mel error 1.335e-5, max isolated vocoder error 2.945e-5, max full FP32 PCM error 2.041e-4; all cases within existing 1e-3 PCM criterion.
- Shared IPC tests 2/2, legacy worker tests 4/4 including real EN and RU lifecycle, explicitly invoked ignored real-checkpoint tests.
- Previous Windows release `6a64a705` completed dedicated CI and real-checkpoint smoke tests. Downloaded ZIP manifest revision and executable SHA-256 independently matched. The current revision still requires its own Windows build.

Reproduce using `tools/benchmark_flex_ab.py` and `tools/benchmark_native_ipc.py --pytorch-oracle`; set `SILERO_EN_PROFILE=1` and capture stderr for phase timings. `tools/create_english_parity_fixture.py` creates independent test-only fixtures. Neither the fixtures nor the official checkpoint is committed.

### GEMM without AVX-512

For a local validation build only, removed `gemm/x86-v4` from `reference-flex`; `cargo tree -e features -i gemm-f32` confirmed only `std` and `rayon`. Restored the unmodified product manifest afterwards. Offline release and 58 unit tests passed; real EN/RU IPC lifecycle and official EN FP32 fixture passed (max PCM error 1.347e-4). Five-request alternating comparison against the product build on the same EPYC:

| Rate | Product dispatch ms | No AVX-512 GEMM ms | Max PCM16 LSB |
|---|---:|---:|---:|
| 8 kHz | 78.205 | 65.961 | 1 |
| 24 kHz | 192.402 | 204.487 | 3 |
| 48 kHz | 360.496 | 339.219 | 4 |

The mixed timing direction is not evidence that AVX2 is intrinsically faster. The CPU exposes AVX-512, and other SIMD routines may still use it; this is a GEMM feature-configuration test, **not** a physical AVX2-only CPU or i5-13600KF validation. Product runtime SIMD dispatch is unchanged.
