# Silero V5 RU — fused Flex CPU spectral head

Date: 2026-11-19. Base commit: d7ecc9b4178fa6b82283fb2aaca3730f7917c761.

## Change

The original vocoder spectral head computes magnitude (exp and clamp) and phase sine/cosine using separate Burn tensor operations. The new Flex-only flex_spectral::split_spectrum consumes the channel-last float32 linear projection once, computes magnitude and sin_cos together, and returns three channel-first tensors. Large arrays use Rayon row parallelism. The public TTS contract, CubeCL backends and model weights are unchanged.

The native spectral path is on by default only when feature reference-flex is built and the backend device is Flex. SILERO_FLEX_TENSOR_SPECTRAL=1 restores the original Burn tensor spectral operations. SILERO_VOCODER_TIMING=1 now reports the final projection and spectral stage separately.

## Release benchmark

Test machine: AMD EPYC 9V74 virtual CPU; five visible vCPUs, four-core cgroup quota. Rust 1.100.0, Burn 0.22.1, PyTorch 2.10.0 CPU, four Rayon threads, release build with target-cpu=native.

Test: aidar, 8 kHz; six repetitions of "Привет, мир!" with spaces. Output 43,800 PCM samples = 5.475 seconds. Alternating A/B, six persistent-worker sessions per variant, median of seven warmed calls per session.

| Metric | Original tensor path | Fused spectral |
|---|---:|---:|
| Median spectral stage | 19.535 ms | 2.331 ms |
| Median vocoder stage | 110.668 ms | 92.587 ms |
| Six-series median full inference | 165.0 ms | 149.5 ms |
| Full inference RTF | 0.0301 | 0.0273 |

Spectral stage accelerated 8.38x, while observed full-inference throughput improved 1.104x (9.4% reduction in latency). Host-level noise affects full-inference timings; they are not universal CPU performance guarantees. Worker measurements include Base64 encoding and exclude initial model load.

## Correctness

44/44 release library tests passed, including the new tensor-versus-native spectral test. Six independent real-checkpoint PyTorch comparisons passed the max absolute PCM error threshold of 0.003. The original and fused Rust paths emitted bitwise identical PCM in all six scenarios.

| Voice | Rate | Samples | Max error vs PyTorch |
|---|---:|---:|---:|
| aidar, short | 8 kHz | 7,900 | 0.00004656 |
| kseniya, short | 8 kHz | 9,600 | 0.00006343 |
| xenia, long | 8 kHz | 54,300 | 0.00004850 |
| baya, long | 24 kHz | 147,900 | 0.00002825 |
| eugene, long | 48 kHz | 246,600 | 0.00031400 |
| aidar, long | 8 kHz | 43,800 | 0.00009276 |

Tests requiring SILERO_PARITY_DIR remain fixture-gated. The six PyTorch tests above use actual model weights and are independent of those fixtures.

## Negative GEMM investigation

A separate opt-in trial enabled gemm/x86-v4 on the AVX-512-capable EPYC host. Four A/B release series per mode found a median of 160.5 ms with AVX-512 and 154.5 ms with the existing FMA path. PyTorch PCM parity still passed, but the latency difference did not support an AVX-512 speedup; the experimental feature was dropped and is not committed.

Source inspection also showed that RHS weights already have a favorable x86 stride for typical measured dimensions: gemm avoids RHS packing when rhs_rs=1 and m stays below the configured packing threshold times MR. A generic persistent RHS packing cache would therefore be redundant for those shapes. Dynamic LHS packing remains a distinct issue.

## Reproduction

Build with reference-flex and --release; run tools/benchmark_release_flex.py with the real v5_ru.pt and exported weights, RAYON_NUM_THREADS=4, repeat count at least eight and --max-pcm-error 0.003. Compare with SILERO_FLEX_TENSOR_SPECTRAL=1 and SILERO_TIMING=1 SILERO_VOCODER_TIMING=1. All experiments were performed locally; no weights, binary builds or vendored checksum changes are included in this commit.

## Next area

Once spectral processing is fused, the two 512→1536→512 ConvNeXt pointwise projections still consume the most measured CPU time. Further work should target actual GEMM microkernel behavior and dynamic input layout with paired release and PCM parity tests.
