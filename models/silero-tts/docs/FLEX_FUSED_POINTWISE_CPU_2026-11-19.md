# Silero V5 RU — fused Flex pointwise projection fast path

Date: 2026-11-19. Base GitHub commit `2ca812962e0a992a1a4ad4ee534bb5adcddfe40e` (branch `codex/silero-tts-cubecl-port-20261118`). Model: real offline `v5_ru.pt` exported to SafeTensors. Rust 1.100.0, Burn 0.22.1, native optimized release on virtualized AMD EPYC Linux, cgroup 4 CPU cores, Rayon 4 threads. All runtimes use the same checkpoint and in-process persistent JSONL worker. No CUDA, CubeCL or Python tensor execution in the Rust worker.

## Root performance bottleneck and strategy

Eight ConvNeXt blocks each perform 512→1536→512 pointwise projections. Existing Burn Flex already dispatches to the `gemm` crate with Rayon, and the Burn `linear` implementation folds batches opportunistically; replacing GEMM itself or globally changing its packing threshold is not sufficient.

The new Flex-specific implementation keeps the full operation sequence within host-owned contiguous f32 buffers:

1. Keep depthwise convolution and layer normalization on Burn Flex, then convert normalized `[batch, frames, 512]` to contiguous f32 once.
2. Execute native strided `gemm` for 512→1536, add the original bias, then apply the same parallel fast GELU in place.
3. Execute native `gemm` for 1536→512, add bias, multiply by `gamma` and add residual on the same host buffer.
4. Return one Burn tensor view in the expected `[batch, 512, frames]` layout.

The `PointwiseEngine` enum chooses the Flex-specific implementation *by default* for the `reference-flex` backend and the tensor fallback for other backends. Set `SILERO_FLEX_TENSOR_POINTWISE=1` **before loading** to select the previous Flex tensor path for A/B comparison. Host weight buffers consume the original tensor weights at load time rather than keeping duplicate tensor and native copies. `SILERO_EXACT_GELU=1` retains the original exact-GELU behavior in the tensor-only fallback; both normal paths otherwise use the same fast GELU approximation.

## Performance: alternating A/B trials

Six sequential release runs using the original long 77-character Russian input, aidar, 8 kHz; generated 43,800 PCM samples (5.475 s). Each run contains 5 in-process inference calls, discards the first and reports the median of four warmed calls. The background host is a shared VM so a single lowest value must not be interpreted as a stable outcome.

| Run | Implementation | Warm median, ms |
|---:|---|---:|
| 1 | Tensor | 172.0 |
| 2 | Native fused | 132.5 |
| 3 | Native fused | 152.0 |
| 4 | Tensor | 200.5 |
| 5 | Tensor | 185.5 |
| 6 | Native fused | 158.0 |

Across the three trial-level medians, tensor = **185.5 ms**, native fused = **152.0 ms**, a **1.22× throughput increase / 18.1% latency reduction**. Representative native RTF = **0.02776**, generating around **36.0 s of audio per compute second**; representative tensor RTF = **0.03388** (~29.5×). The run-to-run spread is significant and the result is specific to this virtualized CPU.

Stage-level timed runs show the eight combined pwconv1 stages improving from approximately 46.6 to 27.4 ms and eight pwconv2 stages from 50.0 to 30.6 ms. These stage measures also include per-operation conversions/bias but not identical upstream scheduling, and are diagnostic rather than a controlled microbenchmark. Resident memory measured shortly after model load: tensor ~234.1 MiB versus fused ~237.6 MiB; after one long infer: tensor ~253.9 MiB versus fused ~251.1 MiB. No large persistent duplication of weights remains.

## Numerical correctness

Directly comparing persistent worker PCM on the above 5.475 s audio yields **byte-for-byte equality** for all 43,800 float32 samples between tensor and fused implementations. SHA-256 of both outputs: `be83493cbe230af4cb2bb4b11aa39bb0a107d16c7dc20d26fa078c71a603660e`; max absolute error exactly 0.0.

Real model against independent original PyTorch PackageImporter reference:

| Text / voice | Rate | PCM samples | Max absolute error |
|---|---:|---:|---:|
| `Привет, мир!` / aidar | 8 kHz | 7,900 | 0.00004656 |
| `На двери замок.` / kseniya | 8 kHz | 9,600 | 0.00006343 |
| `Привет, мир!` repeated six times / xenia | 8 kHz | 54,300 | 0.00004850 |
| Same repeated phrase / baya | 24 kHz | 147,900 | 0.00002825 |
| Same repeated phrase / eugene | 48 kHz | 246,600 | 0.00031400 |

All five match sample counts and pass strict max absolute PCM error ≤ 0.003. Additional unit tests cover native GEMM vs tensor linear on multiple uneven sizes and the fused `projection→GELU→projection→gamma→residual` sequence with 2 batches / 7 frames; **42/42 release library tests PASS**. Existing model parity fixture-dependent tests return early without a fixture directory, so the above original PyTorch comparison is the independent evidence for actual weights.

## Reproduction and scope

Build:

```bash
cargo test --offline --release --no-default-features --features reference-flex --lib
cargo build --offline --release --no-default-features --features reference-flex --bin silero-tts
```

Benchmark with `tools/benchmark_release_flex.py`, `--threads 4`, `--repeats 5`, `--reference-pt <v5_ru.pt>`, `--max-pcm-error 0.003`; compare native default against `SILERO_FLEX_TENSOR_POINTWISE=1`. Set `SILERO_VOCODER_TIMING=1` for per-stage diagnostics.

Only Flex receives the native optimization. The unchanged CubeCL code path was not compiled or benchmarked in this iteration, and CPU results cannot be generalized to CUDA. The offline environment needed checksum normalization on vendored dependencies, which is excluded from the published source. No PyTorch dependency is introduced into the Rust runtime.
