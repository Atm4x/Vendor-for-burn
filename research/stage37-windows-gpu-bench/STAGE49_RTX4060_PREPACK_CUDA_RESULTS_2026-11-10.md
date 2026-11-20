# Stage 49 — RTX 4060 CUDA A/B: persistent Conv1d weight layout

Date: 2026-11-10  
Repository: `Atm4x/Vendor-for-burn`  
Experimental PR: https://github.com/Atm4x/Vendor-for-burn/pull/2  
Source: user-supplied `NSF_PREPACK_CUDA_RESULTS.zip` (includes `summary.json`, 4 independent process JSON files, complete FP32 tensor dumps, stderr logs and GPU diagnostics).

## Executive finding

The static-weight prepack experiment passes **bitwise full-output FP32 parity** on all 3 synthetic NSF input sizes across both baseline and prepacked processes. On 64 frames, both process comparisons favor prepacking by **1.11–1.15×**. On 256 frames, both favor prepacking, but the result varies from **1.18× to 1.98×**. On 1024 frames, results **contradict each other** (0.62× and 1.86×), preventing a general speedup claim. This is a promising optimization of the data preparation / convolution route, **not a confirmed CUDA performance improvement across all sizes**.

No changes to production selection are justified on these measurements alone.

## Benchmark design

GPU: NVIDIA GeForce RTX 4060, Windows / CUDA. `nvcc` release 12.8; `nvidia-smi` reports NVIDIA driver 610.88. The benchmark uses a **synthetic NSF checkpoint**, `ResidualExecution::Sequential`, 20 measured iterations after 5 warmups for each length (64, 256, 1024 frames). Full-forward latency includes CUDA/Burn host dispatch plus synchronous CPU output readback. The time to load the decoder is measured separately and excludes checkpoint file loading and device creation.

Four **separate processes**, in exact launch order:

1. `baseline1`
2. `prepacked1`
3. `prepacked2`
4. `baseline2`

The A/B process pairs use different time periods and device state. They are not interleaved per iteration. Results should therefore be treated as descriptive rather than controlled estimates.

## Full-forward results

Times are the profiler's reported warmed medians (ms, lower is better). Speedup = baseline / prepacked.

| Input frames | Baseline 1 | Prepacked 1 | Speedup 1 | Baseline 2 | Prepacked 2 | Speedup 2 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.3208 | 1.1522 | 1.146× | 1.3881 | 1.2460 | 1.114× |
| 256 | 1.3376 | 1.1340 | 1.180× | 2.2110 | 1.1189 | 1.976× |
| 1024 | 1.4501 | 2.3513 | 0.617× | 2.2371 | 1.2017 | 1.862× |

### Strong drift inside a single process

To expose within-process regimes, these are **conventional medians of iterations 1–10 and 11–20** (ms); unlike the benchmark's selected upper median, the halves use the arithmetic middle-pair median. This minor median-definition difference does not change conclusions.

| Input frames | Process | First 10 | Last 10 |
|---:|---|---:|---:|
| 64 | baseline1 | 1.336 | 1.311 |
| 64 | prepacked1 | 1.142 | 1.135 |
| 64 | prepacked2 | 1.226 | 1.208 |
| 64 | baseline2 | 1.390 | 1.371 |
| 256 | baseline1 | 1.330 | 1.336 |
| 256 | prepacked1 | 1.128 | 1.135 |
| 256 | prepacked2 | 1.149 | 1.104 |
| 256 | baseline2 | 2.208 | 2.186 |
| 1024 | baseline1 | 2.448 | 1.412 |
| 1024 | prepacked1 | 2.390 | 1.483 |
| 1024 | prepacked2 | 1.249 | 1.177 |
| 1024 | baseline2 | 2.264 | 2.209 |

Some outliers are severe: `prepacked1` at 64 frames has a **17.372 ms** single sample; `baseline1` at 1024 has a **13.088 ms** sample. Both change the means but not the median as much. The distinct, long-lived ~1.2 ms and ~2.2 ms timing regimes are more important than isolated spikes.

The 1024-frame result cannot establish whether weight prepacking is faster. This requires within-process alternating comparisons with stable allocations and GPU clocks.

## Initialization time (ms)

| Process | Decoder load |
|---|---:|
| baseline1 | 98.2667 |
| baseline2 | 95.1043 |
| prepacked1 | 83.4765 |
| prepacked2 | 84.6334 |

These numbers **do not prove that prepacking accelerates initialization**: only two samples per mode, measured in distinct processes, and the scope omits device creation and checkpoint reading. The current prototype of prepacking uses a GPU-to-CPU readback (`weight.into_data()`) and uploads the repacked tensor to the target device; do not infer that such round trips are free or optimal. A future prepack implementation should operate on the device and persist the final layout.

## Numeric accuracy

All four output buffers for each tested shape are **bit-identical FP32** (not merely within tolerance). Repeated forward outputs in each process also pass parity with reported `repeated_max_abs_error=0`.

| Frames | Output elements | SHA-256 (all 4 dumps) |
|---:|---:|---|
| 64 | 256 | `881d3edfb6561766ad7a1fd63a809b30f879facdc3a3f35c236aa983bb3445f7` |
| 256 | 1024 | `c37aa8f35dd2f98ba5e7ee6da89f9d67022a82ca5fc45f76ef6c96976d8227ef` |
| 1024 | 4096 | `2c48014fef74b3e083c6fd5583f64595bc5ecf118d0f93853f1b1a274ba7f49a` |

The test is on synthetic NSF, not the full RVC/CrazyMita pipeline.

## Architecture implications

- Stage 48 established that the real synthetic NSF graph has repeated actual contiguous materializations during a single forward (40 requests on the default route, versus 13 with the Stage 49 prepack prototype in CPU experiments).
- Stage 49 confirms that reusing a physical weight layout **preserves output semantics on CUDA**. It may reduce per-forward data preparation and/or host dispatch cost; the present timing does not isolate those contributors.
- The experiment does not establish a universal performance improvement on CUDA or a replacement for general Conv1d/2d/3d algorithms. The current CPU/GPU conclusions should not be conflated.

## Next controlled R&D

1. **Interleave in one process:** construct both baseline and prepacked decoders once, using the same input tensors and checkpoint, then alternate `ABBA` or seeded randomized order per iteration with a synchronization/readback after each forward. Report individual pairs and paired ratios, not just separate-process medians. Hold both sets of weights resident, record actual startup/pack time separately.
2. **Separate latency contributors:** use host timings for graph construction/dispatch, CUDA events for GPU execution, and explicit transfer measurements. Verify the number of physical materialization operations in the CUDA path, not just CPU instrumentation.
3. **Shared algorithm change:** implement device-resident layout preprocessing once per model load, eliminate CPU readback/re-upload from the weight preparation path, and keep a chosen physical layout across Conv → activation → Conv when compatible. Validate memory lifetime, buffer aliasing and numerical parity.
4. Test larger channel shapes and real checkpoint weights. Compare CPU, WGPU and CUDA on identical operators and distinguish initial setup cost from amortized inference cost.
5. Only once common dataflow/layout decisions are stable should optional CUDA Graph capture/replay be evaluated as a separate launch-overhead optimization.

**Decision:** keep `RVC_CONV_WEIGHT_PREPACK` experimental and disabled by default. No measured basis for enabling it in production across all sizes.
