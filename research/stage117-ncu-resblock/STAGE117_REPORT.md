# Stage117 — Real RTX 4060 Nsight Systems RVC ResBlock analysis

Date: 2026-11-16. Source: `20261116T195206Z.zip` and `nsys-csv(1).zip`, user-supplied; 3 Rust/CUDA generator runs per mode. Only local Git bundle was used to inspect source. This report does not claim new GPU experiments or CUDA Nsight Compute counters.

## Benchmarked user data

| Metric | Baseline | Transport+shared NTC | Ratio / change |
| --- | ---: | ---: | ---: |
| Warm resident full-inference median (JSON) | 1506.65105 ms | 609.59315 ms | 2.4715x faster |
| Summed GPU kernel durations, three Nsight executions | 3102.905563 ms | 1110.130028 ms | 2.7951x lower |
| GPU kernel launches | 57638 | 56642 | -996 |
| `direct_conv2d_kernel...92d1fb6c` | 2263.274931 ms; 276 | 79.914813 ms; 60 | -216 launches |
| `preact_residual_temporal_kernel...f2fe9a03` | 0 | 185.966006 ms; 108 | new |
| `preact_residual_temporal_kernel...c7360f08` | 0 | 178.303285 ms; 108 | new |
| Summed CUDA memcpy H2D durations | 189.857436 ms; 1397 | 184.827280 ms; 1397 | -2.65% time |
| Summed CUDA memcpy D2H durations | 0.324763 ms; 9 | 0.325212 ms; 9 | essentially same |

All aggregated GPU kernel durations include three executions; they are NOT one warm-pass time and they exclude host/CPU work. Nsight NVTX summary tracks host NVTX ranges and must not be equated with pure GPU duration. Cold-start / recompilation outliers are visible in range totals.

The difference of 216 old `direct_conv2d...92d1` launches matches exactly 108+108 new `preact_residual_temporal` launches. Assuming the remaining 60 old direct-conv launches have a comparable cost in both modes, the associated computation group fell from about 2183.36 ms to 364.27 ms over three executions, about **5.99x**. This is an attribution estimate, not an instrumented ResBlock-only measurement. That replacement accounts for ~91.3% of the total observed 1992.78 ms reduction in aggregate GPU kernel duration; other kernels changed too.

## NCU status — the current archive does not yet profile the intended ResBlock CUDA kernel

- Baseline C128/K11/D5-first: `No kernels were profiled`.
- Candidate C128/K11/D1-second: `No kernels were profiled`.
- The two other NCU reports profiled mainly scalar, mask, copy and unrelated conv operations; no `preact_residual_temporal_kernel` targeted.
- Why: Rust NVTX push/pop surrounds high-level Burn tensor operators. Kernel launches are not reliably executed within those host-side push/pop intervals because the backend defers/schedules CUDA work. The CSV reports contain very short and/or outlier NVTX durations. This cannot be used to infer ResBlock L1/TEX, register pressure, Tensor Core utilization or occupancy.

## Stage117 measurement method (no new Rust build)

`Run-Stage117-ResBlock-NCU.ps1` selects `preact_residual_temporal_kernel` by actual CUDA kernel name, then skips earlier matching launches based on verified model graph order: four stages C256, C128, C64, C32; each stage 3 kernel branches (K3, K7, K11); three dilation pairs (D1, D3, D5); first and second kernel per pair. One pass = 72 matching kernels. For C128/K11/D5, zero-based first matching launch index = 34, second = 35. The Nsight Compute documentation explicitly states that `--launch-skip` counts only kernel-name matches. The profiler selects two launches and records their actual names/grids in the `.ncu-rep`.

Selection by execution order is a **hypothesis to validate**, because backend scheduling and graph semantics may change launch order. Verify the GPU grid/shape and kernel sequence against Nsight Systems before labeling any resulting profile C128/K11/D5. No implicit NVTX filtering is used.

The script only reads existing models and WAV, invokes the existing Rust executable, stores Nsight reports/JSON/logs, and removes temporary output WAV; it does not recompile code or install tools, and it does not archive WAVs/models.

### Required acceptance gate before promotion

The Stage116 profiling run aborted during PCM compare (relative path resolved under `C:\Windows\System32`), and no `manifest.json`/PCM comparison was saved; output WAVs were cleaned up. Matching output sample count 119200 proves only duration. **Full PCM16 parity and model correctness remain unverified.** Run an uninstrumented four-mode measurement (`baseline,baseline_ntc,transport,transport_ntc`) with the absolute-path-fixed Stage116 runner and inspect max PCM error/RMSE/SNR before enabling transport by default.

### Next actual algorithmic bottleneck

The new first and second residual kernels each consume about half the new ResBlock kernel time (185.97 and 178.30 ms across three executions). Hardware profiler evidence (L1/TEX wavefronts, stalls, FFMA/TC, registers, occupancy) is required before selecting larger output-channel reuse, implicit convolution/GEMM, or persistent layout. Additional NTC copies alone cannot explain the ~2s CUDA kernel gain; do not resume Stage111 deconv GEMM or unmeasured spectral convolution.