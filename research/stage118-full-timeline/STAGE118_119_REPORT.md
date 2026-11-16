# Stage118–119 — Real RTX4060 GPU timeline, ResBlock stage attribution and targeted temporal weight reuse

Date: 2026-11-16. Repository: `Atm4x/Vendor-for-burn` PR #2. Source: Git bundle (branch `experiment/conv-kernel-rnd-stage33-20261110`) plus published Stage115–116 changes. Rust toolchain: 1.100.0.

## Input evidence and provenance

- User uploaded the original exported NSYS SQLite files `baseline.nsys.sqlite` (106.7 MiB) and `transport_ntc.nsys.sqlite` (67.4 MiB) as `baseline.nsys.zip`.
- Prior Stage116 original profiler ZIP `20261116T195206Z.zip` has the full `.nsys-rep`, `.ncu-rep` and per-mode `.benchmark.json`.
- Latest direct user output: 5-run Stage116 PCM parity for 119,200 samples at 40,000 Hz. `baseline_ntc` bit-identical to baseline. `transport` and `transport_ntc` identical to each other; relative to baseline only 260 / 119200 samples differ, all by exactly 1 PCM16 LSB, normalized RMSE = 1.425318312986511e-6, SNR = 96.16997357591481 dB. This is **not** proof of numerical parity on arbitrary audio or of bit-identical FP32 tensors.
- The measured `20261116T195206Z` warm medians: baseline 1506.65 ms; transport+NTC 609.59 ms (2.47x). These two modes were run sequentially, so changes in GPU clocks, OS contention or input model warm-up remain possible confounders.

## Full execution timeline, not just sums

Reconstructed from **each individual** `CUPTI_ACTIVITY_KIND_KERNEL` record, CUDA runtime record, correlation ID, NVTX range and stream ID. All 57,638 baseline and 56,642 transport kernel rows are preserved in separate timestamped CSV exports alongside all CUDA API calls and GPU copies. Every GPU kernel in these traces ran on stream 14; there is no evidence of overlap between separate GPU streams for this capture. CPU NVTX ranges mark host submission, **not GPU completion**.

| Execution | Baseline elapsed ms | Transport elapsed ms | Baseline GPU kernel ms | Transport GPU kernel ms |
|:--|--:|--:|--:|--:|
| First (includes JIT/cold) | 6423.3 | 3443.5 | 1053.8 | 374.2 |
| Warm #1 | 1509.8 | 665.8 | 1028.6 | 370.2 |
| Warm #2 | 1502.5 | 552.7 | 1020.5 | 365.7 |

Elapsed intervals are reconstructed from `stage102_rust_hubert` start to corresponding `stage104_rust_nsf_readback` end. They agree with per-mode `inference_ms` in the benchmark JSON within expected boundaries. The difference between host elapsed and GPU kernel sums contains CUDA API overhead, host waits, memory copies and idle gaps; it is **not** exclusively CPU computation. `cuLaunchKernel` recorded 57,638 vs 56,642 calls. Approximately 48,000 launches per 3 runs were submitted while the host NVTX range `stage104_rmvpe_gru` was active. That is consistent with the opt-out stepwise GRU path.

### ResBlock attribution: actual GPU start/end times

For baseline the relevant `direct_conv2d_kernel_e_f32_nin_4_nout_4_92d1fb6c` launches are identified by gridX 745 (stage0) and gridX 3725 (stages1–3). For transport they are `preact_residual_temporal_kernel` with gridX 373 and 1863, respectively. In each mode there are precisely 216 matching kernels, grouped into three identical 72-kernel stage sequences in launch order. Each stage has 18 kernels: K3/K7/K11 branches, three residual pairs each, first and second convolutions. The grouping was checked against every grid size and the sequential `ResBlock1::forward_with_scheduling` implementation, not guessed from asynchronous host NVTX durations.

| ResBlock stage | Baseline GPU ms / 3 runs | Transport GPU ms / 3 runs | kernel speedup |
|:--|--:|--:|--:|
| stage0, C256 | 568.607 | 71.756 | **7.92x** |
| stage1, C128 | 935.938 | 160.382 | **5.84x** |
| stage2, C64 | 456.742 | 82.581 | **5.53x** |
| stage3, C32 | 223.048 | 49.550 | **4.50x** |
| **Total** | **2184.335** | **364.269** | **6.00x** |

Stage1/C128/K11 alone: baseline 488.764 ms vs transport 83.774 ms over three runs (5.83x). K11 makes up ~52.2% of the accelerated stage1 kernel time. At least on this input, the largest remaining *single ResBlock geometry* is C128/K11.

Do not attribute the entire 2.47x warm full-RVC speedup to ResBlock: the **unchanged** RMVPE/HubERT host-stage timings also differ between separately captured runs. A controlled alternating A/B (and ideally multiple WAVs) is needed to isolate causality.

### Actual critical-path implications

The first call of either mode includes several seconds of CUDA kernel compilation and model preparation. In warm runs, baseline GPU kernel sums are ~1021–1029 ms and transport GPU kernel sums ~366–370 ms, but host elapsed is ~1503–1510 vs ~553–666 ms. ResBlock optimization clearly reduces GPU compute; remaining time is a combination of RMVPE, other convolution kernels, CUDA API submission and queue drain.

The same trace also exposes a second, **already solved but disabled here**, graph problem: stepwise RMVPE BiGRU triggers ~16k very short kernels per inference. The repository has `RVC_RMVPE_GRU_PERSISTENT=1`, which uses one CubeCL persistent recurrent kernel per batch and was previously benchmarked in Stage67/68. Hence the next proper full-graph A/B should hold persistent GRU fixed while comparing ResBlock strategies, rather than waste latency repeatedly submitting the legacy GRU step kernels.

## Stage119 experimental computation: C128/K11 temporal tile4

The existing Stage115–116 `preact_residual_temporal_kernel` computes **two adjacent output positions per output-channel vector** and reuses one weight vector between the two positions. New Stage119 mode computes **four adjacent output positions** in the same unit, reusing the weight vector up to four times. Output channels stay contiguous (NTC), fused LeakyReLU and residual-add remain unchanged, and the original K→Cin summation order is preserved per output. No im2col buffers, new synchronizations, approximate arithmetic, Tensor Core requirements or new CUDA-specific backend code.

- Opt-in: `RVC_NSF_RESBLOCK_TEMPORAL_TILE=4` while `RVC_NSF_RESBLOCK_TRANSPORT=1`.
- Applies **only to C128/K11**; other ResBlock shapes continue using tile2. This targets the measured dominant remaining geometry rather than changing every stage speculatively.
- Potential upside: reduce repeated weight-vector loads approximately 2x for interior positions (relative to tile2), fewer work units / scheduling overhead.
- Risks: more concurrent accumulators, higher registers/thread, occupancy loss, longer dependency chains, potentially worse CUDA speed. These are hypotheses, not performance results.
- Existing tile2 is the default. A baseline per-element CPU reference compared tile2 and tile4 **bit-identically** for C1/C3/C32/C128, K3/K7/K11, D1/D3/D5, odd/short lengths and residual on/off: 2/2 Rust tests PASS.
- Rust 1.100.0: CUDA-feature `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cuda --lib --bin tts-with-rvc-rs` **PASS** (CubeCL code type checks; GPU JIT is not tested on Linux).
- Full PCM parity and CUDA execution of tile4 **not measured**. Do not promote tile4 to production until the Windows RTX4060 test establishes both.

## Stage118 and Stage119 Windows A/B runners

`Run-Stage118-Controlled-ABBA.ps1` holds NSF transport and GRU execution in a 2x2 factorial (stepwise/persistent × baseline/transport_ntc) with alternating order across cycles. It tests whether the currently reported total speedup remains after avoiding the 16k-stepwise-GRU launch storm and tests PCM parity across both dimensions. Results ZIP includes benchmark JSON, CRC/hash summaries and logs only, never WAV or private models.

`Run-Stage119-Temporal-Reuse.ps1` uses the corrected Stage116 companion runner, alternates tile2/tile4 across cycles, compares baseline/transport PCM within each sub-run, reports warm medians and generates a single `STAGE119_RESULTS.zip` (nested runs ZIPs, no WAV or models). No NCU overhead contaminates warm timings. Run it after Windows CI produces the compiled Stage119 candidate.

Primary follow-up: benchmark stage1 C128 K11 with NCU counters for tile2/tile4 (sm efficiency, L1/TEX, registers, long scoreboard) and use NSYS for the true event sequence. If the tile4 path loses, remove it or leave it disabled, and focus on the already available implicit-convolution matrix path under matching FP32 parity.

## Reproduction

From these two SQLite files:

```
python export_full_timeline.py --baseline baseline.nsys.sqlite --transport transport_ntc.nsys.sqlite --out exports
```

Outputs: `analysis.json`, both full CUDA/NVTX/API/memcpy event timelines with original timestamps and correlation IDs, and both per-launch indexed ResBlock summaries. These outputs contain no model weights, personal WAV, or checkpoint files.