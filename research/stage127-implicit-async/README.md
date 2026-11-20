# Stage127 — ImplicitGemm synchronous/asynchronous NCU/NSYS full-waveform RVC experiment

Stage127 uses the exact PR #2 head obtained through Git Bridge: `1e9f23495e97fbbc59fcdeac9c3fde4f637dcfa5`. It changes only the opt-in Cubek matmul loading routine for RVC Conv1d: `SimpleSyncCyclic/Cmma` versus `SimpleAsyncCyclic/Cmma`. It does **not** claim a CUDA speedup or PyTorch parity until real RTX 4060 runs.

## Modes

- `implicit_sync`: old fast global ImplicitGemm, persistent GRU, FP32, all NSF convolutions routed to GEMM.
- `implicit_async`: same route except Conv1d uses asynchronous cyclic CMMA tile loaders.
- `hybrid_sync`: NTC ResBlock fused direct route + shared NTC + activation placement + GEMM ConvTranspose1d, other Conv1d direct.
- `ntc_combined`: Stage125 combined NTC route with *direct* ConvTranspose1d.
- `implicit_fp16` (optional): Stage87/88 FP16 ResBlock precision island with ImplicitGemm; quality drift and CUDA JIT unverified.

All Rust cases use the same WAV, checkpoint, FAISS index, `core_frames=300`, batch=2, persistent GRU and synchronized resident inference. Alternating process order reverses each cycle. `implicit_sync` is the Rust reference; waveform PCM16 differences, SNR and maxima are reported without silently treating different-dtype output as bit-exact. 75 dB and 128 PCM LSB are initial safety thresholds for research, not a final perceptual-quality gate.

## Main Windows command

From the extracted CI artifact folder with your existing `embed/` assets:

```powershell
.\Run-Stage127-Full90-Profile.ps1 -Runs 3 -Cycles 2 -Profiler all
```

The default profiler captures full 90-second NVTX/CUDA timeline via Nsight Systems for `implicit_sync`, `implicit_async`, and `ntc_combined`, plus four selected real Conv/GEMM kernels per mode using Nsight Compute on the **real 3-second WAV segment**, not synthetic tensors. NCU's replay time is never mixed with warm latency. For NCU, the regex default is `conv` on GEMM and `preact_residual_temporal_kernel` on NTC: these are broad selectors, **not guaranteed** to capture the hottest geometric subgraph. Inspect captured kernel names and launch geometry before attributing metrics. For exact GPU bottlenecks, use NSYS's full CUDA-event timeline and its kernel summary to select a kernel name, then rerun with `-NcuKernelRegex` and `-NcuLaunchSkip`. NCU CSV contains the full counter set if the locally installed NCU Python SDK is found; `.ncu-rep` is always retained.

Optional `-TorchNsys` adds a full-waveform PyTorch FP16 NSYS report. The reference uses the installed original source tree, hashes five source files and records differences. It tolerates both old `Config.device_config()` and already-populated `x_pad/x_query/x_center/x_max` interfaces, without modifying installed Python source. Its numerical computation/chunk scheduling and timer boundary are **not** identical to the Rust engine. Do not infer equal-graph latency from the two totals.

For just preflight use `-CheckOnly`; for profiling an already known target kernel use `-Profiler ncu -ProfileModes 'implicit_sync' -NcuKernelRegex 'actual_kernel_name_fragment' -NcuLaunchSkip 0 -NcuLaunchCount 4`.

The report ZIP contains only JSON, CSV, `.nsys-rep`, NCU reports, SQLite event logs, text, SHA and metadata. It does not contain WAV, weights, indexes, EXE or DLL. Existing Stage125 and other legacy source/scripts remain unchanged. RTX 4060 hardware acceptance is required before promoting `RVC_NSF_CONV1D_GEMM_ASYNC=1` to any default.

## Main NCU metrics to inspect

`gpu__time_duration.sum`, `sm__throughput.avg.pct_of_peak_sustained_elapsed`, FMA/Tensor pipe utilization, `smsp__warps_eligible.avg.per_cycle_active`, `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`, local spills, L1/L2/DRAM sectors, shared-memory throughput, achieved active warps, registers/thread, block/grid and static/dynamic shared memory. Metric availability varies by NVIDIA hardware and NSight version. The NCU replay duration is not a free-running end-to-end timer. Detect 0 selected kernels as an incomplete measurement.
