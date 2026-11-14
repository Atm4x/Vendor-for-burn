# Stage 50 — End-to-end NSF Conv1d integration: CPU A/B and CUDA test package

Date: 2026-11-10. Repository: `Atm4x/Vendor-for-burn`; experimental PR #2. Toolchain: Rust 1.98.1, CubeCL 0.11.3, Burn 0.22.1, cubek-convolution 0.3.1.

## Objective and scope

Stage 39's Packed/Quad/QuadInterior tests measured standalone Conv1d. Stages 44–49 measured scheduling, model-load-time weight preparation and fragments of the real decoder. **This stage actually routes NSF's standard Burn `module::conv1d` calls to Packed/Quad/QuadInterior on the full synthetic-checkpoint decoder.** It compares the result with the same unmodified NSF graph, rather than multiplying independent benchmark speedups.

The experiment is **opt-in** and not a production default. It does not change transposed convolutions, grouped convolutions, the encoder, HuBERT/RMVPE, RVC waveform quality, or inference outside the NSF Conv1d set. It is not a complete RVC TTS/voice-conversion benchmark.

## Integration design

1. `Conv1d::load` chooses one of two optional representations. Existing `RVC_CONV_WEIGHT_PREPACK=1` preserves Stage 49's `[Cout,K,Cin]` physically contiguous weight. New `RVC_NSF_CONV_KERNEL=packed|quad|quad_interior` prepares physical `[K,Cin,Cout]` **once** and stores a strided logical `[Cout,Cin,K]` view. The original logical weight shape and public `module::conv1d` API remain unchanged.
2. In `burn-cubecl` `kernel::conv::conv_direct`, the runtime recognizes this specific physical stride layout *only* for Conv1d with `groups=1` and an explicit kernel-mode flag. It reinterprets the same device buffer as `[1,K,Cin,Cout]` with metadata, without an additional forward-time weight copy.
3. It dispatches to the existing shared `cubek-convolution` `launch_prepacked_temporal_pair`, `launch_prepacked_temporal_quad`, or `launch_prepacked_temporal_quad_interior`. Other calls still use the original Direct path. This is not a CUDA-only rewrite: CPU/WGPU/CUDA use the same CubeCL kernel implementation.
4. `ResidualExecution::Sequential` and the CPU-specific `CUBECL_CPU_MAX_TASKS=32` control different levels of scheduling, and remain independent; GPU does not have an equivalent `max_tasks` setting.

**No permanent cache by mutable device-buffer identity.** The packed representation is model-owned and created at checkpoint loading. Runtime correctness for mixed layouts is guarded by a strict stride match. There is still an initialization-time GPU→CPU→device roundtrip, as in Stage 49; eliminating it requires a separate model-owned CubeCL pack path.

## Local CPU matrix — full NSF forward + blocking readback

Hardware: virtualized AMD EPYC 9V74, five visible logical CPUs, affinity 0–4. Toolchain: Rust 1.98.1 + LLVM 23.1.0. Each measured invocation includes the entire NSF forward and CPU readback; 5 samples after 4 warmups. Two independent processes per configuration, with experiment order reversed for the second pass. Same public synthetic NSF checkpoint and inputs.

Times below are **medians across the two per-process medians**, in milliseconds (not kernel-only time):

| Configuration | 4 frames | 64 frames | 256 frames | Ratio to baseline at 64 frames |
|---|---:|---:|---:|---:|
| Original: Parallel, CPU batch 8, Direct | 114.030 | 111.903 | 113.986 | 1.00× |
| Sequential, CPU batch 8, Direct | 87.433 | 116.049 | 96.316 | 0.96× |
| Sequential, CPU batch 32, Direct | 24.095 | 24.828 | 21.263 | 4.51× |
| Sequential, CPU batch 32, Stage49 Prepacked | 16.146 | 16.109 | 12.363 | 6.95× |
| Sequential, CPU batch 32, Packed Conv1d | **11.997** | **11.957** | 12.070 | **9.36×** |
| Sequential, CPU batch 32, TemporalQuad | 12.080 | 13.202 | **11.853** | 8.48× |
| Sequential, CPU batch 32, QuadInterior | **12.000** | **11.960** | 12.639 | **9.36×** |

First and second process values at 64 frames: original 99.862 / 123.943 ms; Sequential batch-32 33.587 / 16.069 ms; Stage49 20.087 / 12.130 ms; Packed 11.932 / 11.982 ms; Quad 14.315 / 12.089 ms; QuadInterior 11.993 / 11.927 ms.

**Interpretation:** the integrated CPU acceleration is real in the tested fixture, but the baseline and intermediate modes are noisy due to virtualization, CPU contention and scheduler behavior. The relative end-to-end Packed-versus-baseline speedup in the two ordered passes is ~8.4× and ~10.3×. These are descriptive observations, not confidence intervals. QuadInterior does **not** reliably outperform Packed in this full-graph workload despite its earlier isolated Conv1d benchmark lead. Do not multiply isolated Stage 44 and Stage 49 speedups.

## Numeric correctness

For each of the 14 configurations/runs and all 3 lengths, exported the **full little-endian FP32 output tensor** and compared element by element against the first Parallel/Direct baseline. All compared shapes and output lengths match, all values are finite, and **all 42 comparisons pass** with `max_abs_error ≤ 3.3527613e-8`. The Direct and Stage49 Prepacked paths are bitwise identical to baseline; Packed/Quad/QuadInterior differ only at FP32 rounding precision.

## What is NOT established

- **No RTX 4060 integrated A/B yet**: standalone kernels were measured in earlier stages, but the full six-mode GPU matrix needs the Windows artifact and user's execution.
- WGPU runtime correctness and performance have not yet been measured on GPU. Local CPU `cargo check` passed; CUDA and WGPU builds require CI checks on this branch.
- CPU `batch=32` is a CPU `SchedulerMultiStream` policy. It is not a setting for CUDA's `CudaServer`/`MultiStream` and must not be represented as portable.
- Stage 41 fused residual aggregation and Stage 47 persistent activation layouts are **not included** in this integration. Their actual incremental effect on real NSF is unproven; combining them without a properly grounded baseline risks false conclusions.
- One-time prepack model-load cost was not included in the CPU forward timings; Stage 49 measured it separately. Future benchmark should measure break-even time for cold and repeated inference.
- This synthetic checkpoint is not the weights of CrazyMita, and the tested lengths 4/64/256 are not the complete RVC inference pathway.

## How to run GPU A/B

Download the Windows artifact from the Stage50 workflow and run in its folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-NSF-Integrated.ps1 -Iterations 20 -Warmup 5 -Frames "64,256,1024"
```

The script executes two order-balanced rounds of Baseline Parallel, Sequential Direct, Stage49 Prepacked, Packed, Quad and QuadInterior. Each process loads the same synthetic checkpoint, warms kernels, measures host submit and blocking readback, emits full FP32 output buffers, and cross-compares them before writing `NSF_INTEGRATED_STAGE50_RESULTS.zip`. Numerical validation is required before declaring any speedup.

## Next research after Windows GPU results

Select a backend-specific **choice of kernel variant**, not a backend-specific implementation: one shared set of CubeCL operators with measured dispatch parameters. If CPU favors Packed and CUDA favors QuadInterior, the layer dispatch policy may differ by geometry and backend. Next evaluate a device-native model-owned weight pack and preserve NLC layouts across adjacent operators; only after a stable end-to-end baseline revisit CUDA Graph replay to reduce launch overhead.

## Control A/B — one pinned logical CPU

To separate CPU scheduler oversubscription from actual compute/layout savings, ran two additional processes per option on one pinned CPU (0), 6 iterations after 4 warmups. CPU batch-8 Parallel baseline vs CPU batch-32 Sequential (Direct, Packed, QuadInterior), in forward and reverse order.

| Mode | 4 frames | 64 frames | 256 frames |
|---|---:|---:|---:|
| Parallel / batch 8 / Direct | 45.960 ms | 46.248 ms | 47.709 ms |
| Sequential / batch 32 / Direct | 36.014 ms | 36.222 ms | 38.174 ms |
| Sequential / batch 32 / Packed | **28.009 ms** | 28.042 ms | **28.303 ms** |
| Sequential / batch 32 / QuadInterior | 28.012 ms | **28.007 ms** | 30.168 ms |

Medians of two independent process medians. At 64 frames this is ~1.65× end-to-end, much less than on five contended CPU workers but still a material benefit. All **24** complete one-core FP32 output comparisons passed with `max_abs_error ≤ 3.3527613e-8`. Raw per-process timings: `evidence/stage50-cpu/onecore_nsf_matrix.json`.
