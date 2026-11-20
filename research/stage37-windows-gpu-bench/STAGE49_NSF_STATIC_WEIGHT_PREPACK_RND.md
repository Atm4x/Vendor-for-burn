# Stage 49 — Cached physical Conv1d weight layout in real NSF

Date: 2026-11-10. Repo: `Atm4x/Vendor-for-burn`, experimental PR #2. **Opt-in research prototype; disabled by default and not a production dispatch policy.**

## Hypothesis and preceding evidence

Stage 47 established that keeping a channels-last physical layout across a synthetic two-Conv1d pipeline can avoid redundant copies. Stage 48 instrumented the *real* synthetic-checkpoint NSF path and found **40 aligned materializations per forward**, including repeated filter-layout materializations in its ResBlock graph.

The fundamental question here is whether immutable Conv1d weights can be converted once when loading the model, while retaining their logical NCK shape for the Burn API and avoiding the repeated reformatting at every `conv1d` forward. This changes neither the convolution algorithm nor CPU/CUDA scheduling and requires no CUDA graphs.

## Implementation

An opt-in environment flag `RVC_CONV_WEIGHT_PREPACK=1` in `Conv1d::load`:

1. Obtains the computed static FP32 weight tensor with logical shape `[Cout, Cin, K]`, after loading/weight normalization.
2. Reorders its elements **once** into a dense `[Cout, K, Cin]` channels-last device allocation; then uses `swap_dims(1, 2)` to restore logical NCK dimensions while preserving physical NKC layout (last channel stride 1 after Burn's NCK→NKC view permutation).
3. Caches the resulting logical NCK tensor in the existing `Conv1d` field; all subsequent `forward` calls reuse it.

On the research prototype, step 2 uses a one-time CPU readback/reupload to construct this backend-independent representation through public Burn APIs. **This is acceptable only as an R&D prototype**: the real architecture should prepack on the device through shared CubeCL and retain buffers for the lifetime of the model. Model initialization latency and CPU/GPU transfer costs were not measured here. `TransposedConv1d` remains unchanged. If the environment flag is absent, the original path remains intact.

## Methodology

- Full NSF decoder from public synthetic checkpoint, actual Burn/CubeCL CPU JIT, Rust 1.98.1 / tracel LLVM 23.1.0-4.
- Virtualized AMD EPYC with five CPU affinity (0–4); `CUBECL_CPU_MAX_TASKS=32`, `ResidualExecution::Sequential` **in both variants** to isolate weight-layout lifetime.
- Three independent alternating Baseline/Prepacked trials. Each trial executes 12 measured forwards after 4 warmup forwards for 4/64/256 input frames, and includes blocking CPU output readback. The weight prepack is done at model load, outside the timed inference segment.
- All complete FP32 outputs are exported after timing and compared **byte-for-byte** per input and per trial.
- A separate environment-gated CubeCL contiguous trace counts actual materializations for a single 64-frame forward in both modes.

## Controlled CPU results

Median of three independent run medians (ms):

| NSF frames | Baseline | Prepacked static Conv1d weights | Speedup |
|---:|---:|---:|---:|
| 4 | 16.1595 | 12.0255 | **1.344×** |
| 64 | 16.0190 | 12.0524 | **1.329×** |
| 256 | 16.2685 | 12.0706 | **1.348×** |

All nine Baseline/Prepacked full-FP32 tensor comparisons are byte-identical (`cmp PASS`). SHA-256 output hashes (identical in both variants):

- 4 frames: `62484aa879281b76da3b31ec5f409200b57c9e629c0f0ab58f47bd22e0061ea8`
- 64 frames: `892aec83a6b4b7a0935487f3e183db7d47dada12a3277405d658d12c7b85358d`
- 256 frames: `a9ae3996a9659ccd5b7bc99b6865fa37adb54dd6d12cb1476d5104d72db85f19`

### Actual Burn/CubeCL materializations, 64-frame full forward

| Route | Calls that materialized a new layout | Logical FP32 data copied (estimated) |
|---|---:|---:|
| Baseline, static weights in NCK | 40 | 0.0237 MiB |
| **Prepacked once at load** | **13** | **0.0182 MiB** |

This is **27 fewer kernel/layout materializations per forward (67.5% reduction)**. Most removed operations are filter reorders, with smaller total bytes than input/activation materializations. Therefore the observed CPU speedup is plausibly due to lower dispatch/allocation overhead, not primarily raw memory bandwidth. This is a hypothesis about the mechanism; micro-profiling would be needed to attribute every millisecond.

## Validation and limitations

- CPU JIT `cargo check`: PASS.
- CPU release binary: PASS.
- 3× CPU A/B and 9/9 complete FP32 parity: PASS.
- Stage48 opt-in tracer: 40 → 13 materializations verified; the tracer is not a kernel-only timing instrument.
- GPU feature `cargo check`: **PASS** for both `cuda` and `wgpu`, Rust 1.98.1, with the opt-in prepack branch present. This establishes build compatibility only, not measured GPU speedup.
- CUDA/WGPU actual performance: **not measured**.
- Full production RVC checkpoint and audio: **not measured**.
- Model load/prepack cost: **not measured**; a one-time host roundtrip could be undesirable on CUDA.

This CPU result belongs to a **real NSF decoder graph with synthetic model weights**, not the full production RVC pipeline. The CPU ratio is not directly transferable to CUDA. Earlier Stage44 CPU scheduler tuning (6–8× in the synthetic graph) is a separate factor, not something to multiply automatically with this improvement.

## Next architectural step

1. Replace the one-time host readback/reupload with a shared CubeCL device-side prepack operation (immutable filter cache) used by CPU, WGPU and CUDA. Define model init ownership and cache invalidation as explicit invariants.
2. Benchmark warm inference and model initialization separately, including prepack amortization break-even for short and long inference sessions.
3. Test checkpoint variants, Conv1d K1/K3/K7/K11, ConvTranspose1d separately, and full RVC output parity against PyTorch.
4. Extend graph-level physical layout propagation through activation, residual additions and ConvTranspose, avoiding forced contiguous copies only when preserving coalesced channel access.
5. After stabilizing the shared algorithm/dataflow, compare CPU/WGPU/CUDA device tuning and optional CUDA Graph replay as orthogonal secondary optimizations.

Evidence: `stage49_cpu_results.json`, `trace_baseline.json`, `trace_prepacked.json`, and complete stdout/trace files under `evidence/stage49-cpu`. The experimental behavior remains opt-in; no production default changes.

## Validation update (2026-11-10)

The GPU feature gates finished with exit code zero for both CUDA and WGPU. Windows CUDA A/B is not yet run. To isolate warm inference from one-time weight preparation, the next Windows package must measure model-load times separately, export complete output buffers, and compare baseline against `RVC_CONV_WEIGHT_PREPACK=1` at unchanged residual execution settings.
