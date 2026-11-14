# Stage 47 — Persistent-layout convolution pipeline (cross-backend R&D)

Date: 2026-11-10. Experimental work in `Atm4x/Vendor-for-burn`, PR #2. **Not activated in production NSF.**

## Motivation and relation to Stage 46

Stage 46 established an important negative result: simply removing the mandatory `into_contiguous` calls and accessing channel-strided tensors directly is much slower on three large NSF Conv1d geometries. CPU measurements confirmed penalties of roughly 4–9×. The bottleneck is the *repeated transition between physical layouts* and the consequent loss of vectorization/locality, not that tensor copies are always expensive.

Rather than insert CUDA-only graphs or refine a single convolution tile, this experiment preserves a **channels-last physical layout across a multi-operator block**. Weight rearrangement is performed once, outside the repeated forward measurement. This is a model/dataflow change intended to be expressible through common CubeCL operations on CPU, WGPU, and CUDA.

## Two-convolution experiment

The exact mathematical graph is:

`Input → Conv1d(K,D) → LeakyReLU(0.1) → Conv1d(K,D) → Output`.

The two variants use the **same** CubeCL `launch_direct::<1>` and activation kernel with the same deterministic synthetic inputs/weights.

**Baseline (modeled repeated conversion):**

1. Start with physical NCL input and physical NCK weights.
2. Materialize input NLC and weights NKC before the first conv.
3. Perform the first convolution.
4. Materialize the output NCL to model a channel-first intermediate operator boundary.
5. Apply LeakyReLU in physical NCL.
6. Materialize the activation NLC before the second conv.
7. Materialize weights NKC again and run the second conv.

**Candidate (persistent physical NLC):**

1. Pack the static NCK weights into NKC **once before timing**.
2. Convert the external NCL input to NLC once per forward (the same external-input cost as baseline).
3. Run Conv1d → LeakyReLU → Conv1d entirely with physical NLC intermediates and packed weights.
4. Do not physically transpose between the two convolutions.

The baseline's artificial NLC→NCL→NLC materialization is an explicit **counterfactual**, not a confirmed trace of the real Burn NSF graph. Burn can represent some transpose operations as metadata-only views. This A/B gives a cost/benefit indication for a possible layout-resident execution contract, **not** a verified production RVC speedup.

## Measurement protocol

- CubeCL CPU JIT, Rust 1.98.1, LLVM 23.1.0-4; one pinned logical CPU on the available virtualized AMD EPYC host.
- Full forward launch and blocking CPU readback each iteration; 12 measured iterations after four warmups.
- Three independent sequential trials A/B/C without concurrent compiler jobs; reported values are medians of per-run medians.
- Deterministic synthetic tensor values; `Cin=Cout` in each two-conv sequence; full output FP32 parity checked before every timing run.
- Both versions use the same scalar operations; new version changes only layout residency and weight preparation lifetime.

| Two-Conv geometry | Baseline, ms | Resident layout, ms | Baseline/Resident |
|---|---:|---:|---:|
| Small K3, C8, T64 | 0.076 | 0.054 | **1.41×** |
| NSF K3, C256, T240 | 9.094 | 4.803 | **1.89×** |
| NSF K11/D5, C128, T2400 | 46.508 | 41.482 | **1.12×** |
| NSF K11, C32, T9600 | 13.193 | 11.872 | **1.11×** |

All four shapes: `max_abs_error = 0` for baseline vs candidate, comparing complete output tensors. Each of the three controlled trials gave zero error.

Raw data: `research/stage37-windows-gpu-bench/evidence/stage47-cpu/pipeline_{a,b,c}.json`.

## Interpretation

A persistent physical layout between operators is especially useful when output channels are wide or kernels are small (e.g. NSF K3). For long K11 convolutions, arithmetic dominates and the saved copies have a smaller effect. The relative benefit is **shape-dependent**, and could differ substantially on GPU. CUDA Graphs would not themselves remove the unnecessary conversion or static weight packing; replay can be measured afterwards as a separate optimization.

The logical data-format API (NCL/NCHW) need not force the same physical storage. The desired abstraction is a backend-independent `LayoutPlan` with explicit logical shape, physical strides, preferred layout, lifespan and producer/consumer constraints. Its operator lowering should preserve a compatible layout, convert at unavoidable boundaries, and cache immutable weights across inference calls.

### Recommended structural direction

- A shared operator/layout planning phase above backend-specific scheduling, using the actual graph: ConvNd, ConvTransposeNd, LeakyReLU, residual adds, and output projection.
- Separate model initialization cost from warm inference; static filters are repacked once and retained as device buffers.
- Propagate preferred physical layout through pointwise and residual operations; avoid repeatedly materializing the same transposed tensor.
- Consider explicit fused Conv/bias/activation/residual epilogues where numerical semantics can be preserved and measured.
- Dispatch among `Direct`, `DirectStrided` for lightweight shapes, and `ImplicitGemm` for larger shapes using shape- and device-informed criteria, rather than blindly applying any one optimization to all backends.
- Tune CPU/WGPU/CUDA launch configurations separately **after** choosing the same model-level mathematical/dataflow strategy.

## Validation and outstanding limits

- Local Rust release compile and actual CPU runtime execution: PASS.
- Complete FP32 parity across the two micrograph implementations: PASS.
- Three controlled CPU A/B trials: PASS.
- CPU, WGPU, CUDA feature checks for the underlying Stage46 shared ConvNd kernel: PASS.
- Full Burn NSF graph materialization count: **not measured yet**.
- CUDA/WGPU full pipeline performance: **not measured yet**.
- Complete production RVC pipeline and real weights: **not tested**.

The next experiment should instrument `burn-cubecl::into_contiguous_aligned` and track real physical buffer allocations/strides across the NSF decoder, then implement persistent layout at the actual appropriate operator boundary. Do **not** assume Burn makes every physical conversion shown in the synthetic baseline.
