# Stage 48 — Actual Burn/CubeCL NSF physical-layout materializations

Date: 2026-11-10. Repo: `Atm4x/Vendor-for-burn`. Branch: `experiment/conv-kernel-rnd-stage33-20261110` (PR #2). Research-only instrumentation; production NSF execution strategy unchanged.

## Purpose

Stage 47 tested an artificial pair of Conv1d operators with/without forced NLC→NCL→NLC materialization. The measured `1.11–1.89×` improvements were **not** established for real Burn NSF. Stage 48 measures the *actual* physical conversions taken in a full NSF forward, independent of the synthetic pair's assumptions.

## Instrumentation

The shared Burn/CubeCL `kernel::into_contiguous_aligned` and `kernel::into_contiguous` functions emit a single diagnostic event, gated on `CUBECL_CONV_LAYOUT_TRACE`. The event contains the logical tensor shape, physical strides and whether the function will materialize another buffer. When the environment variable is absent, original conversion behavior is retained and no diagnostic line is emitted. This experiment does not alter numerical computation, strides, tensors or execution order.

A dedicated CPU fixture runner (`trace_nsf_layout_stage48.rs`, `--no-default-features --features cpu`) loads `synthetic_nsf_decoder.pth`, constructs deterministic 4/64/256-frame inputs, executes each real NSF forward *once* in sequential residual mode, and reads back the full output. The wrapper's `CUBECL_NSF_BEGIN/END` markers identify the forward interval. The analyzer converts the trace into `trace_summary.json`.

## Observed results

Host: virtualized AMD EPYC, Rust 1.98.1, CubeCL CPU JIT, LLVM 23.1.0-4. CPU scheduler batch limit 32. **This is a conversion count, not a runtime speed benchmark.**

| NSF input frames | `into_contiguous_aligned` materializations | `into_contiguous` materializations | Sum of logical f32 elements processed by aligned materializations | Approx. logical FP32 volume |
|---:|---:|---:|---:|---:|
| 4 | **40** | 0 | 1,902 | 0.0073 MiB |
| 64 | **40** | 0 | 6,222 | 0.0237 MiB |
| 256 | **40** | 0 | 20,046 | 0.0765 MiB |

These logical byte volumes are *not actual transfer traffic* or pitched buffer sizes. The same number of materialization calls occurred across the three shapes, although the moved tensor sizes increased with length. The trace shows four groups of six repeated shapes: `[4,3,4]`, `[4,7,4]`, `[2,3,2]`, `[2,7,2]`. These are compatible with Conv1d static filter shapes, but trace lines do not carry stable tensor identity, so six occurrences do **not** prove six copies of *the same* filter within one forward. They demonstrate repeated operator-time layout preparation across the ResBlock chain. Other materializations involve upsampling/ConvTranspose and input/intermediate tensors.

Output checksums are unchanged relative to the earlier full NSF fixture:

| Frames | Elements | Double-precision sum of FP32 output |
|---:|---:|---:|
| 4 | 16 | 0.186992665986 |
| 64 | 256 | 3.379754826427 |
| 256 | 1024 | 13.588609755039 |

These sums check gross consistency only; they do not independently prove bitwise equality to an untouched executable. The Stage 47 full-tensor parity is a **different** experiment.

## Why this matters

The current shared Burn/CubeCL forward route does not universally preserve a favorable channels-last physical layout between all operators. In addition, some static filter layouts are prepared during forward rather than cached at model initialization. The small synthetic checkpoint moves only tens of KiB, so the dominant potential issue may be **extra kernel submissions, allocations and scheduler overhead** rather than memory bandwidth. Larger real RVC models may behave differently.

The Stage 46 result remains valid: merely skipping all copies and directly reading large channel-strided Conv1d inputs can be **4–9× slower**. This experiment does not justify forcing DirectStrided globally.

## Validation

- CPU feature `cargo check --offline --locked`: PASS after instrumentation.
- Rust 1.98.1 release compilation of the runner: PASS.
- Runtime CPU JIT three-shape forward: PASS.
- Trace parser rechecked all three runs as 40 aligned materializations.
- GPU backend performance of this instrumentation: **not measured**.
- Real CrazyMita weights or full RVC inference: **not tested**.
- Production dispatch: **unchanged**.

## Reproduce

```bash
export CUBECL_CONV_LAYOUT_TRACE=1
export CUBECL_CPU_MAX_TASKS=32
cargo build --offline --locked --release -p tts-with-rvc-rs --no-default-features --features cpu --bin trace_nsf_layout_stage48
./target/release/trace_nsf_layout_stage48 4 64 256 > stage48_output.txt 2> stage48_forward.trace
python research/stage37-windows-gpu-bench/tools/analyze_stage48_trace.py stage48_forward.trace --output trace_summary.json
```

LLVM library / toolchain environment must be configured for CubeCL CPU JIT in the usual way. The release runner was built against Rust 1.98.1 and tracel LLVM 23.1.0-4.

## Next R&D

1. Add stable, opt-in IDs for model weights during initialization, distinguishing **each filter's first physical pack** from repeated packs across inference calls.
2. Prepack immutable ConvNd weights once during model initialization using backend-independent operations, while retaining the logical NCK/NCHW API. Keep packed buffers resident on the corresponding device. Never mix packing cost into steady-state forward timing.
3. A/B full NSF forward: baseline vs cached weights, with full FP32 parity, materialization-count reduction and actual latency on CPU. Repeat with a Windows CUDA fixture before promoting any behavior.
4. Only then investigate global physical-layout planning for ConvNd → activation → residual add and shared operator fusion. CPU batch tuning and CUDA graphs are orthogonal optimizations to evaluate afterwards.

Source and evidence: `models/tts-with-rvc-rs/src/bin/trace_nsf_layout_stage48.rs`, `vendor/burn-cubecl-0.22.1/src/kernel/contiguous.rs`, `evidence/stage48-cpu/trace_summary.json`, `stage48_forward.trace`, `stage48_output.txt`.
