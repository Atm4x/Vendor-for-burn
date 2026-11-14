# Stage 93 — Global compute-cost-aware RVC chunk scheduling

Date: 2026-11-14. Parent: `82c1dc34` (Stage 92). Toolchain: Rust 1.100.0, offline.

## Motivation and architectural hypothesis

Stage 90's quiet-splice reduced the total model input on the real 89.94-second `MitaKind.wav`, but this alone is **not a reliable surrogate for inference time**. Both the HuBERT encoder and the RVC TextEncoder contain attention with pairwise time interactions. Longer chunks save repeated convolutional context and dispatches but increase attention work. A single “always use 900 frames” policy optimizes only one term in the overall computation graph.

Stage 93 adds `RVC_CHUNK_STRATEGY=cost-aware`, a whole-recording **shortest-path / dynamic-programming** planner. Each edge is one independent chunk with left/right model context. It minimizes a dimensionless analytical cost:

`launch * chunks + linear * ΣL + (attention_milli / 1000) * ΣL² + splice * Σnormalized_boundary_energy`.

`L = core_frames + 2*context_frames + trailing_frames`. The boundary energy is a normalized 20-ms RMS energy centered on each cut; the total cost accounts for *all* boundaries. Complexity: O(T*max_core_frames) time and O(T) planner memory. Legal windows never exceed 1000 frames including context. There is no hidden fallback or change of default strategy. All schedule-dependent source flags (Stage 91–92) remain opt-in.

**This objective is a cost proxy, NOT wall-clock milliseconds.** The attention calculation is a shape proxy, not a model-specific FLOP counter, and runtime batching currently groups only neighboring equal-length model inputs. Real RTX measurements are required to calibrate coefficients and to establish speed.

## Real-WAV geometry

Input `MitaKind.wav`: 44,100 Hz PCM, resampled by the existing Rust windowed-sinc algorithm and zero-phase high-pass filtered. The resulting 16-kHz PCM contains **1,439,028 samples (8,994 frames)**. Model geometry: 900 core frames maximum, 40 left + 40 right context and 4 trailing frames.

| Strategy / proxy coefficients | Chunks | Σ input seconds | Σ L² | Mean RMS boundary energy |
|---|---:|---:|---:|---:|
| fixed-300 | 32 | 120.54 | 4,593,636 | 0.0012603635 |
| fixed-900 | 11 | 100.38 | 9,721,764 | 0.0004394433 |
| quiet-splice-900 | 10 | 98.34 | 9,670,778 | 0.0003815591 |
| quiet-splice-900 +1 | 11 | 99.18 | 8,944,880 | 0.0000001789 |
| cost-aware default: launch=10000, linear=10, attention_milli=10, splice=100 | 10 | 98.34 | 9,670,778 | 0.0003815591 |
| cost-aware launch-dominated: launch=100000, linear=10, attention_milli=10, splice=100 | 10 | 98.34 | 9,670,778 | 0.0003815591 |
| cost-aware attention-dominated: launch=10000, linear=10, attention_milli=100, splice=100 | 26 | 111.78 | 4,806,104 | 0.0000372477 |
| cost-aware cut-sensitive: launch=10000, linear=10, attention_milli=10, splice=10000 | 11 | 99.18 | 8,943,164 | 0.0000002080 |

`Σ L²` is a shape estimate, **not actual GPU attention FLOPs**. The default and launch-dominated models prefer the lowest number of invocations. The attention-heavy proxy prefers shorter windows, while the cut-sensitive model favors truly quiet boundaries.

In particular, relative to fixed-300, quiet-splice-900 reduces `Σ L` by 2,220 frames but increases `Σ L²` by 5,077,142. Whether it is faster depends on the *actual effective* quadratic-work coefficient and launch/batching costs.

## Code and integration

- `models/tts-with-rvc-rs/src/chunk_plan.rs`: `ChunkStrategy::CostAware`, validated dimensionless coefficients, global shortest-path planner, deterministic tie-breaking, energy regularization, new tests including reference exhaustive search.
- `models/tts-with-rvc-rs/src/chunked.rs`: uses the new strategy only on explicit opt-in; contiguous, nonoverlapping coverage and existing exact final-output sample count; zero crossfade for all contiguous splice strategies.
- `research/stage93-cost-aware/measure_real.rs`: reproducible measurement with the real Rust resampler/filter. No ML weights or internet access required.
- `research/stage37-windows-gpu-bench/windows/Run-RVC-Full-Stage93.ps1`: A/B/B/A quiet-splice vs cost-aware, both with global F0/phase/noise, saves output WAVs and benchmark JSON. **Windows binary must be built from Stage 93 source.**
- `.github/workflows/rvc-stage93-global-compute-plan.yml`: Windows syntax, Rust tests, typecheck, build, artifact.

Environmental coefficients (`0..=1,000,000`, no all-zero model): `RVC_COST_LAUNCH`, `RVC_COST_LINEAR`, `RVC_COST_ATTENTION_MILLI`, `RVC_COST_SPLICE`. These are deliberately **unitless knobs**, and should be fitted using target-device timing data rather than treated as measured hardware constants.

## Verified gates and remaining uncertainty

- Independent Rust 1.100.0 harness: **22 PASS, 0 FAIL**. Includes exhaustive small-reference optimality, deterministic segmentation, boundary/length/partial-frame coverage, quiet cuts, and unchanged fixed/Stage90 paths.
- `cargo check --offline --locked` on Rust 1.100.0 for unified **CPU/CUDA/WGPU library and executable: PASS**.
- On the real source WAV: Rust resampler + high-pass filter + scheduler executed, with geometry evidence in `evidence/real_wav_plans.jsonl`.
- No GPU runtime/JIT benchmark performed, no audio quality claim, and no guarantee that a different plan reproduces the baseline WAV. Global noise/phase addresses source continuity but NOT HuBERT GroupNorm or Transformer context changes.
- Runner order A/B/B/A reduces order bias but does not independently measure GPU thermal/clock state; it records `nvidia-smi -q` for diagnostics.

## Next research gate

Calibrate the cost model from full-WAV RTX timings and per-stage traces (HuBERT frontend, transformer, TextEncoder, NSF). Ideally replace fixed coefficients with observed shape-and-batch latency models. A more complete optimizer would treat equal-length tensor batching explicitly rather than assuming one invocation per segment. Compare seam discontinuity separately; a 26-chunk plan is only acceptable if end-to-end quality holds.
