# Stage 91 — Global Fundamental-Phase Consistency for Long-Window RVC

**Parent:** Stage 90 global quiet-splice segmentation, building on the original PR #2 experimental branch. **Toolchain:** Rust 1.100.0. **Scope:** optional correction for fundamental oscillator phase in NSF source generation, without altering the production route or original Python files.

## Problem and why this is architectural

Stage 90 partitions one recording into contiguous core segments and synthesizes each with reflected context. The original native RVC SineGen receives a hard-coded zero phase for the fundamental at **every segment**. That resets the oscillatory source independently of the global F0 timeline. More or fewer chunks can change the excitation waveform even when every local F0 frame is otherwise identical. No convolution kernel micro-optimization addresses this.

The global phase is the cumulative wrapped cycles of the fundamental source: for frame `t`, the excitation has an advance based on the pitch-scaled F0 of frame `t`, its 400 output samples, and the 40 kHz source rate. For a chunk whose input begins at absolute frame `s - context`, we seed its oscillator with the cumulative phase at that frame; negative frame indices follow exactly the same reflected global F0 timeline that the segment consumes. The global timeline is calculated once and queried in O(1) for nonnegative starts. Float64 is used for the accumulator and Float32 for the GPU injection.

This is not an additional F0 estimator or a layer cache. The algorithm changes the boundary condition of the **generator itself** so that segmentation does not arbitrarily reset the fundamental sinusoid.

## Implementation

- `models/tts-with-rvc-rs/src/nsf_phase.rs`: `GlobalNsfPhase`, deterministic cumulative cycles, valid negative reflected context, pitch-shift support, validation. No GPU operations are required to compute a scalar chunk phase.
- `models/tts-with-rvc-rs/src/nsf_source.rs`: new optional `forward_with_random_and_global_phase` which adds one `[batch,1,1]` phase tensor to the original SineGen angle before sine evaluation. Legacy calls still dispatch with `None` and follow the existing arithmetic.
- `models/tts-with-rvc-rs/src/decoder.rs`: pass optional source phase into `forward_from_f0_with_execution_and_phase`, preserving the previous method and parameters.
- `models/tts-with-rvc-rs/src/rvc_inference.rs`: `RVC_NSF_GLOBAL_PHASE=1` requires `RVC_GLOBAL_F0=1`, computes phase offsets from the same global F0 prepass used for chunk F0 slices, and transfers one scalar per batch element. Existing model flags, weights, and random seeds are unchanged.
- `models/tts-with-rvc-rs/src/main.rs`: adds explicit `nsf_global_phase` to full-WAV benchmark metadata.
- `research/stage37-windows-gpu-bench/windows/Run-RVC-Full-Stage91.ps1`: keeps the established PowerShell interface and executes A/B/B/A, switching only `RVC_NSF_GLOBAL_PHASE` while holding quiet-splice, original WAV, global F0, model precision and noise settings constant.

This mode intentionally cannot run with per-chunk GPU F0 because a global scalar F0 timeline is needed. The existing code also rejects simultaneous global F0 and Stage81 Mel cache.

## Tests, independent reference and falsifiability

- **Rust 1.100.0:** `rustc --edition=2024 --test research/stage91-nsf-phase/phase_test_harness.rs` gave **19 PASS / 0 FAIL**, covering the four new phase tests and fifteen Stage90/legacy chunk geometry tests. Phase tests cover frame-start alignment, reflected precontext, voicing gaps, four pitch shifts, three chunk sizes, invalid inputs, and deterministic phase states.
- **PyTorch 2.10 CPU:** an independent full-length RVC SineGen formula on synthetic F0 (9,000 frames, 3.6M samples) was compared against independently rendered 300-frame cores using Rust-produced phase offsets. **1,080,000 source samples compared across 9 segments.** Reset-per-chunk phase: max abs 0.1996049, RMSE 0.1090586. Global phase: max abs **0.000041879**, RMSE **0.000014239**. Differences remaining are due to floating-point accumulation order. This is an *oscillator-only* comparison; it is not full-RVC WAV parity or a listening evaluation.
- **Rust 1.100.0 Cargo:** `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu,cuda,wgpu --lib --bin tts-with-rvc-rs` **PASS**. Offline LLVM 23.1 supplied by the user was installed in the toolchain cache; no network dependency was used.

## Limitations

The phase-offset correction changes the waveform as intended, and thus cannot be bitwise identical to the old reset-phase mode. It does not globalize stochastic latent Gaussian noise, the separate harmonic noise stream, global HuBERT activation context, or receptive-field-sensitive NSF and RMVPE activations. Consequently, it **does not prove** continuity of the final waveform or audio quality and **does not claim any speedup**.

The independent reference tests validate the phase law. GPU JIT, the real `CrazyMita.pth` end-to-end WAV, seam energy, and actual throughput require a later Windows run with the new EXE. Keep disabled by default until that gate passes.

## Next R&D target

The next structural bottleneck is long-context execution with boundaries that preserve model semantics. Rather than guessing a larger fixed frame count, derive exact receptive-field dependencies where finite, then identify globally coupled layers that require online attention or cross-window state. Only after that can a 60-second original-Python-compatible plan be justified without output artefacts or uncontrolled peak VRAM.
