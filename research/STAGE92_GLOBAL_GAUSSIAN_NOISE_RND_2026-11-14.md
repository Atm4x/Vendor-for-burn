# Stage 92 — Partition-Invariant Gaussian Source R&D

**Parent:** Stage 91 global fundamental oscillator phase, based on PR #2. **Toolchain:** Rust 1.100.0, offline dependencies. **Status:** compiled, source-only R&D; RTX and end-to-end audio acceptance not yet performed.

## Structural problem

Stage 90/91 can change RVC chunk geometry while sharing F0 and the fundamental NSF oscillator phase. However, `RvcV2Inferencer` creates one `Gaussian(seed + chunk_index*constant)` for each chunk and consumes it first for VAE latent samples and next for SineGen stochastic excitation. Inserting/removing a chunk, or changing channel/time layout, changes the random stream. Consequently, the **same absolute audio position has a different stochastic input** if the chunk plan changes. This has nothing to do with Conv1d kernel speed.

## Algorithm

`nsf_noise_field.rs` introduces a deterministic, random-access Gaussian field, keyed by `(seed, domain, channel, absolute position)` using SplitMix64 counter mixing and a paired Box–Muller transform. Separate domains ensure the latent and source streams do not accidentally coincide. Absolute frame and 40kHz sample indices are reflected at the global recording boundary. This yields the same random values independent of local chunk index, batch grouping, or order of evaluation. No giant precomputed full-audio noise buffer is required.

Integration is opt-in with `RVC_NSF_GLOBAL_NOISE=1` and requires `RVC_GLOBAL_F0=1`. The existing sequential Gaussian generator and its original seeded behavior remain unchanged without the flag. `RVC_NSF_GLOBAL_PHASE=1` may also be enabled. This redesign changes RNG identity and is **not** a PyTorch `randn` or legacy-Rust bit-exact reproduction; it preserves the Gaussian distribution and per-coordinate determinism, not identical random values.

The global recording is currently still segmented. This work **does not** guarantee exact full-WAV equivalence across different chunk plans because HuBERT normalization, TextEncoder, Flow and the NSF decoder have context-dependent activations, and the final audio is constructed from separately processed windows.

## Tests and measurements

- `rustc 1.100.0 --edition=2024 --test research/stage92-nsf-noise/noise_test_harness.rs`: **25 PASS / 0 FAIL**, including new global noise field, Stage 91 global phase, and Stage 90 chunk plan regression tests.
- `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu,cuda,wgpu --lib --bin tts-with-rvc-rs`: **PASS** on Rust 1.100.0, with offline LLVM 23.1 and bundled vendor dependencies.
- Optimized standalone CPU probe for a 8,994-frame recording (89.94s at 40 kHz): Gaussian excitation **3,597,600 samples**; mean **0.0005427**, standard deviation **0.9998734**, wall clock **112.948 ms**. Latent noise (192 channels × 8,994 frames) **1,726,848 values**, wall clock **47.506 ms**. These are single-run CPU measurements in the current Linux container; not GPU timings or production throughput.
- Exact random-coordinate comparison on core geometries 300, 600 and 900 frames: **PASS**. The generator also passes reproducibility, overlapping contexts, reflection, domain isolation, and invalid input tests.
- The Stage91 Windows CI failure was diagnosed from its actual GitHub Actions job log: PowerShell reports `InvalidVariableReferenceWithDrive` at the literal interpolated `$group:` token. The script now uses `${group}:`. This was a runner parse failure, not a CUDA/Rust compilation failure.

## Windows validation protocol

The Stage92 Windows runner executes baseline/global/global/baseline. Both variants use Stage91 global F0, global fundamental phase, the same `quiet-splice` geometry, model weights and inference settings. The **only randomized-path difference** is `RVC_NSF_GLOBAL_NOISE`. It checks metadata, deterministic repeated runs and collects WAVs and timing, but cannot assert that the experimental WAV is bitwise equal to baseline — they intentionally use different random realizations. Seam analysis and listening evaluation remain separate gates.

### Success criteria on RTX

1. No CUDA errors, finite WAV outputs for both variants, and exact SHA for repeated runs of each mode.
2. Correct long-WAV length and no gaps across quiet-splice boundaries.
3. RMS/click/seam metrics compared against the previous Stage91 mode; listening test for discontinuities.
4. A/B/B/A full-WAV timing with GPU conditions recorded; report any extra cost of deterministic counter Gaussian noise.

Production remains unchanged unless these gates pass. This stage targets segmentation-invariant excitation, not an immediate speedup or proof of original Python numeric parity.
