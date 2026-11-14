# Stage 56 — native RVC release/LLVM CPU, FAISS R&D and long-audio preparation

Date: 2026-11-11. Repository: `Atm4x/Vendor-for-burn`, experimental PR #2. This stage makes no production GPU-performance claim. No private `.pt`, `.pth`, `.index`, `.wav` or synthesized speech is committed.

## Source basis

The local repository was restored from the Stage 52 Git Bridge bundle (`source-head=e9134c27`) and successive source-only patches for Stages 52–55. Reconstructed **Stage 55 Git tree**: `aad546c10b945bba91a50f72b8948666d14a0c91` — the same as the tree previously verified after publication. Rust toolchain: **1.98.1**, offline vendored dependencies.

## Production change: one high-pass per WAV

The WAV→RVC→WAV user CLI previously invoked `highpass_48hz_zero_phase` twice on the same resampled 16-kHz PCM:

1. Internally in `infer_long_16khz_batched` before chunking and inference.
2. Again in `process_wave_job` for RMS-envelope matching.

`RvcV2Inferencer::infer_long_filtered_16khz_batched` now accepts explicitly prefiltered PCM; the ordinary public `infer_long_16khz_batched` retains its original behavior by filtering and forwarding to the common implementation. `process_wave_job` filters exactly once and reuses the same prepared audio for both model inference and RMS matching. Thus the duplicate O(N) whole-audio preprocessing pass is eliminated without changing the default library API or numerical processing.

**Real-weight CLI parity:** `CrazyMita.pth`, `hubert_base.pt`, `rmvpe.pt`, `CrazyMita.index`, first 0.35 s of `MitaKind.wav` (original 44.1 kHz) using RVC Prepacked/CPU and index rate 0.75. The WAV before and after this change has **identical SHA-256**: `807f55e6379dc3cb0e4427aa0e341cd4073ed841c7cbbe7050eceea744c90f32`; output is 13,997 samples at 40 kHz. One observed cold CLI wall-clock run was 3.82 s before and 3.38 s after; this is **not a controlled speedup measurement** because model loading and cache effects dominate.

## Real-weight release/LLVM CPU proof

The whole 0.35-second `infer-wav-segment` route (HuBERT→RMVPE→FAISS→TextEncoder→Flow→NSF) completed using the supplied real assets. Measured per-stage times in the Stage 56 Prepacked run, excluding load/setup/encoding:

| Stage | Seconds |
|---|---:|
| HuBERT | 0.128 |
| RMVPE | 0.110 |
| FAISS retrieval | 0.011 |
| TextEncoder | 0.018 |
| Flow | 0.016 |
| NSF | 0.164 |

Sum approximately **0.447 s** for 0.35 s input. These timings are a single warm-stage profile, **not** a median, GPU result, or cold end-to-end latency. Baseline and Prepacked on the same **release CPU build** produced binary-identical 13,600-sample PCM16 WAVs. The earlier debug CPU WAV differed at 26 sample positions, each by at most one PCM16 unit (expected different rounding across optimized builds), not bitwise parity across build profiles.

## Extended regression matrix

`cargo test --release --offline --locked -p tts-with-rvc-rs --lib --features flex`, with `RVC_REAL_FAISS_INDEX` pointing to the local private index: **25 PASS, 0 FAIL**.

- Native FAISS search versus an independent exhaustive reference: 28 nontrivial real-index queries, `k∈{1,4,8,16}`, `nprobe∈{1,2,4,8}`; **neighbor IDs and FP32 distance bits identical**. Small artificial indexes also exercise ties, `k` larger than list length, and varying queries.
- Long-audio overlap-add: 1, 159, 160, 161, 3201, 159997 and **1,440,017 input samples** (about 90.0 seconds at 16 kHz). Scalar chunk execution and 4-way chunk batching produce identical outputs and progress events, including partial final frames. The mock decoder reproduces input samples after 16→40 kHz nearest-sample conversion with max error within `2e-6`. **This is chunk orchestration correctness, not 90-second neural inference quality.**
- Prior Stage 55 HuBERT, Conv1d, FAISS, resampling and preprocessing tests remained passing.

## Rejected candidate algorithms (not shipped)

A new branch-and-bound exact top-8 implementation sorted a tiny candidate buffer and pruned squared L2 distances. Despite parity, a matched **release** benchmark on 33 real-index queries showed **15.83 ms Stage 55 vs 16.27 ms new variant**. `select_nth_unstable` also regressed (16.50 vs 16.87 ms). Both variants were removed from the production FAISS path.

Centroid early-abandon block-size sweep (release, 33 queries, eight alternated measurements within one process):

| Block | Median milliseconds |
|---:|---:|
| 16 | 15.23 |
| **32 (Stage 55)** | **15.10** |
| 64 | 15.53 |
| 128 | 16.11 |

An upper-bound seeding approach that scanned 8–64 spaced centroids before bounded search was measured in **seven independent release processes**. Median of per-process medians: baseline 15.48 ms; 8 seeds 15.68; 16 seeds 15.67; 32 seeds 15.99; 64 seeds 16.85. Any isolated apparent improvement did not reproduce. Seeded variants were discarded.

These are narrow FAISS search microbenchmarks on the supplied index, not full RVC speedups. The accepted Stage 55 exact bounded centroid scan with block size 32 is retained unchanged.

## Known limitations / Stage 57

- No local RTX 4060. End-to-end real-weight CUDA correctness and timing remain **NOT TESTED**; prior hardware CUDA Graph results apply to **synthetic NSF**, not full RVC.
- High-pass reuse is verified by real WAV identity but the incremental per-WAV latency benefit needs matched repeated measurements excluding model initialization.
- Long-audio tests use mock DSP; full 90-s neural inference was intentionally not run on the five-CPU virtualized environment.
- HuBERT and RMVPE currently read intermediate tensors to CPU before FAISS/F0 postprocessing; next priority is a measured, safe way to reduce GPU↔CPU roundtrips, preserving retrieval semantics.
- Native long-lived model-worker protocol, dependency packaging and NeuroMita Core integration remain independent workstreams, not completed by Stage 56.

## Backend compile evidence

Rust 1.98.1, `cargo check --offline --locked -p tts-with-rvc-rs --features cuda --all-targets`: **PASS**. The first check failed during a third-party build script because `tracel-llvm-bundler` tried to download its LLVM distribution without network access. The user's supplied `linux-x64.tar.xz` contained the expected **Tracel LLVM 23.1.0-4**; it was installed in the local bundler cache and verified with `llvm-config --version`. The retried offline build completed successfully, with no edits to vendored libraries. This is CUDA **compile** validation only: the runtime CUDA RVC path still needs RTX 4060 execution.
