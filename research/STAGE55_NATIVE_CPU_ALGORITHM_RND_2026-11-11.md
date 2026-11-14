# Stage 55 — native RVC CPU/LLVM algorithm R&D, real assets

**Date:** 2026-11-11. **Repository:** `Atm4x/Vendor-for-burn`, experimental PR #2. **Scope:** local Flex/LLVM CPU research; no claims about CUDA inference speed or fidelity. No model checkpoints, audio or FAISS index are committed.

## Completed work

1. HuBERT positional **grouped Conv1d** now loads kernel-dimension weight-normalized values on host and applies the same `Standard`/`Prepacked` layout selection as other Conv1d blocks, rather than uploading un-prepacked weights first. Covered representative kernels/groups, the masked positional encoder, and the saved HuBERT fixture.
2. Native `IndexIVFFlat` retrieval has an exact **branch-and-bound scan of the coarse centroids** when `nprobe=1` (the supplied CrazyMita index uses `nprobe=1`). Starting from the first complete squared-L2 distance, every subsequent centroid is accumulated in original element order in 32-element chunks. If a nonnegative partial squared distance exceeds the current best, that centroid cannot win; the rest of its coordinates need not be read. Ties continue through the original `(distance,id)` ordering. This is **not** approximate IVF and does not change `nprobe`, `k=8`, inverse-square weights or feature blending.
3. `nprobe>1` retains the exhaustive centroid ranking path. Non-finite stored vectors and centroids are rejected at index load, so the nonnegative partial-sum bound is valid on accepted data. The compiler/backend receives no FAISS or CUDA-specific dependencies.
4. Added the real-index research benchmark `bench_faiss_real_stage55`, which records output fingerprints, selected neighbor IDs and exact FP32 distance bits for `nprobe=1,2,4`.

## Local evidence

Rust 1.98.1. `cargo test --offline --locked -p tts-with-rvc-rs --lib --features flex` with `RVC_REAL_FAISS_INDEX` set: **22/22 PASS**. Checks include exact bounded-vs-full-scan distance, centroid ties, 24 real-index queries, grouped HuBERT positional Conv1d, and other regression tests. Combined Flex+CUDA `cargo check --offline --locked -p tts-with-rvc-rs --features cuda`: **PASS** (compile only, not GPU execution).

Real index characteristics: 768 dimensions, 61,690 vectors, 1,581 IVF lists, `nprobe=1`. For `index_rate=0.75`, the original exhaustive build and optimized build produced identical FP32 feature fingerprints across all 10 recorded combinations of row count and index rate; neighbor IDs and **distance bits** matched in the recorded `nprobe=1,2,4` probes.

Benchmark conditions: **debug CPU**, 5 Rayon threads, one captured unoptimized baseline process and five independently launched optimized processes; no stable release benchmark or CPU utilization normalization. Rows >= 32 use Rayon as before; rows < 32 use serial processing. Times below include retrieval blend, not model inference or checkpoint loading.

| Frames | Path | Baseline (one run) | Optimized median (five runs) | Optimized range | Baseline / median |
|---:|---|---:|---:|---:|---:|
| 1 | serial | 26.41 ms | 7.27 ms | 7.19–7.80 ms | 3.63× |
| 7 | serial | 190.89 ms | 48.15 ms | 45.22–51.23 ms | 3.96× |
| 31 | serial | 775.91 ms | 223.70 ms | 213.66–241.95 ms | 3.47× |
| 32 | Rayon | 358.28 ms | 74.61 ms | 71.38–79.35 ms | 4.80× |
| 33 | Rayon | 397.92 ms | 75.97 ms | 69.75–89.30 ms | 5.24× |

The speedup ratios are **indicative**, not statistically matched confidence intervals, because the original baseline was measured once.

**End-to-end proof:** `CrazyMita.pth + hubert_base.pt + rmvpe.pt + CrazyMita.index + MitaKind.wav`, input segment [0.00, 0.35] seconds, Flex CPU debug and Prepacked enabled. All stages completed, output was 13,600 PCM16 samples at 40 kHz, and the resulting WAV was **byte-for-byte identical** to the prior Stage 55 pre-optimization run (SHA-256 `7871a2ff769ccb9d878e7cf2ebe465964bc1825811ca1ca5fefce3fc26ba59ee`). In-run reported retrieval time fell from 0.474 s previously to 0.155 s; comparisons of other stage timings across cold/debug runs are noisy and should not be interpreted as reliable speedups. This short test proves function, not perceptual voice quality.

## Limits and next focus

- **No end-to-end CUDA results:** GitHub Actions may compile the Windows EXE, but real CrazyMita CUDA runtime correctness and latency remain unverified until the user's RTX 4060 produces `RVC_STAGE54_RESULTS.zip`.
- **No production CUDA Graph for real weights:** Stages 51–53 demonstrate dynamic Graph only for synthetic NSF; it is not connected to `RvcV2Inferencer`.
- **Not yet a long-lived inference service:** user-facing model IPC/lifecycle in NeuroMita Rust remains future work.
- Debug/LLVM timings are not credible predictors of optimized Windows CUDA timings. A subsequent matched **release** benchmark should evaluate whether the extra branch checks still help and whether full `nprobe>1` search merits a separate bounded top-n selection algorithm.
- Original `tts-with-rvc` Python retrieval applies FAISS `index.search(..., k=8)` with `square(1/score)` weighting; native blending preserves this semantic form. No claim of complete PyTorch-to-Rust parity for full-length audio has been made.
