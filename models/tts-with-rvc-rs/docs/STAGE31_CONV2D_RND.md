# Stage 31 — Universal Conv2d: eliminating small-group scheduling overhead

Date: 2026-11-09. Branch: `experiment/conv2d-universal-implicit-gemm-20261109`. Experimental, not promoted to the production RVC branch.

## Design

`conv2d_f32` and `conv2d_f64` continue to enter **one universal native Conv2d engine** (`conv2d_nchw_tiled_gemm`), with no dispatch to the legacy `conv3d` API. The engine retains the dense packed-GEMM contraction for large channel reductions and an analytic row-plan contraction for small reductions; these are internal microkernels with the same convolution semantics.

For the analytic contraction, use the existing vector-friendly `conv_plane_accumulate` row algorithm with spatial validity ranges shared across channel groups. The entire spatial output plane is processed as one unit (no `im2col` allocation, no partial-row tile boundary). For its Rayon tasks, run `for_each` without `for_each_init` or temporary vectors. Packed-GEMM tasks retain their preallocated per-worker buffers. No public API signatures changed; `f16` stays on GEMM to preserve its reduction behavior.

The changed vendored `conv.rs` SHA-256 is recorded in `.cargo-checksum.json`. A targeted release fingerprint rebuild was necessary; Cargo initially reused stale dependent Burn rlibs, causing E0463. Rebuilding the downstream Burn fingerprint chain fixed the build. No full `cargo clean` was used.

## Release A/B — same compiled binary

FP32, AMD EPYC environment, warmup 6 calls then 20 interleaved runs per case, 2 Rayon threads unless indicated. Conv3d is an **external numerical/performance reference only**, never a runtime fallback from Conv2d.

| Case | Old Conv3d ms | Native Conv2d ms | Old/new |
|---|---:|---:|---:|
| 16→32, 128×128 | 5.28930 | 3.11707 | 1.70× |
| 64→128, 32×32 | 2.22178 | 2.15778 | 1.03× |
| 256→256, 8×8 | 1.55376 | 1.01196 | 1.54× |
| 512→512, 4×4 | 3.06224 | 0.91103 | 3.36× |
| Depthwise 16×16, 16 groups | 0.02384 | 0.02402 | 0.992× |

With 4 Rayon threads the depthwise pair was 0.04582 / 0.04566 ms (1.004×). This eliminates the previously large depthwise regression **for the tested shape**, within experimental noise; it does not prove every depthwise geometry is faster. Largest reported finite A/B discrepancy over the suite: `2.8e-7`.

## Correctness and actual RVC

- `cargo test -p tts-with-rvc-rs --release --offline --locked --jobs 2 --lib --test parity -- --test-threads=2`: 15/15 library tests and 30/30 PyTorch-fixture parity tests PASS.
- Freshly rebuilt `bench_stage30_conv2d_matrix`: six geometry parity cases (including grouped, batched, high padding, dilated and 132×132 small-kernel beyond previous packing tile size) plus non-finite padding PASS. Max finite absolute discrepancy: `1.49e-7`.
- Freshly rebuilt RMVPE, real `rmvpe.pt`, 1.15 s WAV: 116 frames, 88 voiced; F0 outputs identical when measured with Rayon 2 vs 4 threads. Median 181.017 ms (2 threads) and 152.374 ms (4 threads), after warmup; timings varied significantly between runs, so do **not** infer cross-session performance uplift from these model numbers.
- Freshly rebuilt `tts-with-rvc-rs` release binary: complete HuBERT→RMVPE→FAISS→TextEncoder→Flow→NSF, 10.1 s WAV, four chunks, tensor batch B=2, 2 CPU workers. 404,000 output samples at 40kHz; 28.87 s wall-clock including weight loads; peak RSS 1,396,216 KiB. Resulting int16 PCM was byte-identical to the Stage 30 output (`max_abs=0`, `RMSE=0`). No private weights, audio or FAISS index belong in Git.

## Still open

1. Check wider randomized f16/f64/bf16, IEEE NaN/Inf, extreme padding and zero-channel dimensions with the full backend suite. Current changed path was exercised by limited targeted tests and generic RVC parity.
2. 90-second RVC acceptance remains open. Stage 30 timed out on 28/32 chunks; do not repeat without a longer strict limit or resumable chunk checkpoints.
3. Compare warmed-up full RMVPE and full RVC against PyTorch under controlled, uncontended CPU conditions; timings above do not establish overall speedup.
4. Test Linux and Windows build/CI on the experimental branch and GPU backends separately before any production promotion.

Sparse validation discipline: one Cargo operation at a time, offline, locked, two build jobs, no entire workspace clean and no uncontrolled concurrent CPU benchmark.
