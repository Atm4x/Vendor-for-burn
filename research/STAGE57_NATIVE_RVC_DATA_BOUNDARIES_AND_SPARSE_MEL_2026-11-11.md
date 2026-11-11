# Stage 57 — Native RVC data-boundary audit and RMVPE sparse Mel projection

Date: 2026-11-11. Repository: `Atm4x/Vendor-for-burn`, experimental PR #2. This is **LLVM/CPU validation** with real CrazyMita weights; no local NVIDIA GPU is available. Inference remains entirely native Rust/Burn, without Python or PyTorch. No private model weights, FAISS index, or WAV samples are tracked in Git.

## Scope and baseline

Stage 56 was published at GitHub commit `a8c17e90dab4634abbb02a2db539654d9a25cb56`, tree `8dc69f978d73655dfb325739c812aefc4e71025f`. Tests use Rust 1.98.1 and vendored offline dependencies. The earlier end-to-end RVC route and high-pass reuse remain unchanged. Original Python `tts-with-rvc` is reference-only.

## Change A — HuBERT content stays in the tensor backend when retrieval is disabled

Previously `RvcV2Inferencer::infer_16khz_batch()` always transferred the complete 50 Hz HuBERT content tensor to the host, doubled frames there and created a new device tensor, including when `index_rate=0` and no FAISS index was requested.

Now `phone_features::upsample_without_retrieval()` uses tensor-native `unsqueeze_dim + repeat_dim + reshape + slice` for the homogeneous no-retrieval path. No CPU readback or upload is performed for that feature tensor. This reproduces `phone[frame] = hubert[frame/2]`, including truncated odd frame lengths and independent batches. It is **not** a new custom CUDA kernel or global CubeCL change: it composes existing Burn/CubeCL tensor operations.

For any positive index rate, the existing CPU FAISS blending path and its `protect < 0.5` behavior remain intact. Mixed batches with any indexed option continue to use the existing host feature materialization for correct per-item behavior. The `RVC_PROFILE` HuBERT line now reports `host_content_bytes` (logical bytes downloaded at this boundary, not wall-clock transfer time).

On the 0.35 s test, the indexed path used `52,224` bytes for D2H HuBERT content. The no-index path used **0**. It also eliminates the corresponding host-to-device upload of `B * frames * 768 * sizeof(f32)` bytes. This savings alone does not establish faster real CUDA inference.

## Change B — RMVPE Mel projection visits only supported FFT bins

The 128×513 dense Mel basis has **65,664** coefficients, only **1,010** of which fall within the nonzero support of the filters (sum of contiguous support spans). Every FFT frame previously multiplied 65,664 pairs even though most filter coefficients are zero. `RmvpeMelSpectrogram::new()` now records the first and last nonzero bin of each band. Runtime accumulation uses the same coefficients, same bin order and same summation, but restricted to each band's span. The Mel formula, window, reflect padding, FFT and log clamp are unchanged.

Verification: 18 input/length combinations (seeds 0/1/9; lengths 513, 801, 3,200, 5,600, 17,123, 32,017 samples) yielded **bitwise identical FP32 Mel outputs** versus a separate dense-reference implementation. The reference benchmark is an ignored-by-default test and is run explicitly when needed.

Matched local release CPU microbenchmark: 160,000 mono 16-kHz input samples (10 s); 16 alternated process-local measurements per variant, same PCM and FFT model; median dense **55.1197 ms**, sparse **3.2614 ms**, ratio ~**16.9×**. This measures **Mel extraction only**, **not** RMVPE neural network, RVC end-to-end, or CUDA throughput. No full-route speedup is claimed from individual noisy stage traces.

## Real-weight end-to-end correctness

Input: `MitaKind.wav`, CrazyMita RVC v2 voice, HuBERT, RMVPE, optional CrazyMita FAISS index. `RVC_CONV_WEIGHT_PREPACK=1`, Flex/LLVM release CPU, deterministic default seed. Baseline executables were saved before each optimization and compared with the new build.

| Segment / index mode | Before / after SHA-256 | Result |
|---|---|---|
| 0.35 s from 0, index disabled | `54fd4485f9bfb6a6cd998bbc92ee56ff702d0e1d27a4f048b50105fc814df9ab` | Binary-identical WAV |
| 0.35 s from 0, index rate 0.75 | `54c1ebede9bbc97e0f08e697b9e85bd756190419da01e47e6a8adf9b07101638` | Binary-identical WAV, unchanged from Stage 56 |
| 0.70 s from 22.4, index disabled | `1e70e1b98fc455d76da9cf2ca138a6fd9b2b66200d5007f2ce3fab6315f24b12` | Binary-identical WAV |

The WAV parity is based on exact file comparison, not merely RMS or sample probes. This does not validate real Nvidia CUDA execution.

## Build and regression results

- `cargo test --release --offline --locked -p tts-with-rvc-rs --lib --features flex`: **27 PASS, 0 FAIL, 1 deliberately ignored benchmark**.
- `cargo test --release --offline --locked -p tts-with-rvc-rs --lib --features flex benchmark_sparse_mel_against_dense -- --ignored --nocapture`: **PASS**; timed 16 comparisons.
- `cargo check --offline --locked -p tts-with-rvc-rs --features flex,cuda --all-targets`: **PASS**.
- `git diff --check`: **PASS**.
- Real-weight CPU segment tests: **3/3 output byte identities PASS**.

## Honest remaining data boundaries

- For `index_rate > 0`, host-side FAISS retrieval still requires a HuBERT D2H readback and a rebuilt phone tensor H2D upload. GPU-native retrieval would require an independent GPU IVF implementation with exact equivalence; it is not in this stage.
- RMVPE still reads the complete `[B,T,360]` salience tensor to CPU to decode voiced bins and the 9-bin weighted local centroid in FP64; this may be an optimization candidate, but no GPU kernel or numerical equivalence proof is supplied here.
- Input audio, pitch, Gaussian source and generated output have other expected device boundaries; do not extrapolate the no-index saving to all transfers.
- CUDA Graph in actual RVC inference (versus synthetic NSF) and full GPU end-to-end parity/performance on RTX 4060 are **NOT TESTED**.
- Native shared type-specific NeuroMita service-worker integration is separate; this commit changes only native model implementation, not NeuroMita Core or a worker framework.

## Next

1. Obtain `RVC_STAGE54_RESULTS.zip` from actual RTX 4060 if available, inspect per-stage runtime and compare WAVs against CPU for matching options.
2. Investigate compact RMVPE salience transfer: GPU top-bin plus 9 neighboring salience values, CPU FP64 centroid postprocessing, with exact tie/boundary behavior and parity; keep as an experiment until full output parity and GPU-only improvement are demonstrated.
3. For the common `index_rate=0.75` path, profile CPU FAISS versus GPU D2H/H2D overhead. Avoid promising GPU residency until FAISS retrieval itself can run natively and correctly on GPU.
4. Validate longer real speech (multiple segments, offsets, speaker IDs), and compare CPU production variants only under matched release conditions.
5. Provide service-worker adapter through the existing shared model-type worker contract, not a separate worker per checkpoint or backend.
