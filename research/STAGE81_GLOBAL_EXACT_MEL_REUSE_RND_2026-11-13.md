# Stage 81 — Exact cross-chunk STFT/Mel reuse for RVC (originality experiment)

Source parent: Stage80 a688f21f (Stage79 reference recovery). PR #2 experimental source branch.
**Algorithm implemented, Rust unit tests PASS; no RTX4060 full-WAV timing yet.**

## Problem reframing

Instead of optimizing TensorCore rounding for one Conv2d, treat full RVC as a dataflow graph with repeated temporal work. On 89.94 s of PCM16k, the currently used 300-frame, B=2 planner extracts 32 local windows containing about 120.54 s of audio. The same samples enter HuBERT, RMVPE and NSF repeatedly, but their intermediate activations cannot all be shared unchanged: HuBERT's first GroupNorm operates over the current segment, both HuBERT attention and RMVPE bidirectional GRU have temporally nonlocal dependence, and NSF uses per-chunk phase/noise. Blindly reusing final F0 or latent vectors would be a semantic change.

Stage81 extracts the **mathematically exact shareable subgraph**: STFT1024/HOP160 + 128-band log-Mel preprocessing of RMVPE. Every original sample's STFT frame is computed once at source scope; only windows requiring *local chunk-boundary reflection* are computed locally. Neither RMVPE U-Net nor GRU or F0/voicing thresholds is touched, and the original Direct Conv2d still runs. This is a real elimination of duplicate work, not a compensatory approximation to a TF32 error.

## Exactness condition and proof sketch

A local chunk starts at original sample-grid offset `o = (chunk_start_frame - context_frames)*160`. Its STFT frame `j` has center `160*j` and window indices `[160*j-512,160*j+511]` in the local chunk. If this entire interval is inside the chunk, the original chunk signal at sample `k` equals `reflect(global_pcm, o+k)`. The global STFT centered at absolute position `o+160*j`, provided it is nonnegative and belongs to the source frame lattice, reads the same sequence of 1024 floats, in exactly the same order and with the same Hann, FFT and Mel summations. The resulting 128 f32 Mel values therefore have identical bits. If any predicate fails, the code performs the original local STFT; no approximation is introduced.

Per-frame source samples remain unmodified. The reuse cache has a temporary 128 MiB cap; the production flag is off. If the cap is exceeded, opt-in reports an explicit error instead of silently switching algorithms. Future stages should replace the full-recording cache with bounded streaming tiles.

## Implementation

- `rmvpe_mel.rs`: `MelReuseCache`, `MelReuseStats`, exact `forward_with_cache` with guarded 1024-sample window; `forward()` is the unchanged algorithmic path through the same kernel; batch Mel upload preserved.
- `rvc_inference.rs`: `RVC_RMVPE_MEL_REUSE=1` activates a shared Mel cache on already filtered PCM and passes original sample-grid offsets through the batching scheduler to RMVPE. Supports both CPU salience readback and GPU-resident F0. Explicit incompatibility with global F0 (already skips per-chunk RMVPE).
- `main.rs`: benchmark report includes `rmvpe_mel_reuse` boolean.
- Windows PowerShell Stage81 A/B: Direct full WAV baseline vs Direct with Mel cache; requires exact PCM16 equality and no F0 reference injection.
- Workflow Stage81: Rust unit parity, PowerShell parse, unified Windows build and zipped EXE.

## Verified locally

Rust 1.98.1, offline vendored packages, user-supplied LLVM23 Linux archive:

`cargo test --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu --lib mel_reuse_tests`

**2 passed, 0 failed** on source lengths 3200,8009,16000,42093,87773 samples, 35/93-frame core geometries, negative and reflected windows, tail partials. The tests assert `f32::to_bits` equality for every log-Mel value and verify local STFT boundaries are recomputed. All FFT-sharing input index matches were separately checked. The first attempt failed only due missing LLVM23 offline integration and was fixed by installing user-supplied archive outside the repository; no vendored source changes.

## Performance hypothesis (not an RTX measurement)

For `core=300`, current full 89.94 s recording has about 12,054 model-input frames across 32 chunks. Shared Mel computes roughly 8,994 original global frames plus at most ~8 local edge frames per chunk; this potentially removes about one quarter of repeated **CPU STFT operations**, at the expense of a ~4.6 MiB Mel cache and reconstruction copies. The unchanged GPU HuBERT, RMVPE U-Net/GRU and NSF still dominate much of full inference, so the end-to-end benefit may be small. Do not extrapolate a percentage of full RVC from the FFT count.

## Stage82 research direction (NOT yet implemented)

Treat RMVPE U-Net as a temporal local operator until BiGRU, compute its receptive field analytically from the encoder/decoder layers, and derive a boundary-correct tiling theorem for its intermediate tensor. If the receptive field is bounded by available overlap, share U-Net outputs across chunks with an explicit cached boundary region, then run chunk-local BiGRU without changing its initial hidden state. Compare original full PyTorch FP16/FP32 and Rust Direct first (Stage80 has recovered the original snapshot); separate speed attributable to redundant model execution from precision. The expensive shared CNN, not the small Mel FFT, is the intended next global algorithmic target.

## Windows command

After Stage81 CI produces `rvc-stage81-global-mel-reuse-windows-x64`, unpack it beside the unchanged `embed` directory; **Stage78 executable cannot enable this new flag**. Run:

`powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage81.ps1 -Device cuda -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -WarmRuns 3 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -Modes all`

Expected `RVC_STAGE81_FULL_WAV_RESULTS.zip`. Validate `failures=[]`, `baseline.wav` and `cached.wav` identical SHA256, per-run times stable, real F0 outcomes unchanged. Repeat on additional recordings and core sizes before any default change. This stage leaves original and legacy scripts alone.
