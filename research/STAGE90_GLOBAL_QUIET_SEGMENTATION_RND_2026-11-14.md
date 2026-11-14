# Stage 90 — original-RVC-oriented global segmentation R&D

**Parent:** Stage 89 `a51ebbc1700b311b09f7833007a9109849452a1f` from the private source branch, with Stage 88 already in GitHub PR #2. **Compiler:** Rust 1.100.0. **Scope:** new optional Rust inference strategy, no legacy, model-weight, or vendor edits.

## Structural problem

The current Rust full-WAV pipeline applies a fixed 300-frame core with 12 frames of overlap, 40 frames of context on each side, and four trailing frames. The source Python `tts_with_rvc/infer/vc/pipeline.py` is different: it globally high-pass filters and analyzes pitch before synthesis segments, searches for quiet cut locations around long-window boundaries, and crops the context before concatenation. The original Python FP16 config uses x_pad=3s, x_center=60s, x_max=65s; FP32 x_pad=1s, x_center=38s, x_max=41s. The present Rust inference API rejects segments longer than 10s, so Python's 60s geometry **cannot** be reproduced simply by changing a parameter.

Stage90 targets one global architectural bottleneck: 32 small Rust model invocations duplicate audio context and overlapping core work in HuBERT, RMVPE and NSF, independent of any faster convolution kernel.

## Implemented Stage90

`models/tts-with-rvc-rs/src/chunk_plan.rs`: deterministic global segmentation using dynamic programming. For input with `T` 10-ms frames and maximum core `C`, first choose the minimum `K=ceil(T/C)` number of segments (plus an explicit user-controlled `0..3` optional segment count). Candidate boundaries lie at most `min(60,C/8)` frames from each evenly spaced target. The objective minimizes locally measured mean-square signal energy in a 20-ms neighborhood, normalized by whole-record energy, plus a weak displacement penalty. Transitions reject segments larger than `C` or shorter than the permitted minimum. The scheduler chooses **all boundaries jointly**, not through local greedy choices. Per-frame squared energies are accumulated directly from PCM, keeping the planner's working memory O(T) for T ten-millisecond frames instead of O(number of samples).

`models/tts-with-rvc-rs/src/chunked.rs`: a first-class range-plan inference API. Old `infer_chunked_batched_with_progress` retains fixed-overlap behavior by default; `RVC_CHUNK_STRATEGY=quiet-splice` switches to the global partition. Non-overlap splicing uses unit weights, so there is no fade-to-zero or uncovered sample at boundaries. The same tensor batch grouping remains legal only for identical segment lengths, avoiding padding which would alter HuBERT normalization.

`models/tts-with-rvc-rs/src/rvc_inference.rs`: the same plan is passed to the model driver and to optional global-F0 and exact-Mel slicing. This avoids off-by-one discrepancies between preprocessing and output assembly.

`models/tts-with-rvc-rs/src/main.rs`: the full-WAV benchmark now records actual planned segment count, total model-input seconds, and all boundaries. Windows A/B/C/C/B/A runner covers fixed300, fixed900, quiet900+N; experimental engine remains off without the flag.

## Real original WAV: measured geometry, not GPU speed

Source: `MitaKind.wav` read from the user library, 44.1 kHz mono, 3,966,323 samples. The **actual Rust project's** `audio_io::resample` and `preprocess::highpass_48hz_zero_phase` produce 1,439,028 PCM samples at 16 kHz (89.9393 seconds). No proxy signal was used.

| Configuration | Segments | Total segment-input duration | Approx. batch-2 invocations* | Average source signal energy at cut |
|---|---:|---:|---:|---:|
| Fixed300, overlap12 | 32 | 120.54 s | 17 | 0.0012603635 |
| Fixed900, overlap12 | 11 | 100.38 s | 6 | 0.0004394433 |
| Quiet900, extra0 | 10 | **98.34 s** | 7 | 0.0003815591 |
| Quiet900, extra1 | 11 | **99.18 s** | 11 | **0.0000001789** |
| Quiet900, extra2 | 12 | 100.02 s | 12 | 0.0000007735 |

* A batch is possible only for **adjacent equal-length segments**; the approximation counts groups at `TensorBatchSize=2`, not GPU dispatches inside the model.

Comparing fixed300 and quiet900 extra0: **18.42% fewer audio seconds forwarded to the models**, and 32 → 10 segments. Fixed900 alone already saves 16.7% of that workload; the separate global scheduler contributes another 2.04 seconds beyond fixed900. Quiet900 extra1 uses 99.18 seconds of model input (17.72% less than fixed300) and seeks far quieter boundaries than minimal-chunk scheduling. It has **more model invocations than fixed900 at batch size 2**, and can lose on latency despite lower input duration.

The cut-energy metric is computed on the original filtered source. It is **not** a proven measure of audible clicking in synthesized output. Synthesized voices can emit nonzero energy at a source-silent cut. Different segment lengths also change HuBERT context, F0, NSF source phase, RNG seeds and possibly output quality. No bitwise parity against fixed300 or Python is expected.

## Verification

- Rust 1.100 std-only `rustc --test` targeting the actual `chunk_plan.rs` plus `chunked.rs`: **15 PASS / 0 FAIL**. Includes exhaustive length and partial-frame cases, duplicated source, empty inputs, optional split geometry, determinism, uncovered boundary rejection, and **bitwise exact identity-resampler reconstruction** for both short and long WAVs.
- `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu,cuda --lib --bin tts-with-rvc-rs`: **PASS**. Rust 1.100 stable.
- The same explicit `cargo check` with `cpu,cuda,wgpu`: **PASS**. Compilation is not execution of GPU kernels.
- `cargo check ... --tests` without `flex`: **not a Stage90 failure**; an older unrelated `bench_stage30_conv2d_matrix.rs` uses `Device::flex()` without the `flex` feature. That obsolete benchmark was intentionally not modified.
- No new Stage90 GPU execution on RTX 4060, no end-to-end RVC WAV generated; performance beyond modeled input duration and full-WAV acoustic quality remain **UNVERIFIED**.

## Decision and further R&D

Retain fixed300 as the default. Treat quiet-splice as an A/B experiment, not an automatic improvement. First benchmark fixed300, fixed900 and quiet900+1 with identical checkpoint, FAISS index, device, precision, warm count and audio. Compare physical latency, synthesized PCM, frequency of audible seam artifacts, peak VRAM and batched invocation counts.

The next genuinely large step is **Python-compatible long-context inference without arbitrary ten-second model limits**. This requires handling streaming/global HuBERT attention, exact RMVPE timing, source phase and NSF decoder memory rather than just incrementing the max chunk constant. Stage83 online attention and Stage84 exact NSF tiling may contribute after their numeric/runtime gates close. Do not introduce a silent fallback to old chunk geometry, do not present source-energy improvement as a validated audio-quality improvement, and do not call reduced forwarded audio time a measured GPU speedup.
