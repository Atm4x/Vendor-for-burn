# Dettato Rust / Burn — Cycle 04 (2026-11-20)

## Source of truth

- Base repository: `Atm4x/Vendor-for-burn`, branch `codex/dettato-burn-port-20261120`.
- Local source tree before this cycle: `13a6059d90ad2961c0b28669df5f71e8a2eaed90`, identical to the remote branch tree at commit `9f5d4b47277ade7f55301d0fbc8829e952ae32a3`.
- The user's `dettato.zip` contains all four complete released checkpoints. The Evidence files are named `detatto-evidence-*`; they are recognized without renaming. Model files are **not committed**.

## Work performed

1. **Drums frontend** (`drums/features.rs`): RMS -18 dBFS normalization; 128 × low-mel / 32 × high-mel / 120 × log-frequency bands; centered Hann STFT at 512 / 1411 / 6144 FFT points, 100 fps, and FP16 round-trip matching the original feature dtype. The required 44.1 → 16 kHz conversion uses native windowed sinc, not libsoxr.
2. **DrumNet on Burn** (`drums/net.rs`): residual separable 2D stem with BatchNorm; band-direction bidirectional GRU, temporal depthwise convolution, recurrent and full-song content attention; five GRU-based percussion family towers; 61-key onset and velocity heads. Long song embeddings use overlapped chunks.
3. **OrchestralExpert on Burn**: 120-band CQ pitch branch, three-channel overtone shifts, GroupNorm, two-layer bidirectional GRU, content attention adapter, 61 drum corrections, 13 timpani logits and a song-level router.
4. **Three-expert orchestral ensemble** (`drums/pipeline.rs`): load one drum-kit model and three independent expert checkpoints; average *probabilities* and routers, preserving the upstream threshold selection and drum decoder. Released config and source-rate constraints are checked.
5. **Tuning normalization** (`retune.rs`): native 8192 FFT, 1024 hop, harmonic salience (six harmonics), five-cent candidate grid, circular weighted mean, minimum 10-cent / 0.12-strength retune gates, and restoration of original timestamps.
6. **End-to-end API** (`pipeline.rs`): mono PCM and direct file transcription through all four stages, including empty pitched note handling, 40 kHz minimum original sample rate for drums and the optional preservation of independent same-instrument tracks (`set_merge_same_instrument(false)`). `audio.rs` decodes common formats via Symphonia at the original sample rate.
7. **Checkpoint discovery and regression tests**: the native loader resolves canonical Hugging Face subdirectories and the supplied flat artifact layout with the `detatto` misspelling. Tests/fixtures were created for drums spectral features, tuning, ensemble probabilities, and optional complete weight initialization.

## Validation actually performed

- `dettato.zip` extracted and verified with the original PyTorch classes, **zero missing or unexpected keys** for: Transcriber (337 tensors, 15,285,216 parameters), Evidence (86 tensors, 4,930,929 parameters), Grouper (202 tensors, 2,896,613 parameters), drum-kit (269 state entries, 188,764 parameters) and the three orchestral experts (315 state entries and 1,606,407 parameters each).
- Original PyTorch inference on a deterministic 220 ms audio test: 22 × 280 front-end features; 11 × 61 kit/orchestral onset probabilities, 11 × 13 timpani probabilities; orchestral router mean **0.9192254742**. Reference fixtures are committed for later comparison with Burn.
- Original Python tuning benchmark on 2 s synthetic harmonic tones: 440 Hz → -0.5163 cents (no shift), 447 Hz → +27.0406 cents, 432 Hz → -32.3098 cents. Native regression expectations were added.
- Local Git whitespace/diff audit: PASS.

**Not verified:** compiling this Rust crate, running Rust tests, loading the checkpoints through Burn, matching any Burn layer outputs against PyTorch or GPU performance. This local container has no `cargo`/`rustc` toolchain and no outbound DNS. No Rust numeric-parity or functionality/runtime percentage is claimed as verified by those missing tests.

## Known fidelity and performance differences

- The standalone native resampler is windowed-sinc, **not soxr HQ**, and numeric deviations are possible. Frontend parity fixtures intentionally allow an initial tolerance while a strict numerical comparison is pending.
- The orchestrator gathers Transcriber taps in separate bounded-memory passes, instead of the upstream fused single pass. This favors completeness and bounded buffers, at the cost of repeated compute.
- The GRU and attention implementation uses small Burn tensor operations rather than fused kernels; expect poor performance until numerical parity is established.
- Audio file support covers common Symphonia codecs; the original FFmpeg fallback can additionally handle Opus and other formats. Direct PCM input remains supported for any codec handled by the caller.
- Recurrent/output tensors are presently all FP32; upstream CUDA BF16 autocast and exact rounding are still to be audited.

## Next priority

1. Obtain Rust 1.100+, perform `cargo fmt`, `cargo test`, and fix compile/runtime issues in `models/dettato` **without changing Beat This**.
2. Run ignored `released_checkpoint_smoke` tests against unpacked `dettato.zip`, then compare Drum frontend and ensemble fixtures. Validate every checkpoint tensor consumed by Burn.
3. Run per-stage PyTorch–Burn numeric parity in Transcriber, Evidence, Grouper, Drums (in that order), and tighten tolerances.
4. Only after functional and numeric verification, optimize repeated GRU operations, batching and long-song memory/GPU execution.
