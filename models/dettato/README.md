# Dettato — native Rust / Burn port

**Status: partial implementation. The transcriber network is implemented in Rust/Burn but is not yet numerically verified or tested with real model weights. Complete multi-instrument audio-to-MIDI inference is not available.**

The original Dettato project is by Nathan Mills / kobimusic (Apache-2.0). Original source: <https://github.com/kobimusic/dettato>. See `LICENSE` and `NOTICE`.

The crate is independent of `models/beat_this`; it has its own manifest and workspace.

## Delivered in the first development cycle

- Original four-stage directory layout and independent local paths for `config.json` / `model.safetensors` (`dettato`, `dettato-evidence`, `dettato-grouper`, `dettato-drums`). No download or Python required.
- Safetensors reader for F32, F64, F16 and BF16, preserving named tensor shapes and loading them to Burn tensors.
- Transcriber HCQT frontend for 16 kHz mono input: 2048-point Hann/STFT, 3 bins/semitone, eight harmonic channels, 512-point short-window channel, zero-centered padding and 10 ms hop. CPU frontend uses RustFFT; it is not CUDA optimized.
- Full structural Burn transcriber network: `Stem`, sparse `PitchMix`, dilated `LaneBlock`, pitch/time axial attention with RoPE and relative bias, bidirectional GRU, refine blocks and six output heads. Reference GRU implementation is frame-sequential and not optimized for GPU.
- Overlapping 8 s windows (2 s hop), Hann-weighted six-map aggregation and two-voice peak decoder with interpolation, NMS, offset/frame termination and confidence. This early runner does not aggregate intermediate hidden/stem taps for the later models.
- Multi-track Standard MIDI File (type 1) serialization with instrument programs, drum channel 10, velocities and independent MIDI ports for more than fifteen melodic tracks.

## Not implemented yet

- Transcriber numerical parity with PyTorch, proper BF16 CUDA GRU execution, speed/memory optimization and collection of overlap-averaged intermediate hidden/stem taps (mandatory for Evidence and Grouper).
- Evidence transformer and insertion/repair algorithm.
- Grouper front, instrument embeddings, clustering, naming, and `merge=false` track separation.
- Drum expert models / routing, tuning correction and resampling.
- End-to-end model inference, checkpoint parity and GPU benchmark.

## Weights layout

```text
weights/
  dettato/config.json
  dettato/model.safetensors
  dettato-evidence/config.json
  dettato-evidence/model.safetensors
  dettato-grouper/config.json
  dettato-grouper/model.safetensors
  dettato-drums/config.json
  dettato-drums/model.safetensors
```

## Validation

`cargo test --manifest-path models/dettato/Cargo.toml` from the repository root, with Rust 1.100.0 and dependencies available. The container that authored this cycle did not have `cargo`/`rustc`, so tests have **not** been run; this is a source-only first pass.

The next cycle must compare HCQT and decoding with the supplied original Python sources, then implement the full network under parity fixtures. Functional coverage and runtime verification are tracked separately.
