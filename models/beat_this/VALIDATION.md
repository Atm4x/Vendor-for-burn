# Inference validation — 2026-11-17

## Scope

The input was the user's FLAC (`8-MattYeux-Sometimes-9C3LFD.flac`). The two supplied native Lightning checkpoints were read directly using `pytorch-reader`. PyTorch references were calculated from the original Beat This! Python architecture with a local implementation of the equivalent interleaved rotary embedding operations and the same loaded checkpoint. Comparisons used the *raw frame logits* before peak picking, with `float32` tensors on CPU. The validated paths do not require GPU.

| Check | Model | Frames / input | Beat logits MAE | Downbeat logits MAE | Max absolute difference across logits | Result |
|---|---|---|---:|---:|---:|---|
| Neural model parity with identical input log-mel | `small1.ckpt` | 101 | 2.30e-6 | 4.67e-6 | 1.24e-5 | PASS |
| Neural model parity with identical input log-mel | `final1.ckpt` | 101 | 3.84e-6 | 4.70e-6 | 1.26e-5 | PASS |
| End-to-end PCM → SoXR → log-mel → model | `small1.ckpt` | 101 (2 seconds) | 5.89e-6 | 4.67e-6 | 2.46e-5 | PASS |
| End-to-end PCM → SoXR → log-mel → model | `small1.ckpt` | 301 (6 seconds) | 5.89e-6 | 4.24e-6 | 3.28e-5 | PASS |
| End-to-end full-file PCM → SoXR → log-mel → model | `final1.ckpt` | 301 (first 6 seconds) | 5.99e-6 | 8.57e-6 | 5.44e-5 | PASS |
| Entire 209-second FLAC, all overlapping windows | `small1.ckpt` | 10,458 | 1.09e-5 | 7.52e-6 | 1.66e-4 | PASS |
| Entire 209-second FLAC, all overlapping windows | `final1.ckpt` | 10,458 | 1.08e-5 | 7.92e-6 | 1.48e-4 | PASS |

The 2-second front-end-only log-mel MAE was about 2.95e-6 (maximum 6.3e-5). The CPU implementation took approximately 6.4 seconds for `small1` 101-frame neural inference and 18.2 seconds for `final1` 101-frame neural inference, excluding weight loading/audio processing. These numbers are diagnostic, not optimized benchmark claims.

The 6-second small model detected 9 beats and 4 downbeats. A release build (Rust 1.100.0, default Flex CPU backend) successfully analyzed the **entire 209-second source FLAC** for both provided checkpoints. `small1` produced 290 beats/135 downbeats in 43.88 seconds and `final1` produced 290 beats/124 downbeats in 68.04 seconds. The original Python postprocessor gave identical beat and downbeat counts for both checkpoints. These CPU runtimes were observed under concurrent local compilation and are **not controlled throughput benchmarks**.

The original 6-second Python reference cut the audio before resampling, while the CLI resampled the complete source and truncated *afterward*, leading to different boundary conditions in the last 2 Mel frames and large downstream logit differences due to noncausal attention. Recomputing the Python reference with the **same full-file resample followed by the 6-second truncation** restored end-to-end `final1` parity (MAE: beats 5.99e-6; downbeats 8.57e-6). The log-mel frontend MAE was 3.55e-6 in this comparison. Identical-input model-only 301-frame parity was better than 4e-6 for both heads.

The full-track original Python forward pass used the same `split_predict_aggregate` logic and matching full-clip resampling. The entire Rust pipeline matched its 10,458 output logits on both checkpoints with no non-finite output. The release binary was verified independently from the debug build.

Postprocessor regression tests now cover adjacent equal-valued maxima, including subframe running-mean positions identical to the upstream implementation. The synthetic chunk/aggregation tests check coverage up to 10,000 frames, including 1,500+, 3,000, and 10,000 frames. In addition to those tests, both models now complete a real 10,458-frame end-to-end inference.

## Limitations

- CUDA `cargo check --offline --no-default-features --features cuda` and `cargo build --offline --no-default-features --features cuda` **passed** (including final linking) after installing the previously supplied `linux-x64.tar.xz` (Tracel LLVM 23.1.0-4) into the local toolchain cache. A CUDA-capable binary still requires a compatible NVIDIA CUDA driver/runtime, and **GPU execution and Nsight profiling remain unverified**. No claims are made about GPU parity or real-time latency.
- SoXR is a dynamically loaded system runtime dependency. Fallback Rubato resampling is **not numerically equivalent** to SoXR and is not included in the parity results above.
- The available comparisons validate both supplied checkpoints on the entire track, but do not certify arbitrary other checkpoints or the optional DBN/madmom postprocessing.
- Local development started from Git Bridge snapshot `e41ce4b2b1d9c5c89b8c4f5cdef7e059817a03bc`; the isolated `feature/beat-this-rust-inference` branch was published through the GitHub Connector on top of a newer experimental-branch base. Other branches were not modified.

## Reproduction

In the checked-out repository, with the same checkpoint(s) and FLAC accessible:

```sh
cd models/beat_this
cargo test --offline
cargo build --offline --release
BEAT_THIS_RESAMPLER=soxr ./target/release/beat-this-rs \
  --checkpoint /path/to/small1.ckpt --input /path/to/sample.flac \
  --limit-seconds 6 --output /tmp/sample.beats --logits-json /tmp/sample.json
```

For finer numerical comparisons, compute an equivalent `[frames][128]` reference log-mel matrix in Python and feed it through `--spect-json` to isolate the neural network from the DSP frontend.
