# Inference validation — 2026-11-17

## Scope

The input was the user's FLAC (`8-MattYeux-Sometimes-9C3LFD.flac`). The two supplied native Lightning checkpoints were read directly using `pytorch-reader`. PyTorch references were calculated from the original Beat This! Python architecture with a local implementation of the equivalent interleaved rotary embedding operations and the same loaded checkpoint. Comparisons used the *raw frame logits* before peak picking, with `float32` tensors on CPU. The validated paths do not require GPU.

| Check | Model | Frames / input | Beat logits MAE | Downbeat logits MAE | Max absolute difference across logits | Result |
|---|---|---|---:|---:|---:|---|
| Neural model parity with identical input log-mel | `small1.ckpt` | 101 | 2.30e-6 | 4.67e-6 | 1.24e-5 | PASS |
| Neural model parity with identical input log-mel | `final1.ckpt` | 101 | 3.84e-6 | 4.70e-6 | 1.26e-5 | PASS |
| End-to-end PCM → SoXR → log-mel → model | `small1.ckpt` | 101 (2 seconds) | 5.89e-6 | 4.67e-6 | 2.46e-5 | PASS |
| End-to-end PCM → SoXR → log-mel → model | `small1.ckpt` | 301 (6 seconds) | 5.89e-6 | 4.24e-6 | 3.28e-5 | PASS |

The 2-second front-end-only log-mel MAE was about 2.95e-6 (maximum 6.3e-5). The CPU implementation took approximately 6.4 seconds for `small1` 101-frame neural inference and 18.2 seconds for `final1` 101-frame neural inference, excluding weight loading/audio processing. These numbers are diagnostic, not optimized benchmark claims.

The 6-second small model detected 9 beats and 4 downbeats. `final1` completed the 101-frame check but **did not complete** the 301-frame CPU check within the available execution timeout. No claim is made about long-form `final1` throughput or accuracy.

The synthetic chunk/aggregation tests check coverage up to 10,000 frames, including 1,500+, 3,000, and 10,000 frames, but do not exercise long-form neural inference on the 209-second original song.

## Limitations

- CUDA compilation and Nsight profiling **not completed** in the local offline CPU sandbox: `tracel-llvm-bundler` tried to fetch an LLVM archive despite offline mode. No NVIDIA device was available. Hence no claims about GPU parity, NCT/NTC transport throughput, or real-time latency.
- SoXR is a dynamically loaded system runtime dependency. Fallback Rubato resampling is **not numerically equivalent** to SoXR and is not included in the parity results above.
- The available comparisons verify the inferred logits and minimal postprocessor for short clips. This is not a certification of all potential checkpoints or the optional DBN/madmom postprocessing.
- The target Git branch for local development was restored from a Git Bridge bundle at `e41ce4b2b1d9c5c89b8c4f5cdef7e059817a03bc`. The remote experimental branch had advanced to `6609b56efa722d85c53797306b728cbbec37978a`; no remote synchronization or push is performed by this work.

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
