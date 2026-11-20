# Dettato Burn port — cycle 06 (2026-11-20)

Base: GitHub `codex/dettato-burn-port-20261120` at `e337a83e606a891f5410c5c438f480b9ca75a9a2`. All committed paths are within `models/dettato`. Original Python implementation and released safetensors from the user's `dettato.zip` were used as references. No model weights are published.

## Main result: numerical checks now exist for all four stages

This cycle adds **actual full-network forward comparisons** for the two pitched models that lacked neural-output parity: Transcriber and Grouper. These fixtures were generated with the original `LaneNet` and `NoteNet` using the supplied checkpoints and deterministic synthetic audio, then checked with the Rust/Burn Flex backend.

| Stage / output | Input | Maximum absolute difference | Result |
| --- | --- | ---: | --- |
| Transcriber `onset` | 0.2 s audio, 10 frames, 88 keys | 6.98e-6 | PASS |
| Transcriber `frame` | Same | 3.26e-5 | PASS |
| Transcriber `offset` | Same | 7.2e-7 | PASS |
| Transcriber `onset2` | Same | 3.0e-7 | PASS |
| Transcriber `frame2` | Same | 4.75e-6 | PASS |
| Transcriber `offset2` | Same | 1.1e-7 | PASS |
| Grouper instrument probabilities (73 classes) | 2 s audio, four notes, 3 zero Transcriber taps | 1.00e-4 | PASS |
| Grouper embeddings (32 dimensions) | Same | 1.33e-4 | PASS |
| Grouper hidden states (176 dimensions) | Same | 1.765e-3 | PASS |
| Grouper keep logits | Same | 7.00e-4 | PASS |

Transcriber head threshold: `0.01`; Grouper threshold: `0.005`. Both tests are explicitly `#[ignore]`, since they need ~15 MB and ~3 MB released checkpoint files and the Transcriber Flex CPU execution is slow. They are automatically compiled and become runnable with `DETTATO_WEIGHTS`.

Previous cycles measured Evidence selected output logits (**4.77e-6**) and Drums selected outputs (worst **3.10e-4**). All four **neural stages now have partial output parity evidence**, not full-layer parity or validated end-to-end MIDI results.

## Compilation, ordinary tests and performance

- Linux x86_64; Rust **1.100.0**; Burn **0.22.1**; backend **Flex CPU**, dev/test profile; offline Cargo source overlay outside Git.
- Ordinary test suite: **52 PASS**, **0 FAIL**, **8 ignored** (including the two new released-weight parity tests).
- Measured PyTorch CPU on the 0.2 s Transcriber audio: **0.209 s**, with `torch.set_num_threads(2)`.
- Measured Burn/Flex CPU on same audio: **41.03 s** in the unoptimized test profile.
- A temporary per-layer trace confirmed that recurrent, attention and convolution blocks are executing, not hanging; most layers cost 1.2–1.8 s. This trace was removed from the published source.
- A trial that batched the GRU input projections was **reverted** because the same test failed to finish within a 75-second local limit. No unverified performance optimization is shipped.

**Do not extrapolate these numbers to release/CUDA.** No optimized release benchmark, eight-second Transcriber forward completion, full-song MIDI validation, CUDA or GPU parity was performed in this cycle. The current end-to-end pipeline is still unverified.

## How to reproduce

Use the supplied checkpoint directory with the original flat layout, e.g. `/weights/dettato/dettato.safetensors` and the three accompanying stage checkpoints. The original name `detatto-evidence.safetensors` is supported.

```sh
cd models/dettato
export DETTATO_WEIGHTS=/weights/dettato
cargo test --no-default-features --features flex --test transcriber_forward_parity -- --ignored --nocapture
cargo test --no-default-features --features flex --test grouper_forward_parity -- --ignored --nocapture
```

The two reference fixtures are checked into `tests/fixtures/`; running the Rust tests requires neither Python nor a PyTorch environment. To regenerate the references, instantiate the *original* `dettato.transcriber.model.LaneNet` or `dettato.grouper.model.NoteNet`, load the released corresponding `safetensors`, evaluate the same deterministic input and write the full sigmoid/softmax heads to JSON. Both stages run under `eval()` and `torch.inference_mode()`. The Rust tests use matching deterministic sine-wave inputs.

The local offline build used a temporary overlay of the existing Cargo vendor. That overlay is **not** in Git; offline MP3/AAC codecs from the optional default feature set were not compiled during this cycle.

## Next priorities

1. **Functional:** complete an actual full-length Transcriber run and end-to-end song-to-MIDI test on adequate hardware, including two simultaneous notes at the same pitch and independent instrument tracks.
2. **Numerical:** extend four-stage parity from short selected outputs to individual intermediate layers, normal song-length windows, and full MIDI-event lists.
3. **Performance:** run a correctly optimized release build before drawing conclusions about Flex CPU speed; profile pointwise convolutions and recurrent kernels and then benchmark a CUDA backend.
4. **DSP:** compare the native resampler against upstream `soxr HQ` on real recorded audio; close optional codec and packaging verification.
