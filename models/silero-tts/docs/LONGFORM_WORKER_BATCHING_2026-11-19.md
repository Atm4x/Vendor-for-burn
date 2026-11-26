# Silero V5 RU — automatic long-form inference

Date: 2026-11-19. Base: `59875b2d37eaf5787e694ecb2f5c739f7782e7c6` (Silero Git Bridge source export). Runtime: Rust 1.100.0 / Burn 0.22.1 / Flex release. Real source weights: `v5_ru.pt` exported to the expected SafeTensors layout.

## Problem and design

The decoder's positional embedding table supports 5,000 acoustic positions. The predicted expanded duration, not the character count, determines whether a request exceeds the model limit. An ~1,221-character single request used to fail with `Expanded Silero utterance exceeds acoustic positional encoding limit`.

`src/longform.rs` adds a bounded, model-agnostic scheduler shared by the `text` and `accented` entrypoints. It splits at punctuation/whitespace with a target of 400 Unicode characters; it refuses to split inside an unbroken word or an explicit accent marker. The worker retains a single loaded model and concatenates PCM from successful subrequests in order. If a segment still expands to more than 5,000 acoustic positions, it is divided again and retried; other errors are propagated without retry. For control-bearing SSML and `symbol_durations`, segments preserve rate, pitch and trailing breaks while being split/grouped, then produce concatenated PCM and duration arrays. Short requests continue through one ordinary inference, and no streaming claim is made.

Resource limits: 16,384 input characters, 256 queued/completed chunks and at most 600 seconds of generated audio. Unbreakable overlong tokens fail explicitly. The `capabilities` response advertises the long-form feature and states `streaming=false`; the worker protocol version is unchanged. Audio is returned as one buffer, so this does **not** implement incremental delivery or cancellation. Raw concatenation is intentionally bit-equivalent to manual successive segment synthesis, not a crossfade.

## Local tests

- `cargo test --offline --release --no-default-features --features reference-flex --lib`: **54 passed, 0 failed**. The new tests cover short/long text, Unicode splitting, refusing unsplittable words, retries only on the specific acoustic limit, SSML prosody/break retention and unrelated error propagation.
- Four short request modes (`text`, `accented`, `ssml`, `symbol_durations`) generated exactly the same samples and SHA-256 hashes as the previously published worker (`73463735` source lineage).
- One automatic 1,832-character / 122.125-second request and five manual previous-worker requests on the exact selected chunks (389, 389, 389, 389, 272 characters) generated **identical 977,000 float32 PCM samples**, SHA-256 `920b051c68456b706ddd7d457e530d27984491f28582a3842828d2c05a2befd4`.
- Longer SSML, accented and `symbol_durations` requests completed. An oversized unbroken token was rejected while the persistent worker remained available for the following request.
- Long samples at both 24 and 48 kHz completed: 77.8875 seconds of audio, 1,869,300 and 3,738,600 samples, respectively.

The real-model worker benchmark used a shared AMD EPYC 9V74 virtual CPU with a four-core cgroup quota. All timings are **single observations**, not statistically robust production medians. Worker time includes inference and inline Base64 encoding; model loading is excluded.

| Input characters | Generated seconds | Worker latency | RTF | Audio throughput |
|---:|---:|---:|---:|---:|
| 610 | 40.7625 | 1.196 s | 0.02934 | 34.1× real-time |
| 1,221 | 81.575 | 2.328 s | 0.02854 | 35.0× real-time |
| 1,832 | 122.125 | 3.518 s | 0.02881 | 34.7× real-time |

Comparisons with source PyTorch across *all* possible long strings are **not** claimed; the deterministic old-worker segmentation parity and earlier short-form PyTorch validations are distinct checks. The reported long-input performance depends on voice, language content, CPU and chunk schedule.

## Release boundaries and future work

The fix removes the observed 5,000-position failure for sentence-separable long input and remains backward-compatible with short requests. It does not support uninterrupted inference of one very long unsplittable word, streaming audio backpressure, prompt-level cancellation, or crossfaded chunk boundaries. Those are separately scoped improvements before a full production release, especially for live TTS.

Neither Burn/CubeCL source nor third-party license assets were modified. Restricted-container vendored `.cargo-checksum.json` repairs remain local and are not part of the change.
