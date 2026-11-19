# Silero v5 RU + native RUAccent: real PCM integration (2026-11-19)

## Purpose and provenance

This test validates the actual `ruaccent_text` opt-in frontend, not just crate compilation. The existing `text` frontend remains SileroStress.

- Source commit: [1ec02d37](https://github.com/Atm4x/Vendor-for-burn/commit/1ec02d37cb66a0b56f001878d9f576a68784cc83).
- Rust toolchain: 1.100.0; `--offline --release --no-default-features --features reference-flex,ruaccent`.
- CPU: virtualized AMD EPYC, 4-thread limit; Flex backend, no GPU.
- Silero model: locally provided/exported v5 RU, with weights separate from sources.
- RUAccent: official `ruaccent/accentuator` `tiny`, revision `b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855`; ONNX Runtime 1.22.0 Linux CPU from Microsoft's NuGet package.
- The pinned licensed model pack and native ORT library were downloaded through `.github/workflows/ruaccent-tiny-offline-modelpack.yml`. Model files were not checked in.

## Found and fixed: case-dependent acoustic text preparation

Previously, the adapter lowercased and normalized the input **before** sending it through RUAccent. This changes the RUAccent classification:

| RUAccent input | Neural output |
|---|---|
| `На двери висит замок.` | `На дв+ери вис+ит зам+ок.` |
| `на двери висит замок.` | `на двер+и вис+ит зам+ок.` |

In the failing version the first input produced **13,400 PCM samples** (1.675 seconds), whereas the manually accented control produced **13,800 samples** (1.725 seconds) at 8 kHz.

The corrected sequence is:

1. Pass the unmodified long-form fragment to `RuAccent::process_all`, preserving case/context.
2. Normalize the resulting stressed string with Silero `ModelConfig::prepare_text` (including lowercasing and symbol-set filtering).
3. Feed the resulting `+`-accented string into the existing acoustic/vocoder path.

This is a targeted frontend correction; neither neural weights nor the SileroStress or vocoder implementation changed.

## Real release-binary checks

The worker was built from the corrected source, loaded both weight sets into one persistent process, and exercised through its public JSONL interface.

| Test | Result |
|---|---|
| Full Silero release unit test suite with features `reference-flex,ruaccent` and both model paths | **57/57 PASS** |
| Regression test on `На двери висит замок.` | **PASS on real weights** |
| RUAccent PCM vs explicit accented input, two distinct sentences (8 kHz) | **2/2 bitwise identical** |
| RUAccent synth, `baya` at 24 kHz | PASS |
| RUAccent synth, `eugene` at 48 kHz | PASS |
| 730-character RUAccent text within one worker request | **62.975 seconds of 8-kHz audio** |
| Long-form synthesis, 730 characters | **2.104 seconds measured worker inference** |
| Long-form RTF | **0.03341, approximately 29.9× real-time** |
| Original `text` with RUAccent loaded versus not loaded | **Bit-identical PCM** |
| `ruaccent_text` requested without `ruaccent_dir` | Explicit error, worker remains usable |
| Successful worker process shutdown | Exit status 0 |
| Silero Flex `cargo check --offline --release --no-default-features --features reference-flex` (without RUAccent) | PASS |

The first worker run loaded both models in approximately 5.42 seconds; load time is **excluded** from inference/RTF measurements. A second independent long-form run produced the same 62.975-second audio in **2.218 seconds** (RTF **0.03522**). These two observations are not a statistically robust latency benchmark. A same-session repeat of an RUAccent request was **bit-identical** to the first output. After rejecting an unavailable RUAccent provider, the worker continued to synthesize ordinary `text` with **bit-identical PCM**.

The regression test is model-gated: set `SILERO_RUACCENT_INTEGRATION_DIR` to the exported Silero weights and `RUACCENT_TINY_WORKDIR` to the RUAccent tiny pack when running `cargo test`. When these variables are omitted, the test returns early; a green no-weight test run must not be called evidence of real neural/PCM parity.

### Full RUAccent golden corpus through the Silero acoustic pipeline

A separate persistent-worker smoke test processed **all 20 captured golden inputs** from the RUAccent text corpus with actual Silero v5 RU and RUAccent tiny weights. **18/18 pronounceable Russian phrases** (including contextual homographs) returned nonempty, finite PCM; the two non-pronounceable cases (empty string and punctuation only) were rejected as expected. The worker remained usable afterwards and exited with code 0.

This is a corpus-level **TTS execution smoke test**, not PCM parity against PyTorch for all 18 phrases. The two earlier explicit-accent PCM comparisons above are the bitwise parity evidence. Reproduction script: `check_golden_corpus_pcm.py`; results: `golden-corpus-pcm-results.json` (both shipped in the evidence ZIP).

## Remaining release constraints

1. **Strict ONNX logits parity**: inherited two raw-logit tests still fail at `1e-4` tolerance with the pinned model (accent absolute difference ~0.00142, omograph ~0.00207). These tests are not relaxed; separate Python ONNX audit showed the current omograph weights disagree with the older golden fixture even in Python. Final categorical/string results remain correct for the tested corpus.
2. **Offline vendor integration**: a pinned source bundle with 93 missing Cargo crate directories was prepared separately. This local bundle enabled the integration tests but has **not** been merged into the repository's committed `vendor/` tree. Do not silently claim a clean offline source checkout is ready.
3. **Windows real PCM**: previous Windows CI passed native RUAccent inference and compiled Silero with its optional dependency, but did not run a *combined full Silero+RUAccent audio synthesis* check on Windows.
4. Streaming and in-flight cancellation are separate worker features still not implemented; the current worker returns the complete PCM buffer per request.

Evidence includes `verify_silero_ruaccent.py`, `tts-integration-result.json`, `silero-full-real-ort-tests.log`, `ruaccent-case-regression-test.log`, `silero-default-optout-check.log`, and the compiled model provenance (without weights or native binaries).
