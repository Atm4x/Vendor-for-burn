# Fish Speech 1.5 Rust CONT13 — behavior-first continuation

Date: 2026-11-18. Base: CONT12 source-only ZIP, original Fish-speech-pipeline-main(1).zip, Git Bridge offline vendor, Rust 1.100.0. All work local; no GitHub updates or changes to legacy Python sources.

## Requested progress-tracking correction

The former ~47% mixed implementation with GPU/Windows validation. It is **retired**. `CONT13_FUNCTIONALITY_MATRIX.md` explicitly identifies 22 original behaviors and separate V01–V10 verification gates. Functional checklist: **7 closed / 11 partial / 4 missing**, weighted bookkeeping **~57%** with equal weights and 0.5 for partial. This is not a claim that a model can run or produce speech. Verification: **5/10 gates** have local evidence, **0 GPU/e2e tests**.

## Implemented this cycle

1. `core/src/text.rs`: exact cleaning of Python source's quote symbols, specified emoji Unicode ranges, repeated commas, `split_text()` with float protection, stages `.!?。！？ → ,， → space → UTF-8 bytes`, and final punctuation-only fragment filtering. Generated a 144-case golden corpus directly from the supplied Python `clean.py` and `spliter.py`. Tested under Rust; 144/144 cases match.
2. `core/src/prompt.rs`: `encode_tokens()` reproducing original user/assistant voice conventions with and without reference audio codes, VQ part semantics, `LongPromptPlanner` holding successive input/output segments, reference examples and context rotation. It sends to the acoustic decoder only generated acoustic columns **after** the initial column (`generated[1: , 1:]`), while retaining the entire generated block in context.
3. `core/src/synthesis.rs`: generic but executable orchestration over typed `SemanticGenerator` and `AcousticDecoder` interfaces. It supports multiple samples, segment-by-segment generation, PCM concatenation at 44.1 kHz, cancellation between chunks, empty text rejection and finite output validation. All tests use fake generator and decoder; no real speech is claimed.
4. `src/fish_tokenizer.rs`: production-facing `FishBpeTokenizer` type that parses `.tiktoken` base64/rank lines, validates byte-vocabulary and unique contiguous ranks, declares all 1036 Fish-specific special tokens in original order, runs original Fish BPE regex with `tiktoken-rs 0.12.1`, accepts 400k-char Python text windows and implements the Core `TokenEncoder` interface. Synthetic-vocabulary tests cover multi-byte UTF-8, merged `hello`, voice/semantic IDs and invalid input. No real checkpoint `tokenizer.tiktoken` is present.
5. Added the already-vendored `base64 0.22` as a direct outer dependency; no new network access.

## Checks performed

- `cargo test --manifest-path core/Cargo.toml --locked --offline`: 169 unit + 28 integration = **197/197 PASS**.
- `cargo test --locked --offline` outer crate: **5/5 PASS**.
- `python -m unittest discover -s tools -p 'test_*.py'`: **5/5 PASS**.
- `cargo fmt --all --check`: **PASS**.
- `cargo check --locked --offline --features tensorrt-api`: **PASS** with vendored TensorRT bindings; no SDK/GPU execution verified.
- Original Python split function generated **144** golden cases; native Rust test passes all.
- `tensorrt-mock` not re-run in this cycle.

## Limitations and code debt

- `LongPromptPlanner` deliberately preserves original Python's unusual context selection, including the `-0` slice behavior under short thresholds. Retain until golden tests or a user-approved algorithmic fix; do not silently redesign source behavior.
- The Python source uses `\d` Unicode decimal digit; Rust `char::is_numeric()` is a superset for some non-decimal numeral characters. This narrow normalization discrepancy is still a possible remaining parity edge case; UTF-8/Russian/CJK and supplied regression cases match.
- Runtime cancellation is checked between chunks only. Actual one-token/one-codebook cancellation requires backend support.
- `FishBpeTokenizer` has not been tested with authentic `tokenizer.tiktoken` or full checkpoint; synthetic BPE is not proof of tokenizer parity on true vocabulary.
- No real semantic generator or HiFiGAN decoder implements the new trait interfaces. Full TTS still does not run.
- Reference audio file decoding/resampling and NaiveTransformer runtime are not implemented.

## Handoff

Continue from this source-only archive, first validate native reference WAV/audio input, mono/resampler and connect it to existing CPU LogMel. Then focus on actual Slow/Fast AR generation and binding the backend traits to weighted model execution. Do not count missing CUDA/Windows test results in the functional completion percentage. Keep binary/testing/engine/Windows acceptance in a separate progress ledger.
