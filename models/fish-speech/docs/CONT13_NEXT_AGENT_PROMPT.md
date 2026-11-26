# CONT14 handoff — Fish Speech 1.5 Rust

Use CONT13 source-only ZIP (local, no Github pull required). See `docs/CONT13_REPORT.md` and `docs/CONT13_FUNCTIONALITY_MATRIX.md`.

Track **two independent scales**. Functional: original Python behavior inventory F01–F22; DONE=1/PARTIAL=0.5/MISSING=0; currently 7 DONE, 11 PARTIAL, 4 MISSING (~57% ledger). Do not reduce this score because Windows/CUDA verification is unavailable. Operability: V01–V10 separate, 5 PASS, GPU/e2e gates unverified.

Last changes: full `clean_text()` and `split_text()` with 144 Python golden cases; reference/no-reference `encode_tokens` plus long-context planner; `synthesize()` abstraction supporting multiple samples and final PCM concatenation via mock generator/decoder; native `FishBpeTokenizer` parser/regex/specials via offline `tiktoken-rs` + base64. Tests: 197 Core, 5 outer, 5 Python exporter, fmt and tensorrt-api compilation PASS.

Next work on functionality, not yet another synthetic GPU gate:
1. Actual `ReferenceLoader.load_audio()` behavior: detect bytes/files WAV/MP3, mix down multichannel, resample to 44100 with quality aware of torchaudio parity, feed `ReferenceMelPreprocessor`.
2. Real `config.json` model-parameter map and weighted semantic generator (Slow prefill, Slow 1-token, Fast codebook loop, correct EOS and cache reset). Keep CUDA/SDK availability as a separate verification gate.
3. Attach a real Firefly decoder to `AcousticDecoder`, ensure correct acoustic index layout and output PCM, and close `TTSInferenceEngine` entrypoint.
4. `NaiveTransformer` alternative, reference-ID/hash cache, and streaming header/segment/final event API.

Do not edit obsolete Python. Don't run `dotnet` for Unity. Keep Rust 1.100.0 and pinned offline vendor. No real `tokenizer.tiktoken` or model weights were provided, therefore never claim authentic model parity.
