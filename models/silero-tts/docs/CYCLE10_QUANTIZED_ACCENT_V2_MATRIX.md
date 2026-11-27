# Cycle 10 — native recurrent accentor and full V2 catalogue

Base: `96a470e34e479dc383b6c1452d07b7063d1344af`. Rust 1.100, offline Flex release. No new runtime dependencies; one executable, original checkpoint paths.

The package object reader accepts signed per-tensor affine QInt8 storage only for package reconstruction; the scale, zero point and signed values are retained. The ordinary checkpoint reader still rejects quantized storage. Packed linear and RNN tuple states are recognized only for their explicit class names. The typed recurrent accent importer validates four bidirectional layers, packed projections, embeddings and classifiers. The shared runtime is used by the early Russian V2 packages and `ru_v3`; other graphs retain their existing paths.

## Evidence

- All 16 V2 original checkpoints import. The fixed-mask PyTorch oracle covers 18 scenarios, 37 PCM comparisons and 37 isolated vocoder comparisons. All passed; maximum PCM absolute error 0.002215996 (`multi_v2`). Encoder and mel comparisons also passed. Both implementations reached the 1000-frame cap for the tested `gilles_v2` utterance; this is not evidence of normal EOS for that utterance.
- `ru_v3`: eight Russian texts, three sample rates, 24 PCM comparisons passed, maximum absolute error 0.000620246.
- Recurrent accentor: 20 selected accent/yo outputs agree. Internal logits do not yet match (maximum observed absolute error about 1.599); this is not exact quantized LSTM parity. The oracle and ignored diagnostic test are retained to investigate the discrepancy. No claim of general accent parity.
- IPC smoke: 44/45 in the initial batch because the `v5_ru.pt` path was missing; the remaining V5 smoke succeeded independently after restoring the official checkpoint path. All 45 entries therefore have successful real-checkpoint IPC evidence. Stochastic V2 output is checked for validity, not identical WAV hashes.

## Reproduction

Use the original weights matching `verified_models.json`. `tools/create_v2_parity_fixture.py` produces the developer-only oracle data and matched dropout masks; `tools/create_recurrent_accent_fixture.py` captures recurrent accent outputs and logits. Run the ignored `original_v2_fixed_mask_parity` and `official_recurrent_accentor_parity` tests with their fixture environment variables (see test source). `tools/verify_multilingual_matrix.py` and `tools/smoke_native_worker.py` exercise original checkpoint paths. Python is exclusively a test oracle, not a runtime dependency.

## Remaining

Exact recurrent accentor logits, V5 CIS no-stress JIT representation and 20 old standalone JIT catalogue entries. Windows build of this revision is not yet verified. No generic interpreter is introduced.
