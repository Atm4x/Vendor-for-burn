# Fish Speech 1.5 — CONT28: close source-grounded F01, connect configurable F08 reference decoder

## Baseline

- Base local Git tree: `e4a2412d0c575fa9a0ff4b60660f271352ed3007`, already checked against remote `codex/fish-speech` HEAD. Existing local Git history is complete; **do not create a new Git Bridge export**.
- Source of truth is the user-uploaded `Fish-speech-pipeline-main (1).zip`, including `config.py` (one Firefly 1.5 layout) and `llama.py` (`BaseModelArgs`, `DualARModelArgs`, `NaiveModelArgs`).

## Changes

1. F01: `model_config.rs` now rejects unknown Semantic config fields and unsupported reward-model configurations instead of ignoring semantics-changing inputs. All upstream dataclass metadata fields are explicitly accepted. Naive configs cannot masquerade as DualAR by carrying Fast-only fields.
2. F01: Slow `head_dim` follows Python `__post_init__`: derive from `dim/n_head`, even when the serialized value is stale. Fast `head_dim` follows Python's default-to-Slow behavior and fails closed for inconsistent unsupported geometries; explicit compatible Fast overrides are accepted.
3. F01: Firefly architecture validation is performed before loading the large weight shards, while manifest-config SHA-256 verification still happens before constructing a runnable model. Prior 450/450 original Python↔Rust weight shape agreement and checkpoint alias/weight-normalization tests are retained.
4. F08: `CpuFireflyReferenceEncoder` now owns its configurable `ReferenceAudioLoader`. The public `CpuFishModel::set_reference_audio_loader` injects one and clears the reference cache on changes. MP3/AAC/M4A/WMA become usable in the reference encoding path when an explicit local FFmpeg decoder is supplied; no global FFmpeg activation and no Python inference requirement. Added a test confirming an MP3 reaches the encoder after FFmpeg configuration; without configuration, unsupported MP3 fails at the decode boundary.
5. Added `CONT28_FUNCTIONALITY_MATRIX.md`: concrete acceptance checks for all six remaining PARTIAL components, separate operability gates and two progress meters.

## Verification

- Main Rust crate: **53 PASS, 2 ignored**, including new upstream semantic configuration and configured MP3/Firefly encoder tests.
- Full Core suite: **225 PASS, 2 ignored** across unit/integration tests.
- Python checkpoint exporter and Firefly config normalizer: **9 PASS**.
- Rustfmt: PASS; `git diff --check`: PASS.
- An additional Linux CUDA+TensorRT feature test build did not complete its dependency compilation within the environment time limit. It does not receive a PASS and does not reduce the functional score.
- Authentic RTX inference and numeric output parity: NOT TESTED.

## Progress

The previous 84.1% plateau used the coarse `DONE=1, PARTIAL=0.5` rubric. F01 is now DONE **for the concrete source provided**, not for unspecified variants, resulting in **16 DONE / 6 PARTIAL = 86.36%**. The remaining-component acceptance gates count **22/36 = 61.11%** and will advance for smaller functional milestones without falsely marking entire items DONE. See `CONT28_FUNCTIONALITY_MATRIX.md` for every remaining operation.

## Next work strictly in F order

F08: obtain/vendor the missing MP3/AAC/MP4 codec implementations and a WMA/ASF decoder or a licensed offline media backend; wire it through the already shared `ReferenceAudioLoader` without a subprocess dependency. Then F11: native production model facade, followed by F12 dynamic decode, F13 on-device sampling, F14 Naive native backend and F22 memory/ownership.
