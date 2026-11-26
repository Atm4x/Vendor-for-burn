# F5-TTS Rust — CONT12: native Jieba and pypinyin differential parity

Date: 2026-11-20. Based on published GitHub `codex/f5-tts-rust-cont11` at `a91d99b8a49902c28dddefa687c825bdf63c0945`. Changes are confined to `models/f5_tts`. No ONNX or `maryml` code was used.

## Functional changes

- Replaced the heuristic Han/ASCII boundary splitter in `F5TextFrontend::encode()` with a lazily constructed, process-wide `jieba-rs 0.11.0` segmenter (`cut(..., true)`) that preserves the original `rjieba.cut()` word boundaries in the tested corpus.
- Added a pinned `pypinyin 0.55.0` TONE3 character table (26,703 entries) and phrase table (47,111 entries) for native pronunciation. These are read by the existing model-specific `ChinesePinyin` provider, with Unihan kept as a fallback.
- Word pronunciations derived from `pypinyin(..., tone_sandhi=True)` are treated as already sandhi-processed, avoiding a second pass. Explicit user word overrides still take precedence; the old `from_tsv` path remains functional.
- Original F5-TTS token concatenation/punctuation handling is retained. Standard and Cross-Lingual synthesis share the same native frontend. No Python process is required at inference time.
- Added independent Python differential fixtures and a pinned generator under `tests/tools`, with source/asset license and version information in `assets/README_pinyin.md`.

## Numerical/functional parity evidence

The independent CI oracle uses `rjieba 0.2.2` and `pypinyin 0.55.0` in `Style.TONE3`, `tone_sandhi=True`. Inputs combine Han words, polyphones, Latin text, punctuation, and randomized phrase sequences. A separate frozen test fixture checks F5-TTS's exact normalized token sequence.

| Functional gate | Result |
| --- | ---: |
| `jieba-rs` vs `rjieba.cut` segmentation | **1,809/1,809 exact** |
| Native word pronunciation vs `pypinyin` | **5,859/5,859 exact** |
| End-to-end normalized F5-TTS token IDs vs Python | **1,809/1,809 exact** |
| Rust library unit tests | **46/46 PASS** |
| Standalone frontend regression suite | **35/35 PASS** |
| Existing full DiT/CFM/Vocos PCM PyTorch parity | **PASS**, two scenarios |
| WGPU and CUDA `cargo check` | **PASS**, compilation only |

The three language corpus gates cover finite samples and cannot be extrapolated to arbitrary Chinese context, user-injected dictionaries, or another version of `rjieba` / `pypinyin`. The existing full audio PyTorch gates use synthetic model weights; no pretrained checkpoint or GPU execution is claimed.

## Dependency reproducibility

The GitHub Actions runner [#44756323605](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44756323605) produced the language fixture archive and a complete Cargo vendor archive (both downloaded). The vendor artifact SHA-256 was verified: `b679f297598d3808fe90aa58a9b3606d6c248fb3a3fb47e066d1505117c7654f`.

The runner regenerated a Cargo lockfile from scratch and selected newer transitive CubeCL packages. To prevent unrelated changes to the verified inference stack, **the committed CONT12 Cargo.lock is instead derived incrementally from CONT11**: 16 added/updated dependency versions for `jieba-rs`, preserving Burn `0.22.1` and CubeCL `0.11.3`. The `vendor_additions.tar.zst` export carries the 16 additional source trees; merge them into the pre-existing CONT09 vendor directory before offline builds. Vendor is not committed to Git.

Reproduction uses Rust 1.100.0 with vendor and `--offline --locked`:

- `cargo test --test linguistic_parity -- --test-threads=1 --nocapture`
- `cargo test --test offline_frontend -- --test-threads=1`
- `cargo test --lib -- --test-threads=1`
- `cargo test --features cubecl-cpu --lib full_dit_cfm_vocos_pcm_matches_independent_pytorch -- --test-threads=1`
- `cargo test --features cubecl-cpu --lib full_dit_cfm_vocos_epss_sway_pcm_matches_pytorch -- --test-threads=1`
- `cargo check --features cubecl-wgpu` and `cargo check --features cubecl-cuda`

The original `F5-TTS-main` repository is used only as the mathematical/text-processing reference. Changes to its scripts and all other models are excluded.

## Functional gaps after CONT12

1. Native independent **TensorRT-RTX** execution backend for DiT, CFM, SpeedTransformer, and Vocos. The existing CubeCL path must remain intact. No ONNX conversion or dependency is permitted.
2. Production-width DiT numerical checkpoints, further full-architecture parity under different input lengths and CFG values. Work against synthetic, fixed weights first, not pretrained downloads.
3. Edge-case Chinese tone sandhi / custom phrases beyond the pinned rjieba/pypinyin corpus; add failing differential examples before changing algorithms.
4. Hardware workability (real GPU, Windows, pretrained checkpoints, performance) is a separate late-stage track.

**Progress:** functionality estimate **84%**, up from 78% because the native Chinese segmenter/pronunciation/tokenization now has comprehensive differential tests. Workability **43% (3/7)**, unchanged.
