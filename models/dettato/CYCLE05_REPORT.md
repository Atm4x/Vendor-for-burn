# Dettato Burn port — cycle 05 (2026-11-20)

Base: `codex/dettato-burn-port-20261120`, GitHub commit `11eb74f673f1d1f71bc70bba9e94044e649b5978` (tree `12e8045f8a2419365ec47806d69503c7c8b177c1`). All changes remain under `models/dettato`.

## Functional coverage

All four neural architectures were introduced by cycles 01–04. This cycle makes that code verifiable on Rust 1.100.0 / Burn 0.22.1, fixes compilation errors, validates real model loading, and adds independent model forward tests. It does **not** claim complete end-to-end transcription correctness.

- Fixed missing Drums constant reexports, Grouper slice and inference type mismatches, and an ownership error in the orchestral CQ frontend.
- Enabled an offline Flex build without the optional Symphonia MP3/AAC/ISOBMFF/ALAC codecs. Standard defaults retain those codecs via `extended-audio`. The official vendor snapshot does not contain those optional codecs' manifests, so offline Cargo resolution still needs an external overlay; see below.
- Corrected the downsampler's truncated sinc kernel normalization: outside-recording samples must contribute zero without renormalizing the impulse response at the edges. Added a regression test.
- Aligned Evidence `_box_argmax`, mel filterbank and FP16 floor tests with observed Python/PyTorch semantics; added real-weight integration tests for Evidence, Grouper, Transcriber and Drums.

## Local test results

Test environment: Linux x86_64, Rust 1.100.0, Burn 0.22.1, Flex CPU, `--offline --no-default-features --features flex`.

- `cargo check`: **PASS**.
- `cargo test`: **52 PASS, 0 FAIL**, plus 6 ignored tests requiring released weights (four model forwards, pipeline initialization, and Evidence numerical parity). Tests include HCQT, Evidence feature quantization, PCEN, HDemix, Drums audio features, instrument grouping, MIDI export, retuning and resampling.
- All four released checkpoints initialize `Dettato::from_directory`: **PASS**, using the provided `dettato.zip`. The loader supports the flat `detatto-evidence` filename.
- Evidence forward: **PASS** with real checkpoint.
- Grouper NoteNet forward: **PASS** with real checkpoint.
- Drums full ensemble forward: **PASS** with real checkpoint (11-frame PyTorch reference).
- Transcriber 8-second `LaneNet::infer_features` on Flex CPU: **NOT CONFIRMED**. Its ignored smoke test exceeded the 105-second execution limit. No success result can be claimed.

## Numerical agreement

Numerical comparisons are between the original PyTorch model and Burn/Flex using the provided released checkpoints; they are **not** full-layer or GPU-wide parity proofs.

| Component / reference | Result | Worst absolute error |
| --- | --- | ---: |
| Evidence Repairer — selected output logits, two nontrivial synthetic notes | PASS | **4.77e-6** |
| Drums router — released model, synthetic signal | PASS | **3.10e-4** |
| Drums kit logits — selected points | PASS | **4.4e-5** |
| Drums kit velocity — selected points | PASS | **1.12e-4** |
| Drums orchestral logits — selected points | PASS | **2.3e-5** |
| Drums orchestral velocity — selected points | PASS | **1.00e-4** |
| Grouper PCEN / HDemix — fixed fixtures | PASS | See integration test fixtures |
| Transcriber learned layers | Not measured | — |
| Grouper learned layers, vs PyTorch | Not measured | — |
| Full song MIDI output | Not measured | — |

The Evidence fixture was generated from the original Python `Repairer` loaded with `detatto-evidence.safetensors`. Both notes use nonzero, deterministic 769-dimensional evidence inputs. Test threshold is 0.001. The Drums regression threshold was tightened to 0.005; actual selected values are substantially closer.

## Reproducing locally

Rust toolchain and external source overlay used in this cycle:

```sh
cd models/dettato
cargo test --offline --no-default-features --features flex \
  --config 'source.vendored-sources.directory="/mnt/data/dettato-vendor-overlay"'

DETTATO_WEIGHTS=/path/to/unpacked/dettato \
  cargo test --offline --no-default-features --features flex \
  --config 'source.vendored-sources.directory="/mnt/data/dettato-vendor-overlay"' \
  --test evidence_forward_parity -- --ignored --nocapture
```

The vendor overlay is **outside the Git repository** and is not shipped in this commit. It contains metadata-only stand-ins for missing *disabled* optional audio codecs and corrected local checksum metadata for 25 existing source packages. The original vendor directory and other models were not modified. A standard online Cargo environment with all dependencies available does not require this local workaround.

## Next priorities

1. Execute Transcriber forward successfully (prefer Release/Flex or GPU) and compare outputs against PyTorch using matched inputs and actual checkpoint.
2. Obtain full Grouper NoteNet numerical parity, including the real-weight LaneFront, instrument/embedding/keep heads.
3. Exercise complete audio-to-multitrack-MIDI pipeline and test temporal consistency, instrument boundaries and duplicate notes.
4. Expand DSP parity, particularly soxr HQ versus the current deterministic windowed-sinc approximation; test real audio and standard MP3/AAC defaults, which were excluded from the offline build.
