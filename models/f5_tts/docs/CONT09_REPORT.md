# F5-TTS Native — CONT09

Date: 2026-11-20. Base: locally restored CONT08, published GitHub commit `4bf0a7cc4fb4609cf145b8be1f57fb0b0184b961`. Modifications limited to `models/f5_tts`. The supplied `maryml-main.zip` was not opened. ONNX not used. No changes to Fish Speech, RVC, Legacy, or the rest of the repository.

## Functional scope and accuracy

- F5 v1 complete reduced DiT numerical path now actually executed against an independently generated PyTorch fixture, on Burn 0.22.1 / CubeCL CPU / LLVM 23.1.0. Fixed identical safetensors weights, conditional/unconditional text and input embeddings, 2 DiT blocks, CFG output; all checked tensors satisfy `max_abs <= 2e-4`. **PASS 1/1**. This does **not** establish parity for the pretrained 22-block checkpoint or actual speech output.
- Cross-Lingual counting and sentence chunking previously tested against 2,521 Python reference cases. Fixed punctuation accumulation to use `f64` and Python-compatible ties-even rounding.
- Added 2,200 independent Python float64 timing cases, spanning predicted-speed classes (0.25–8.0 units/sec), input speeds, reference sizes, and short text adjustments. Corrected scheduling arithmetic and public Cross-Lingual user settings (`speed`, `fixed_seconds`) to `f64` to prevent off-by-one frame errors.
- Added 128 NumPy crossfade references covering different numbers and lengths of segments and fade windows. Changed fade-duration configuration to `f64` and crossfade weights to `f64` intermediate arithmetic before producing `f32` audio. This resolves off-by-one overlap lengths caused by representing user-supplied duration as `f32`.
- Cross-Lingual functions retain validation of invalid/out-of-range durations; the independent duration corpus uses valid-inference ranges.
- Replaced a JSON test dependency in the existing US English hyphenation oracle with a 552-word TSV Python reference. The runtime is unchanged.

## Cargo vendor via GitHub Actions

- CI runner [44741264056](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44741264056), **success**; artifact ID `14631086557`, name `f5-tts-cargo-vendor-cont08` (~103 MB ZIP). The package includes `f5-tts-cargo-vendor.tar.zst`, `Cargo.lock`, `.cargo/config.toml` and source commit marker.
- Pinned source commit: `4bf0a7cc4fb4609cf145b8be1f57fb0b0184b961`. Artifact SHA-256 verification **PASS**; `vendor/` extracted locally (~867 MB). `Cargo.lock` is included in the F5 source tree for reproducible offline resolution; **the 867 MB vendor source tree is not added to Git history**.
- Rust: 1.100.0; Burn: 0.22.1; CubeCL: 0.11.3; LLVM: 23.1.0-4. Used supplied `linux-x64.tar.xz` and installed LLVM to the exact path expected by the `tracel-llvm-bundler` package, with its sentinel. No online dependencies in the local compilation.

## Executed results

- `rustc --edition 2024 --test tests/offline_frontend.rs --test-threads=1`: **35/35 PASS** (including 2,521 Cross-Lingual text tests, 2,200 timing cases, 128 NumPy fade cases, previous CFM/pinyin/standard text tests).
- `cargo test --offline --locked --features cubecl-cpu --lib cross_lingual::tests -- --test-threads=1`: **7/7 PASS**.
- `cargo test --offline --locked --features cubecl-cpu --test offline_frontend -- --test-threads=1`: **35/35 PASS**.
- `cargo test --offline --locked --features cubecl-cpu --lib full_dit_cfg_matches_independent_pytorch_from_text_to_output -- --test-threads=1`: **1/1 PASS**, 26–28 seconds on this Linux CPU runner.
- `rustfmt` and `git diff --check`: **PASS**.

A complete serial sweep of every compiled CPU numerical test was **not** run this cycle; earlier iterations documented occasional hangs during an all-in-one CPU test process. No GPU/Windows testing was attempted.

## Progress (functionality and workability are distinct)

- **Functionality: estimated 74%**, from 72% in CONT08; improved numerical correctness for Cross-Lingual and confirmed complete *reduced* DiT. This is an estimate of feature coverage, not an inference-speed metric or count of tests.
- **Workability: 3/7 = 43%**, unchanged. No new GPU, Windows, pretrained-weight or TensorRT-RTX runtime execution was validated.

## Next functional priorities

1. Extend DiT parity across additional shapes, timesteps and CFG values, then assert full 22-block layer-by-layer numerical equivalence using deterministic synthetic weights. Do not confuse this with real pretrained checkpoint testing.
2. Add deterministic request-level noise injection for exact CFM trajectory parity independent of PyTorch vs Rust RNG; compare complete reduced DiT + CFM outputs, then Vocos.
3. Strengthen native Chinese pinyin/segmentation by importing a licensed full word and polyphonic pronunciation lexicon and running direct Python differential tests beyond the current controlled cases.
4. Implement separate native TensorRT-RTX engine in the existing specialized model architecture, without ONNX or `maryml` adaptations.

**Unverified:** 22-block pretrained F5 v1 and Cross-Lingual checkpoints, full-audio parity, CubeCL CUDA/WGPU inference, Windows, TensorRT-RTX, performance.
