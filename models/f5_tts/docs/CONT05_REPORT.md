# F5-TTS Native — CONT05: Cross-Lingual functional coverage and PyTorch parity

## Scope

Parent: local CONT04 `dfb40ddae379e5919c141bd0fb8a8c07e39fffbf`, matching GitHub `codex/f5-tts-rust-cont04` at `0bee79af005bebbb4ca42765cf6a9a7da1e684d6`. All changes are isolated to `models/f5_tts`. No Maryml source or ONNX runtime used. Pretrained checkpoints and GPU hardware are explicitly deferred: functionality and numerical parity have priority over operability.

## New functionality

1. `speed_predictor.rs`: Cross-Lingual SpeedTransformer with independent architecture (mel projection, grouped conv positional embedding, Mish, RoPE/masked multi-head attention, six residual Transformer blocks with tanh-GELU FFN, masked attention pooling, GELU classifier, 32 classes mapped to 0.25–8.00 syllables per second). Runtime accepts `[B,T,100]` mel and independently provided valid sequence lengths. Full-production configuration: dim 512, depth 6, heads 8, FF multiplier 4, 16 groups.
2. `hyphenation.rs`, `assets/hyph_en_US.dic`: native Tex/Hunspell pattern-based English syllable counting with bundled US dictionary and original BSD-style redistribution text; no pyphen runtime or Python dependency. Tested against 552 independent pyphen examples.
3. `cross_lingual.rs`: text chunk duration planner that combines reference frames and predicted rate, correctly distinguishes fixed **total** duration from generated-only duration, and linear PCM overlap cross-fade using NumPy `linspace` endpoint semantics.
4. `pipeline.rs`: CrossLingualF5Runtime sharing the existing DiT/CFM/Vocos runtime. Rate prediction occurs before chunking; a caller-supplied text normalizer/tokenizer handles future Chinese/pinyin support without binding the neural model to a linguistic implementation. Chunks are generated sequentially and joined with cross-fades.
5. `dit.rs`: one attention implementation, optional key/output masks for the speed predictor and no mask overhead for the standard DiT.

## Numerical parity

- Independent PyTorch `SpeedTransformer` fixture (`tests/tools/generate_speed_predictor_fixture.py`) exports full toy network weights and expected logits. Batched test contains padded sequence lengths `[5,3]`, 100 mel channels, two heads, one block. CubeCL CPU `max_abs = 1e-8` at logits (printed from the test). This verifies the implemented graph on fixed parameters, **not** the original 512-wide / six-layer pretrained checkpoint.
- 552 native English hyphenation decisions match Python `pyphen` with US dictionary on a fixed mixed corpus.
- Previous DiT, MelSpectrogram, Vocos and ISTFT numerical fixtures remain present; original Vocos isolated regression passes.

## Test results

- `cargo test --lib --offline`: 32/32 PASS.
- `cargo test --features cubecl-cpu --lib --offline -- --test-threads=1 --nocapture --skip vocos_backbone_head_and_waveform_match_pytorch`: 40/40 PASS.
- Isolated `vocos_backbone_head_and_waveform_match_pytorch`: 1/1 PASS.
- Isolated `masked_speed_transformer_matches_independent_torch`: 1/1 PASS, max_abs = ~1e-8.
- `cargo check --features cubecl-wgpu --offline`: PASS.
- `cargo check --features cubecl-cuda --offline`: PASS.
- Full combined CubeCL CPU suite stalls intermittently after the SpeedTransformer or on Vocos; it has **not** been recorded as 41/41 PASS. All cases passed when split into the above deterministic runs. The cause of intermittent shared-process CPU backend stalls remains open.

## Functional coverage score (engineering estimate): 59/100, previously 50/100

| Category | Done / total |
|---|---:|
| Core runtime, mel, checkpoints and CFM | 14 / 15 |
| DiT Transformer math | 12 / 25 |
| Text frontend and normalization | 10 / 15 |
| Mel, reference and Vocos audio | 12 / 15 |
| Cross-Lingual speed, syllables, segments | 8 / 10 |
| TensorRT-RTX backend | 0 / 10 |
| Integration, IPC and full voice service | 3 / 10 |

## Operability: 3/7 (43%), unchanged

CubeCL CPU, CUDA feature compile, WGPU feature compile confirmed. Real GPU/Windows and original pretrained checkpoint execution not tested by design.

## Remaining functional work in order

1. Directly verify the full DiT output on deterministic input with identical checkpoint weights and explicit noise tensor; build comprehensive comparison for attention, text conditioning and CFG beyond toy-scale layers.
2. Finish native text/pinyin normalization and resampling semantics, replacing the caller-supplied normalizer with a tested Rust adapter; add cross-lingual end-to-end CPU parity using deterministic model slice.
3. Implement full Vocos real architecture mapping (including exact weight names and phase convention), while leaving original heavyweight checkpoint execution for the separate operability stage.
4. Introduce the independent TensorRT-RTX backend against the same model-specific graph contract after CubeCL parity.
5. Separately complete real checkpoint/Windows GPU/IPC performance and feature-gating coverage. This does **not** block the first four functional objectives.
