# F5-TTS Native — CONT08

Date: 2026-11-20. Source: locally restored CONT07 (`c84f4c4f204ad3e001a5b0b7c8757de89bd534dc`), published remote parent `d310a27dae283a79f31f13aceaf56a074db5a1fd`. Scope: `models/f5_tts` only. The user-provided `maryml-main.zip` was not opened. No ONNX, no change to Fish Speech, RVC, Legacy or other workspace files.

## Functionality first

### Native Chinese frontend

- Added `ChinesePinyin` with 26,676 Unicode Han readings covering the original F5-TTS `U+3100..U+9FFF` range, built deterministically from a locally available Unihan-based xpinyin database. Readings are converted from diacritical pinyin to ASCII TONE3 tokens with the correct `v` spelling.
- Added bundled Chinese phrase overrides, longest-match dictionary word selection, and limited word-local tone sandhi (third-tone pairs, 不 before tone 4, 一 before tones 1–4, 第一 exception). Allows replacement or extension via `ChinesePinyin::extend_words`.
- Changed standard F5-TTS to cache a native Chinese provider (shared `Arc`) at runtime creation and accept Chinese text through `synthesize_text()` and multi-voice `synthesize_script()` without Python. The explicit `synthesize_text_with_pinyin()` path remains available for exact external vocabulary providers.
- Added `CrossLingualF5Runtime::synthesize_chinese()` with the same native provider and `replace_native_pinyin()` on both runtimes. Existing `synthesize_english()` uses its prior behavior.
- Preserved separation of the native tokenizer from the backend/DiT/CFM/Vocos computation. No GPU host sync or new model framework.

### Important accuracy boundary

The bundled character readings are derived from **Unihan**, and the small curated phrase lexicon is **not the full `rjieba` word dictionary or the `pypinyin` polyphonic dictionary**. Third-tone and 一/不 heuristics do not cover all lexical exceptions or neutral-tone policies. Therefore **Chinese pronunciation parity with Python is NOT established** on arbitrary input. The existing 505 Python tokenizer cases use *controlled segmentation/pronunciation*, not a real `rjieba`/`pypinyin` corpus. Do not confuse successful token-shape parity with verified phonological parity.

The original source (`xpinyin-database.def`, Qing Lee 2012–2022, LPPL 1.3c+) and derivative licensing are documented in `assets/README_pinyin.md`. The generated asset is renamed and restricted to the supported Unicode range. Generator: `tests/tools/build_pinyin_unihan_asset.py`. No TeX/Python files are needed at runtime.

## Verification evidence

- Rust 1.100.0 `rustc --edition 2024 --test tests/offline_frontend.rs`: **22/22 PASS**.
- All **26,676** bundled Chinese readings can be loaded and queried, and the direct tokenization of representative mixed Chinese/English text passes.
- Regression checks: 1,006 Python UTF-8 chunking cases, 505 Python controlled pinyin tokenization cases, 52 PyTorch CFM schedules and 4 deterministic Euler+CFG trajectories retained and passed as parts of the offline harness.
- `rustfmt --check` on changed Rust sources: PASS.
- `cargo check --offline --no-default-features`: BLOCKED, missing registry crate `burn` (even optional dependencies require resolution). **No new CubeCL CPU DiT parity test or GPU test was executed.** Existing full DiT numerical fixture remains unverified.

## Cumulative progress estimate

- **Functionality: 72/100** (CONT07 70/100). Native Chinese fallback and runtime integration added, but full `rjieba`/`pypinyin` parity is still open.
- **Workability: 3/7 = 43%**, unchanged. Hardware, Windows, TensorRT-RTX and pretrained end-to-end execution are separate workability gates.

## Remaining functionality, prioritized

1. Run full DiT CPU numerical fixture once Burn/CubeCL source+Cargo registry are available offline; resolve mismatched operators by numerical evidence.
2. Replace/augment small Chinese phrase lexicon with a full licensed rjieba-compatible segmentation dictionary and pypinyin word/phrase readings; add direct corpus differential tests for polyphonic words, sandhi, neutral tones and mixed CJK/Latin text.
3. Complete model weight mapping, CFM/Vocos end-to-end parity, Torch RNG semantics or deterministic input-noise injection, and remaining integration/IPC contracts.
4. Implement independent **native TensorRT-RTX backend**, no ONNX and no reuse of Burn execution internals.

No progress was claimed for pretrained-weight execution, GPU throughput, or Windows runtime.
