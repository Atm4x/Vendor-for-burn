# RUAccent tiny: Windows native verification and ONNX golden diagnosis

Date: 2026-11-19. Rust toolchain: 1.100.0. ONNX Runtime native binary: Microsoft.ML.OnnxRuntime 1.22.0 (official NuGet). Official RUAccent model repository: `ruaccent/accentuator`, pinned revision `b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855`. Models are never committed.

## Windows x64 test

[GitHub Actions run #44601030315](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44601030315). Workflow: `.github/workflows/ruaccent-windows-native.yml`.

- Verified 13/13 model assets (dictionaries, tokenizers and four ONNX models).
- Native `models/ruaccent` tests with `onnx` enabled: **39/39 functional unit tests PASS**, **2/2 worker unit tests PASS**, **1/1 golden text-splitting test PASS**.
- The **two strict raw-logit tests are intentionally excluded** from the passing functional gate; they remain failing in the prior Linux audit. Do not claim 41/41.
- Actual Windows `ruaccent-worker.exe` with real weights: 5/5 JSONL requests passed: load, two process requests, status, shutdown. Expected phrases exactly matched:
  - `На двери висит замок.` → `На дв+ери вис+ит зам+ок.`
  - `Он все понял и ушел.` → `+Он всё п+онял и уш+ёл.`
- Optional Silero frontend compiled on Windows: `cargo check --no-default-features --features reference-flex,ruaccent --lib` **PASS**, approximately 1 minute 45 seconds.
- The normal `text` frontend retains SileroStress. RUAccent is opt-in using `ruaccent_text` and `ruaccent_dir`, see `models/silero-tts/docs/RUACCENT_OPT_IN_FRONTEND_2026-11-19.md`.

## Python ONNX Runtime 1.22.0 versus captured golden raw logits

[GitHub Actions run #44601566531](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44601566531). Workflow: `.github/workflows/ruaccent-onnx-logits-diagnostics.yml`. The script feeds the **original recorded `input_ids`, `attention_mask` and `token_type_ids`** from `golden/tiny.jsonl` directly to the pinned official ONNX models via Python `onnxruntime==1.22.0`, then compares logits.

Four graph-optimization modes (DISABLE_ALL, BASIC, EXTENDED, ALL) and intra-op thread counts 1 and 4 were tested.

| Neural component | Max absolute difference vs original golden | Violations of `1e-4` | Predicted class parity |
|---|---:|---:|---|
| Accent logits, EXTENDED or ALL | **0** | 0 | yes |
| Accent logits, DISABLE_ALL or BASIC | **9.54e-7** | 0 | yes |
| Omograph logits, all tested modes | **about 0.0589623** | 8/8 values | yes |

Conclusion: the omograph's captured Python golden logits do **not** exactly match the current official pinned model when run with Python ONNX Runtime 1.22.0 either. The discrepancy is independent of graph optimization and tested CPU thread counts. Its cause remains unverified (candidate explanations include model or export revision mismatch in the original golden harness). The resulting class selection is identical in all tests.

The Rust tests previously also reported at least one `1e-4` violation each (accent sample `7.4491653` versus `7.4505863`; omograph sample `-1.541515` versus `-1.5394485`). **Those examples are not the measured maximum deviations across their tensors.** Do not infer that all Rust logits agree with Python within `0.0021`. Functional string parity is established separately.

## Outstanding release gates

1. True end-to-end `ruaccent_text` → Silero PCM test with **both** real models loaded, preferably Linux and Windows; `cargo check` by itself does not establish PCM output.
2. Determine provenance of the original Python golden model export, especially for `nn_omograph/tiny`, and investigate the Rust/Python accent-logit discrepancy using a common native ONNX Runtime build.
3. Windows executable packaging must include the ONNX Runtime DLL and licensed RUAccent model pack; no implicit network download at runtime.
4. Optional Cargo dependency versions and the repository's source-only vendor mirror should be included in the eventual release/build packaging plan.

## Evidence

- Workflow artifact `ruaccent-windows-native-evidence` (run 44601030315): `ruaccent-tests.log`, `ruaccent-worker-build.log`, `windows-worker.json`, `silero-adapter-check.log`, `model-manifest.json`.
- Workflow artifact `ruaccent-onnx-numerical-audit` (run 44601566531): `python-onnx-logits.json`.

Both artifacts contain **no model weights**.
