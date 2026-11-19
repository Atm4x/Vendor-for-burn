# Fish Speech 1.5 — CONT26: upstream Python source audit and F01 config contract

## Source and integrity

- Exact source archive supplied by the user: `Fish-speech-pipeline-main (1).zip` (40 `.py` files), local-only extracted to `fish-cont26/python/`. Original source contains `fish_speech_lib/config.py`, `vqgan/modules/firefly.py`, `vqgan/modules/fsq.py`, text2semantic/llama/inference and tokenizer routines. The upstream source archive is **not copied into the GitHub repository**.
- Base: exact Git Bridge `fish-speech-cont25-git-bridge.zip`, `source-head.txt = 7b5f84b857d4429c266fd066b81d79725b682e50`, matches remote `codex/fish-speech` HEAD; SHA-256 and `git bundle verify` PASS.
- Rust 1.100.0 offline vendored source, no GPU/checkpoint assets or internet Git development.

## Observed upstream Firefly architecture

`FIREFLY_GAN_VQ_CONFIG` in the uploaded source specifies sample rate 44,100; 160 mel bins; FFT/window 2,048 and hop 512; ConvNeXt depths `[3,3,9,3]` and widths `[128,256,384,512]`; HiFiGAN rates `[8,8,2,2,2]`, kernel sizes `[16,16,4,4,4]` and resblock kernels `[3,7,11]`; 512-wide FSQ, 8 groups, 1 codebook, levels `[8,5,5,5]`, downsample `[2,2]`.

Original `firefly.py`/`fsq.py` modules were imported using CPU PyTorch meta tensors and an explicitly shape-compatible, parameter-only FSQ scaffold. The existing Rust `FireflyGraph` produced 450 distinct parameter names and shapes. **Source = 450, Rust = 450, missing = 0, extra = 0, mismatched = 0.** This is an **architectural parameter audit**, not numerical parity, checkpoint validation, CUDA performance or full FSQ implementation proof.

## Changes

1. `src/firefly_config.rs` introduces a typed, fail-closed Firefly schema, checking the source variant's explicit `_target_` implementations, mel/STFT, backbone, decoder, FSQ and convolution geometry against the Rust graph. Unsupported variants, unexpected/missing attributes and non-JSON expressions are rejected. This check now occurs at `LoadedFishArtifacts::open`, after the manifest's config SHA-256 check and before any model construction.
2. `tools/normalize_firefly_config.py` converts the source's literal `FIREFLY_GAN_VQ_CONFIG` into canonical JSON **without importing or executing Python model code**. Multiple assignments, dynamic expressions and other statements are rejected. `tests/fixtures/firefly_1_5.json` is generated from the exact supplied source.
3. `tools/audit_firefly_shapes.py` now reads the actual source config instead of using embedded hard-coded parameters. `tests/fixtures/firefly_python_weight_shapes.tsv` stores all 450 upstream-derived name/shape pairs; a Rust test checks the encoder+decoder union against these values.
4. `tools/export_checkpoint.py` now verifies the supported normalized Firefly config *before* loading or exporting tensors, preventing incompatible manifests. Existing `generator.` prefix handling and legacy/PyTorch parametrization weight-normalization conversion remain unchanged, with their existing tests.
5. `model_runtime` tests now use a valid normalized Firefly fixture and check that a mismatched Firefly architecture is rejected even when its config hash matches a manifest. Added Python static-normalizer and exporter rejection tests.

## Verification

- Rust outer crate `cargo test --offline --locked --lib`: **50 PASS, 2 ignored**.
- Rust Core full `cargo test --offline --locked`: **193 PASS, 2 ignored**, remaining integration suites PASS (see local `tests-core.log`).
- Python `unittest test_export_checkpoint.py test_normalize_firefly_config.py`: **9 PASS**.
- Python original-source shape audit: **450/450 PASS**.
- Source JSON regeneration compared byte-for-byte with fixture: **PASS**.
- Rust formatting and Git diff whitespace: **PASS**.
- `tensorrt-mock` and CUDA+TensorRT compile were attempted on a freshly cloned checkout. The former initially failed due to unset `LIBCLANG_PATH`, and subsequent compilation retries exceeded local time limits; do **not** mark these checks PASS for CONT26. Prior CONT25 CUDA+TensorRT gates still apply only to that earlier commit.
- No RTX execution, real checkpoint inference, or numerical parity was run.

## Compatibility and migration

The old runtime only checked the byte hash of arbitrary `config.py` content and accepted it without interpreting its architecture. The strict Rust runtime now requires a **normalized Firefly JSON config** that matches the checked graph. Old manifests referencing `.py` must be re-exported (or otherwise regenerated in a verified offline workflow) with the same normalized `.json` supplied as `--config`; unmodified manifests are intentionally rejected rather than silently treated as equivalent. Normalization command:

```bash
python models/fish-speech/tools/normalize_firefly_config.py --source /path/to/fish_speech_lib/config.py --out /path/to/firefly-config.json
```

Use `/path/to/firefly-config.json` for the Firefly checkpoint exporter and as `ModelPaths::new(..., firefly_config_source)` at runtime. The Python source is **never executed by the Rust inference runtime**.

## Functionality/operability tracker

- F01 now has source-grounded validation, checkpoint aliases already exercised by exporter tests and exact structural parity of the **single concrete Firefly 1.5 configuration supplied in this zip**. Arbitrary upstream Firefly configuration variants remain unsupported rather than guessed; therefore F01 stays PARTIAL if the tracker promises all variants.
- All F11–F14/TensorRT functional limitations from CONT25 remain; this cycle did **not** claim a completed Slow decode or CUDA inference.
- Functional historical metric remains **15 DONE + 7 PARTIAL / 22 = 84.1%**; operability historical **7/11** pending fresh GPU validations, reported separately.
- Next F01: decide whether supported Firefly variants should be exactly those shipped in the reference or whether configurable graph generation is required. If full variants are required, derive those variants from actual source configs rather than assuming their geometry.
- Next F12: wire the resident Slow AR executor into top-level inference and validate generated logits, KV progression and error recovery against reference Python generation.
