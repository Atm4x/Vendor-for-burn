# Fish Speech 1.5 Rust — CONT15 report

**Date:** 18 November 2026. **Source:** `FISH_SPEECH_CONT14_MODELS_FISH_SPEECH_SOURCE_ONLY_2026-11-18.zip` plus the provided modified `Fish-speech-pipeline-main(1).zip`. **Workflow:** wholly local; no GitHub mutations, clone, CI dispatch or remote dependency download. Rust 1.100.0 and pinned offline vendored crates, no Windows/.NET or GPU invocation.

## New implementation

1. `core/src/cpu_weighted_ar.rs`: `CpuWeightedDualAr` — stateful, executable weight-bound CPU reference. It calls Slow prefill, then cached Slow decode, extracts semantic logits, uses the Fish sampling implementation, primes Fast AR on step 0 and derives acoustic codebook 0 from semantic ID, then executes Fast steps 1..N−1 with separate per-Slow-step KV. Explicit EOS, max tokens, RNG, cache reset and `SemanticGenerator` implementation. Trace records Slow/fast logical lengths and resets. This is reference math, not optimized CPU inference.
2. `core/src/cpu_weighted_naive.rs`: `CpuWeightedNaive` — single Slow AR plus `codebook_norm` and `codebook_output` to generate N independent acoustic codebook distributions. Semantic sampling overrides (`temperature=0.1`, `top_p=0.1`, penalty=1) follow the *non-agent* `decode_one_token_naive` in this Python archive. Caches and windows are request-scoped; implements the synthesis generator interface.
3. `core/src/transformer_ir.rs`: new typed Naive `codebook_norm` and `codebook_logits` outputs; no fabricated Fast AR for `ModelKind::Naive`.
4. `core/src/cpu_ir.rs`: RoPE CPU reference now delegates to `rotate_bhtd`, which uses BF16-rounded frequency cos/sin coefficients like the Python `precompute_freqs_cis`; removed the prior unrounded divergent formula.
5. `src/weights.rs`: SafeTensors `LoadedWeights::to_cpu_named_tensors` and `semantic_cpu_weights` resolve all Slow/Fast checkpoint keys, enforce shapes and convert F32/F16/BF16 to CPU F32. Manifest hashes were already checked by the loader. Nonfinite values are rejected.
6. `src/weighted_cpu.rs`: `CpuSemanticGenerator::from_checkpoint` chooses DualAR/Naive by `FishModelConfig`, builds semantic-ID LUT from `SpecialTokens`, loads the weights, and exposes the common `SemanticGenerator` interface.
7. Test additions cover cache isolation and progression, Fast prime vs sample positions, unequal Slow/Fast hidden sizes, Naive logits layout, EOS, zero/missing weights, BF16/F16 SafeTensors conversion and one end-to-end *interface* test using a fake PCM decoder.

## Validation — exact, no invented GPU claims

| Check | Result |
|---|---|
| `cargo test --locked --offline --manifest-path core/Cargo.toml` | **181 unit + 27 integration = 208 PASS** |
| `cargo test --locked --offline` outer crate | **18 PASS** |
| `python tools/test_export_checkpoint.py` | **5 PASS** |
| `cargo fmt --all -- --check` | **PASS** |
| `cargo check --locked --offline --features tensorrt-api` | **PASS**, C++ bindings compilation only |
| `cargo test --features tensorrt-mock --lib` | **TIMEOUT**, while compiling `trtx-sys`; no test result obtained |
| Authentic Fish Speech checkpoint/tokenizer full parity | **BLOCKED_ON_ARTIFACT**, not supplied |
| Real TensorRT-RTX build/inference | **NOT_TESTED_GPU** |
| 44.1 kHz speech using model weights | **NOT_TESTED**, weighted Firefly runtime missing |

The test setup uses a shared pre-existing Cargo target directory and offline `vendor/` outside the source ZIP. Portable path configuration must be adjusted to the target user's Cargo vendor location.

## Progress ledgers

- **Functional source-behavior coverage: 77.27%** (12 DONE / 10 PARTIAL / 0 MISSING across 22 equal-weight items), versus CONT14 70.45%. Full detail and remaining concrete work in `FISH_SPEECH_CONT15_FUNCTIONALITY_MATRIX_2026-11-18.md`.
- **Operability/acceptance:** local Core/outer/exporter/FFI tests pass; real RTX/Windows checkpoints are separately 0/3, independent of functional progress.

## Architectural boundary and remaining risks

- CPU executor intentionally clones weight tensors into per-graph inputs, and is O(context × layers) with repeated graph construction; it is a **functional reference**, not a recommended production backend. GPU execution still needs TensorRT/stream/allocator implementation.
- The `SemanticGenerator` now uses real weighted CPU graphs, but the **`AcousticDecoder` remains a trait/fake in E2E tests**. Real Firefly waveform inference (and reference encode) is not done.
- Exact model weight naming/layout assumptions and tokenizer parity with authentic assets must be verified when those artifacts are available. Do not claim exact complete-model parity from synthetic weights.
- Neither MP3/AAC format support nor model warmup/unload/queue/per-token cancellation was completed.
- Do not edit legacy/obsolete Python scripts; they were used only as references.
