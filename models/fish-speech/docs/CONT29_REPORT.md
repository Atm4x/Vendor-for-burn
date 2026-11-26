# Fish Speech 1.5 — CONT29: F11 native CUDA model facade

## Decision on F08

F08 remains 4/5. The currently vendored Symphonia 0.6 components only include PCM/ADPCM/FLAC/Vorbis and RIFF/OGG/MKV. MP3/AAC, MP4/M4A container support, and WMA/ASF decoding require implementing substantial additional codecs/containers or adding a platform decoder / FFmpeg. No new dependency was requested; there is no portable in-tree implementation of all four. Rather than claim a fictitious DONE, this cycle focuses on F11.

## Implemented

1. Refactored `model_runtime.rs` to a generic `FishModel<G: SemanticGenerator>` facade. `CpuFishModel = FishModel<CpuSemanticGenerator>` preserves existing public `open`, `reload`, `prepare_references`, `set_reference_audio_loader`, `synthesize`, streaming, `close` and cache controls; no copy of the synthesis pipeline.
2. Added `cuda_model_runtime.rs` behind `cuda-driver + tensorrt-api`: `CudaFishModel = FishModel<CudaSemanticGenerator>`, `CudaFishModel::open_cuda(ModelPaths, CudaRuntimeOptions, seed, SelectionMode::Sample)` and `LoadedFishArtifacts::into_cuda_model`.
3. The GPU semantic generator takes the verified original semantic SafeTensors, tokenizer IDs and Firefly checkpoint, builds its semantic LUT and selects DualAR; Naive and greedy mode explicitly error rather than falling back silently.
4. For each prompt chunk it derives `PreparedDualArPlans` automatically from actual prompt length and maximum tokens, validates required checkpoint tensor names, shapes and dtypes before context creation, and invokes `generate_prepared_cuda_dual_ar` with owned `EngineRegistry` and shared `DeviceMemoryBudget`.
5. The same top-level TTS request may contain multiple samples/chunks and may be called repeatedly with the same model. Serialized TensorRT cache and GPU budget persist on the model. Request-scoped execution contexts are still recreated per chunk (F12 remains open).
6. The existing static-profile generator is bounded to `max_profiled_new_tokens` 1..=64; larger requests are rejected before hardware access. F12 must replace this bounded static-profile strategy with dynamic/windowed decode to support long generations efficiently.
7. Added no-GPU tests for GPU options, bounded profiles and source-derived checkpoint weight preflight, plus CPU regression tests inherited by the generic facade.

## Verification

- `cargo test --offline --locked --lib`: **53 PASS, 2 ignored**.
- `cargo test --offline --locked --lib --features cuda-driver,tensorrt-api`: **59 PASS, 2 ignored**.
- `cargo test --offline --locked --features tensorrt-mock`: **53 PASS, 2 ignored** plus blueprint 4 PASS, resident-context 3 PASS, tensorRT mock 1 PASS.
- `core/cargo test --offline --locked`: **225 PASS, 2 ignored**, summing all suites.
- `cargo check --offline --locked --features cuda-driver,tensorrt-api`: **PASS**.
- Source formatting and whitespace verified.
- No real Windows/NVIDIA inference, native GPU performance measurement, or numerical parity with original PyTorch was executed.

## Functional status

- F11: **6/6 DONE** (source-level native model integration) from 4/6 in CONT28.
- All 22 top-level items: **17 DONE, 5 PARTIAL**; coarse score `(17+5*0.5)/22` = **88.64%**.
- Stable acceptance baseline: **24/36 = 66.67%**, up from 22/36 = 61.11%.
- F08 4/5; F12 5/7; F13 3/6; F14 2/5; F22 4/7.
- Separate operability baseline remains **7/11** until actual hardware gates are verified.

## Next functional milestone: F12

Add bounded dynamic/windowed Slow decode profiles with real TensorRT context/engine reuse *between* chunks/requests. Current `CudaFishModel` is a public runtime choice, but the request-scoped contexts and fixed-length profile expansion remain costly; production long-form generation is not yet covered. Then address F13 GPU-side sampling and Naive AR F14.

## Git workflow

Commit locally and publish to `Atm4x/Vendor-for-burn:codex/fish-speech` through GitHub Connector, preserving parent and tree. Do **not** rerun Git Bridge simply because a commit was published if the local repository already has the full history and the tree matches.
