# Fish Speech 1.5 — CONT27: source-matched DualAR generator, CUDA/TensorRT integration

## Source and scope

- Base local HEAD `0badb7894f2ca8e95ac49a23253197d8c250d078`; `fish-speech-cont26-git-bridge.zip` already contains that tree and history. No Git Bridge export is needed or requested for this cycle.
- Functional source: uploaded `Fish-speech-pipeline-main (1).zip`, specifically `fish_speech/models/text2semantic/inference.py::decode_one_token_ar`, `generate`, `decode_n_tokens` and `llama.py::DualARTransformer.forward_generate{,_fast}`. This archive remains local only.
- TensorRT-RTX remains unavailable for *execution* in this Linux environment. GPU feature compilation does not establish numeric inference correctness.

## Functional changes

1. `core/src/ar_generation.rs` introduces `DualArBackend` and a single `generate_dual_ar` source-order controller shared by CPU and CUDA. The controller owns the semantic/fast-codebook sequence, prompt prefill, cached Slow decode, Fast reset each Slow step, Fast position-zero priming, semantic-derived first acoustic code, remaining Fast samples, EOS termination, cancellation, and the existing 16-token repetition history. Request progress trace persists on mid-generation cancellation.
2. `core/src/cpu_weighted_ar.rs` now uses the *same* controller through `CpuDualArBackend`, reusing its existing weighted interpreter and stable Slow/Fast cache ownership rather than maintaining an independent copy of the generation loop.
3. `src/cuda_generation.rs` adds `PreparedDualArPlans`, bounded per-position Slow profiles and Fast steps, and `CudaDualArBackend` over the resident `WeightedCudaArSession`. It uploads token buffers and the semantic LUT, submits weighted TensorRT plans on one stream, transfers Fast context directly as a CUDA buffer, and reads F32 logits only at sampling boundaries. `generate_prepared_cuda_dual_ar` connects this backend to the shared control loop, with seeded CPU sampling and request cancellation.
4. A verified lowering mismatch was fixed: the original graph output `logits` is emitted as `logits.gemm` by the TensorRT lowering. CUDA output lookup now uses **original Graph ValueIds**, not lowered step names; tests cover Slow logits, Fast logits, Fast context, profile positions and runtime inputs.
5. Before requesting any engine/build/device session, GPU request validation checks the model identity, compiled plan profile, channels, mask lengths, semantic LUT type/length and EOS/token boundaries. All GPU-plan fields are private, avoiding tampering of a validated profile.

## Verification (Linux, Rust 1.100.0, offline)

- Core full suite: **225 PASS, 2 ignored**, including **197 Core unit PASS** and all existing integration suites.
- Outer crate `--lib`: **50 PASS, 2 ignored**.
- TensorRT mock suite: **50 PASS, 2 ignored**, plus **4 blueprint**, **3 resident session**, **1 TensorRT mock** pass.
- CUDA + TensorRT (both features) `cargo check`: **PASS** with Clang 17 (`LIBCLANG_PATH=/usr/local/swift/usr/lib`).
- CUDA + TensorRT (both features) `cargo test --lib`: **53 PASS, 2 ignored**. The new plan test runs without initializing CUDA or invoking TensorRT inference.
- Rustfmt and `git diff --check`: **PASS**.
- Authentic checkpoint on Windows/RTX, GPU numerical parity and GPU performance: **NOT TESTED**.

## Functional vs operability gates

- F12: **PARTIAL**, now linked from source-order autoregression through `WeightedCudaArSession` to resident TensorRT dispatch. The complete prefill/decode route can be invoked for an explicitly **bounded** prepared request, but authentic model logits/PCM and dynamic/unbounded decode profiles are not verified.
- F13: **PARTIAL**; Fast AR codebook ordering is wired, but GPU-resident sampling (no readback/sync each token) is not implemented. Current sampler is deterministic Rust CPU-side, not original PyTorch RNG parity.
- F01: source-grounded Firefly 1.5 config and 450/450 weight-shape parity remain intact; untested configurable variants are still unsupported.
- F08 and F22 remain partial as in CONT26. No fake functional DONE was credited.
- Functional tracker: **15 DONE + 7 PARTIAL of 22 = 84.1% historical equal-weight**.
- Operability tracker: **7/11 historical confirmed gates**; CUDA feature compilation and mock tests are new Linux evidence but do not replace missing Windows / authentic-checkpoint GPU gates.

## Architectural limitations and next priority

- `PreparedDualArPlans` explicitly prepares one TensorRT engine profile for each Slow decode length up to `max_new_tokens`; for long prompts this is expensive. A production implementation should support dynamic sequence-length profiles or bounded compilation windows with cache reuse and lazy context materialization.
- Sampling still requires GPU→CPU logits readback and synchronization; F13 should add a GPU sampler and EOS/termination transfer contract.
- The TTS model-runtime facade still defaults to CPU. A production GPU model loader and Firefly GPU acoustic decoder must be integrated before claiming complete native CUDA TTS.
- Next: validated, windowed Dynamic Slow decode + GPU sampler + real checkpoint/CUDA parity harness. Continue native F01 variants only when supported source configurations are available.

## Git Bridge policy

Do not re-run a Git Bridge export merely because a new GitHub commit was published. Use the exact local working tree, compare Git tree/blob SHA against the published GitHub tree, and export anew only when source history/artifacts actually cannot be reconciled locally. No Git Bridge action is started for this cycle.
