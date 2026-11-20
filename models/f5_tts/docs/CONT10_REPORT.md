# F5-TTS Native — CONT10: Full DiT/CFM parity and deterministic noise injection

Date: 2026-11-20. Based on published F5-TTS CONT09 GitHub HEAD `5f4405d016d5d3684ed0636917c5aaf2aa06434e`; all changes are confined to `models/f5_tts`. `maryml-main.zip` was not opened or used. ONNX is not used. Fish Speech, RVC, and Legacy files were not changed.

## Scope: mathematical correctness first

- Extended the independent **PyTorch** oracle from a single DiT test configuration to five additional configurations: variable mel length (1, 5, 7, 8, 9), text longer or shorter than mel sequence, batch sizes 1 and 2, varied CFG strengths (including zero) and varied timesteps (including `t = 0` and `t = 1`). The oracle uses the same synthetic `safetensors` weights as Rust, and tests complete text → convolutional input → DiT/CFG forward.
- Added two **complete Euler trajectories**, at 4 and 3 steps respectively, with explicit initial noise, reference masking, PyTorch CFG velocity at every step, and prefix inpainting after integration. The Rust test compares **each intermediate state** and the final `integrate_with_channels` output to independent PyTorch reference vectors. The real model keeps the original 100-mel input constraint; the shared internal integration helper allows testing an isolated 2-mel synthetic DiT without duplicating the inference algorithm.
- Added a separate 22-layer DiT synthetic checkpoint, with distinct random weights across 22 layers and independent PyTorch reference states after **each** Transformer block. The Rust 22-block path checks all intermediate states and CFG output. This checks the depth of F5TTS_v1_Base, but not its production width (the test intentionally uses dimension 16).
- Refactored `DiTBackbone` so normal forward and test-only intermediate tracing execute the **same** math path through `forward_with_observer`. There is no separate fork of the DiT forward equations.
- Added `StandardF5Runtime::synthesize_with_noise()` for deterministic external noise injection. This permits later whole-pipeline PyTorch parity without emulating `torch.randn`'s RNG algorithm. Normal seeded synthesis retains its existing request-level RNG. The shared inference path validates noise length and finiteness before constructing a GPU tensor.

## Numerical evidence

- Five independent DiT shape/CFG scenarios: **5/5 PASS**, each compared to PyTorch at `max_abs <= 2e-4`.
- Two independent DiT + CFM trajectories: **2/2 PASS**, checked at every Euler step (4 + 3 steps) and after inpainting, `max_abs <= 2e-4`.
- Full **22-block** DiT forward: **22/22 intermediate block outputs + final CFG PASS**, same numerical tolerance.
- Existing baseline full DiT reference still **PASS** after the shared-forward refactor.
- Existing autonomous Rust/Python frontend differential tests: **35/35 PASS**.
- Deterministic-noise validation tests added (invalid length, NaN, infinity rejection), independent of learned checkpoint inference.

The relative and maximum absolute errors beyond the asserted upper bound are **not claimed**: they were not individually recorded in the test output. Inference on pretrained weights and complete audible TTS output are not part of this contour.

## Reproducibility

- Offline Cargo vendor: [GitHub Actions run #44741264056](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44741264056), artifact `f5-tts-cargo-vendor-cont08` (bundle and lockfile), same pinned Burn/CubeCL dependency set as CONT09.
- Toolchain: Rust 1.100.0, Burn 0.22.1, CubeCL CPU using LLVM 23.1.0-4.
- Oracle regeneration: `python tests/tools/generate_full_dit_fixture.py` (requires Python/PyTorch/safetensors only in the **test environment**, not the Rust runtime).
- Core CPU parity tests: `cargo test --offline --locked --features cubecl-cpu --lib full_dit_dynamic_shapes_and_cfg_match_pytorch -- --test-threads=1`, `cargo test --offline --locked --features cubecl-cpu --lib twenty_two_dit_blocks_match_independent_pytorch_layer_by_layer -- --test-threads=1`.
- No vendor tree, model pretrained weights, or original test reference audio are added to Git.

## Progress

- Functionality: **estimated 76%** (CONT09: 74%), primarily numerical coverage at full 22-block depth and deterministic input-noise support. This is an engineering estimate, not a count of passing tests or end-to-end application readiness.
- Workability: **43% (3/7)**, unchanged. No new evidence of actual CUDA/WGPU inference, Windows execution, TensorRT-RTX native runtime or real checkpoint functionality.

## Remaining priorities

1. Generalize parity fixtures from dimension 16 to realistic dimension 1024 while preserving isolated, measurable numerical checkpoints; do not confuse this with real pretrained checkpoint validation.
2. Verify Vocos + ISTFT and mel frontend end-to-end on identical fixed synthetic mel/noise to establish full audio numerical agreement, maintaining pure Rust/CubeCL inferencing.
3. Close Chinese `rjieba` and `pypinyin` exact segmentation, polyphones and tone sandhi against a licensed lexicon and Python corpus.
4. Implement the separate native TensorRT-RTX backend under the same model-specific interfaces, without ONNX. CUDA/WGPU and Windows workability checks remain secondary.
