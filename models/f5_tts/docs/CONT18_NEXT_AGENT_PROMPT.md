# CONT18 → CONT19, F5-TTS native

Start from published `codex/f5-tts-rust-cont18`. User priority is immutable: **1) cover missing functions, 2) local numerical parity, 3) TensorRT hardware/workability**. Read `docs/CONT18_REPORT.md`, original attached `F5-TTS-main (1).zip`, and `src/f5_tts/model/cfm.py`/`src/f5_tts/infer/utils_infer.py` from original ZIP. Do not open MaryML or alter Fish Speech/RVC/Legacy. No ONNX.

CONT18 implements full speech editing in both CubeCL and TRT (GPU-resident arbitrary CFM mask + full Vocos mel), including exact Python AST masks (320/320), PyTorch Euler/midpoint masked trajectories (3/3), and mock TRT edit routing. Check branch SHA before obtaining bundles.

Next **functional** steps: audit the original CFM sample `no_ref_audio`, `duplicate_test`, `t_inter` and any genuine optional inference features not yet modeled; add only those that belong in model-specific inference. Implement true TensorRT midpoint without computing the second Euler update from the midpoint state (must restore original base state). Audit silence trimming and API-mode differences. Do not assign 100% based on mock compilation alone.

Then extend **numeric** tests for large/unequal edit intervals, real mel extraction before zero injection, F5TTS v1 synthetic weights full PyTorch DiT+CFM+Vocos speech editing and TRT CPU-side contract fixtures. Existing 22-layer and E2E fixtures may be reused; do not copy weights from another model. Only later look at TensorRT GPU real engine, Windows or pretrained weights.

Offline build:
`export PATH=/mnt/data/f5-cont07/rust/bin:$PATH`
`export CARGO_HOME=/mnt/data/f5-vendor-cont09/cargo-home`
`export CARGO_TARGET_DIR=/mnt/data/f5-vendor-cont09/target`
`export LIBCLANG_PATH=/usr/local/swift/usr/lib`
`cargo test --offline --locked --lib --config 'source.crates-io.replace-with="vendored-sources"' --config 'source.vendored-sources.directory="/mnt/data/f5-cont12/vendor-union"'`

Use `--features cubecl-cpu` for numeric Burn tests and `--features tensorrt-mock` for TRT orchestration tests. These aren't evidence of real NVIDIA execution. Publish only changes under `models/f5_tts` to new branch CONT19 via GitHub Connector.
