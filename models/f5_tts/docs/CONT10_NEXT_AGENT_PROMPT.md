# Next F5-TTS Rust contour (CONT11)

Continue from GitHub `Atm4x/Vendor-for-burn`, `codex/f5-tts-rust-cont10`. Do not open/use `maryml-main.zip` as implementation architecture. Keep all changes inside `models/f5_tts`; never edit obsolete or other model scripts. Work locally, publish only at end via GitHub Connector in a **new branch**.

Main priority: functional coverage of original F5-TTS v1 and Cross-Lingual and numerical PyTorch equivalence. Maintain **two separate progress bars**, functionality first, workability second. Do not prioritize Windows, real checkpoint or hardware tests over closing known model functions.

Installed offline Cargo vendor originates from GitHub Actions artifact `f5-tts-cargo-vendor-cont08` (run 44741264056), created at exact pinned source CONT08. Rust 1.100.0, LLVM 23.1.0-4, Burn 0.22.1. Need no repeated Cargo network/environment diagnostics if these are available.

CONT10 validates 5 shape/CFG cases, 2 DiT+CFM trajectories, all 22 Transformer block outputs on independent fixed synthetic PyTorch weights, and a deterministic noise-injection API. Source fixture generator is `tests/tools/generate_full_dit_fixture.py`. Next close complete mel/Vocos waveform comparison using explicit noise and small synthesizable checkpoints, then native Chinese segmentation. Avoid adding parallel PyTorch runtime to production; Python is test oracle only.
