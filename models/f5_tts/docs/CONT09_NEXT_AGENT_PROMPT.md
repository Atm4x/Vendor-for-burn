# F5-TTS CONT09 → CONT10

Continue at `models/f5_tts` based on `codex/f5-tts-rust-cont09` in Atm4x/Vendor-for-burn. Read `CONT09_REPORT.md` and the local sources. Work locally; publish a separate branch via GitHub Connector at the end. Do not touch Fish Speech, RVC, Legacy, or maryml. Keep functionality (main) and workability (secondary) progress separate. No ONNX.

The previous Burn blockage is resolved. GitHub Actions `44741264056`, artifact ID `14631086557`, provides Cargo.lock and complete offline vendor archive (~103 MB ZIP, 867 MB unpacked). Verify SHA-256 before use. The tar.zst contains `.cargo/config.toml`; use paths appropriate for the actual extraction location. LLVM `linux-x64.tar.xz` (23.1.0-4) is provided by the user. The tracel-llvm-bundler build.rs requires its matching installed LLVM directory and sentinel.

CONT09 numerical coverage: full *reduced* DiT PyTorch parity 1/1 PASS with 2-block model, 35/35 offline Rust tests, 7/7 Cross-Lingual library tests, including 2,200 Python duration cases and 128 NumPy cross-fade cases. Cross-Lingual timing and fade settings use f64 to preserve Python float semantics. Do not claim large-model parity, CUDA runtime or TRT-RTX. Functional estimate 74%, workability 3/7=43%.

Next focus: additional full DiT numerical fixtures with varying sequence lengths and CFG, reduced DiT+CFM end-to-end parity using deterministic input noise, and direct Chinese pypinyin/rjieba differential corpora. Avoid spending cycles on already solved vendor acquisition or hardware testing unless it is needed to unblock numerical work.
