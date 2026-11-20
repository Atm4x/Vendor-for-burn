# F5-TTS Rust — NEXT (CONT12)

Continue in a **new GitHub branch** based on the published CONT11 `models/f5_tts` only. User requires a clean independent architecture and two backends, CubeCL CUDA/WGPU first and native TensorRT-RTX separately; no ONNX, no `maryml` architecture copying. Do not touch Fish Speech/RVC/Legacy. Publish at cycle end through GitHub Connector; work locally between GitHub operations.

The two PyTorch-vs-CubeCL CPU end-to-end audio fixtures now cover uniform Euler and EPSS+sway 100-mel pathways, both exactly matching PCM16 (CONT11_REPORT). 22 independent Transformer layers passed on narrow synthetic DiT in CONT10. Next priority is **functional coverage and numerical equivalence**: expand exact text/Chinese segmentation or add production-width forward numerical probes without depending on pretrained model download. TensorRT-RTX execution backend remains the largest missing architectural feature, but do not replace CubeCL implementation with it.

User wants **two progress bars**, functionality most important, workability second. CONT11: functionality estimated 78%, workability 43% (3/7). Do not raise workability for `cargo check` alone.

Cargo vendor artifact remains from GitHub Actions run 44741264056 (`f5-tts-cargo-vendor-cont08`); Rust 1.100.0 and LLVM 23.1.0 are available. Avoid repeated environment rebuilds or GitHub GET flooding; use local vendor and verified bundle/branch. Separate Python tooling is an oracle, not runtime.
