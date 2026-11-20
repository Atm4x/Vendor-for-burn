# F5-TTS CONT17 → CONT18

Start from published `codex/f5-tts-rust-cont17`. Never modify legacy/obsolete, unrelated models or vendor packages. Do not reopen MaryML as an implementation source. No ONNX.

Priority order from user: **(1) functional feature coverage, (2) locally reproducible Python/PyTorch numerical parity, (3) actual TensorRT-RTX/CUDA/Windows workability**. Do not raise functional coverage from test-coverage or compilation alone.

CONT17 added `src/backend/trtx_factory.rs` (direct per-shape engine builder with persistent thread owner), `src/backend/trtx_worker.rs` (one active model and hot reference voices), lazy LRU factory in `trtx_text.rs`, plus independent Python/NumPy multi-voice fixture. Current functional estimate 99%; workability 43%.

Before claiming 100% functionality, audit the exact original API and model variants for remaining features (e.g. rare Chinese tone sandhi and mixed-script chunk boundaries, active-voice switching and multi-reference request boundary); distinguish features deliberately handled by the application from capabilities required of this model library. Do not invent additional frameworks or unrelated functionality.

Then extend **numerical parity** across more multi-chunk, Chinese/English mixed scripts and 22-layer DiT + CFM input forms, using pinned fixture generation and identical weights/noise. The new original F5-TTS fixture generator uses `/mnt/data/F5-TTS-main (1).zip` only for offline generation, and commits its generated `.json` and `.f32le` files; test execution requires no Python.

Only when functionality is demonstrably complete and local numerical parity is adequate, proceed to *real hardware* work: build TRT CFM/Vocos/Speed engines with actual DLL paths, compare PCM/mel against PyTorch and measure warm cache behavior. Mock is not GPU validation. Do not run a cold CubeCL CPU build repeatedly just to obtain a green compilation badge.

Reproducible local checks use Rust 1.100.0 at `/mnt/data/f5-cont07/rust/bin`, Cargo offline vendor `/mnt/data/f5-cont12/vendor-union`, Cargo target `/mnt/data/f5-vendor-cont09/target`, Cargo `--config 'source.crates-io.replace-with="vendored-sources"' --config 'source.vendored-sources.directory="/mnt/data/f5-cont12/vendor-union"'`.
