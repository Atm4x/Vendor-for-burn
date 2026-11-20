# F5-TTS native Rust — CONT21 functional inference closure

Parent branch: `codex/f5-tts-rust-cont20` at `2d138f597ab40c582b6817cc8cec0ee39d86fd73`.
Branch: `codex/f5-tts-rust-cont21`.

## Priorities and limits

1. Cover the original Python F5-TTS inference features.
2. Check independent numerical behavior locally against the original Python/PyTorch implementations.
3. Defer NVIDIA / TensorRT-RTX hardware operation and actual checkpoint performance tests.

No ONNX, no CubeCL fallback inside the TensorRT execution path, and no changes to other models or legacy modules. The original F5-TTS full training, ASR, GUI and Gradio service implementations are not ported as parts of this inference-focused native crate.

## Functional additions

- Streaming output (`infer_batch_process(streaming=True, chunk_size=...)`) now has native and TensorRT public APIs. Frames are emitted in order, using a configurable nonzero PCM chunk size and a 24 kHz sample rate, without crossfade, exactly as the original streaming branch does. PCM consumers can abort streaming with a returned error. Generated text batches keep their common seeded RNG stream rather than restarting it for each batch.
- Full optional inference return (`waveform`, `sample_rate`, `spectrogram`) now exists for native standard F5 and TensorRT. Generated spectrograms are returned in channel-major `[100, frames]` layout, with multiple text batches concatenated along the frame axis; PCM retains the original overlap/crossfade behavior in non-streaming mode.
- TensorRT spectrogram download is explicitly opt-in. The standard PCM-only inference path does not incur an extra device-to-host copy. The generated mel stays in the session until requested.
- Both features support the existing `CfmSampleMode` (`no_ref_audio`, `duplicate_test`, `t_inter`) and Euler/Midpoint pathways; previous Vocos/BigVGAN selection is preserved.

## Verification

- Ordinary Rust unit tests: **63 passed, 0 failed**.
- TensorRT mock unit tests: **96 passed, 0 failed**, with **2 actual hardware tests deliberately ignored**.
- CubeCL CPU `cargo check`: **PASS** with supplied Tracel LLVM 23.1.0-4, without installer bypass (`DOCS_RS` not set).
- CubeCL CPU feature unit tests: **83 passed, 0 failed**; includes independent PyTorch DiT/CFM/Vocos numerical fixtures, and all new API shape/chunking tests.
- `git diff --check`: **PASS**.
- Independent full NVIDIA BigVGAN waveform-vs-PyTorch execution and actual production model/weights tests remain **not established**; they are numerical / operational verification gaps, not claimed as functionally complete proofs.

## Scope

Source modifications are restricted to:

- `models/f5_tts/src/standard_chunking.rs`
- `models/f5_tts/src/pipeline.rs`
- `models/f5_tts/src/model.rs`
- `models/f5_tts/src/backend/trtx_session.rs`
- `models/f5_tts/src/backend/trtx_text.rs`
- `models/f5_tts/src/backend/trtx_worker.rs`

The isolated CONT21 Git Bridge bootstrap workflow is removed after importing the exact previous branch into the local development environment. No other model or obsolete/legacy code is touched.
