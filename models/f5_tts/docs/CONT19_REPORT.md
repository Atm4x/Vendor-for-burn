# F5-TTS native Rust — CONT19 functional report

Parent: `codex/f5-tts-rust-cont18` at `67944102506f350e6e61f4bba8a3c12b81dd656f`.
Branch: `codex/f5-tts-rust-cont19`.

## Priority order

1. Original F5-TTS and Cross-Lingual function coverage.
2. Local differential/numerical comparison against the original Python/PyTorch code.
3. TensorRT hardware operability, real checkpoints, Windows GPU runtime and speed.

No ONNX. No CubeCL fallback inside the TensorRT runtime. The implementation changes are confined to `models/f5_tts`; one isolated CI workflow is added for verification.

## Feature additions

- Exposed `CfmSampleMode { no_ref_audio, duplicate_test, t_inter }`; the optional `t_inter` default is **0.1**, matching the original `CFM.sample()`.
- For `duplicate_test`, the reference condition is shifted to later frames and mixed into the initial Gaussian noise. A nonzero start bypasses EPSS and truncates the remaining integration step count using Python-compatible `f64` arithmetic.
- For `no_ref_audio`, the diffusion condition and final preserved condition are zeroed, but the duplicate initialization can still use the original reference mel just as the original Python code does.
- Both modes and both `OdeMethod::{Euler,Midpoint}` are exposed from the CubeCL synthesis path and through the TensorRT warmed session / shape bank / persistent worker.
- True GPU-resident TensorRT midpoint: first speed evaluation generates a midpoint buffer; second evaluation uses the midpoint as the model state, but adds the update to the original base state. It is not two successive Euler steps.
- Speech editing with midpoint is available through `TrtModelWorker::edit_speech_with_method`, including arbitrary frame preservation.
- Native PCM16 silence removal implements the optional generated-audio `pydub.split_on_silence(min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10)` behavior. Enabled only by opt-in `synthesize_wav_with_silence_removal`.
- Original WAV quantization is retained; the postprocessor writes PCM16 without a second float quantization.

## Numerical tests and verification

- Previous local test before container resets: 6/6 CFM-mode cases, 4,800 FP32 samples compared against an independent PyTorch CPU reference. **Those temporary files were lost during the container reset and are not part of the published regression fixtures; they should be regenerated for independent reproducibility.**
- Earlier six local pydub differential tests: 432,576 PCM16 samples matched bit-exactly. The published `postprocess.rs` contains six exact PCM16 fixtures.
- TensorRT mock includes a state-dependent affine velocity test of true midpoint versus an explicitly calculated PyTorch float32 reference. It also tests `duplicate_test + no_ref_audio + t_inter` on the warmed session, without GPU download of intermediate mel.
- GitHub Actions run [44809314188](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44809314188) completed SUCCESS: 59/59 default Rust unit tests, 88/88 TensorRT mock unit tests (2 hardware tests intentionally ignored), CubeCL CPU compile-check PASS. This was the previous revision, before the latest worker midpoint forwarding.
- Final head verification: [44809707498](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44809707498). This checks all worker routing modifications, formatting, default Rust, TensorRT mock and CubeCL CPU. Consult its actual conclusion before treating final-head gates as PASS.
- True NVIDIA TensorRT engine build, numeric TRT-vs-PyTorch and Windows driver/DLL validation were deliberately deferred (priority 3).

## Still missing from original inference feature coverage

- **BigVGAN vocoder alternative:** the source `utils_infer.py` supports both `vocos` and `bigvgan`; this native crate currently implements Vocos only. BigVGAN requires an independent vocoder graph and compatible mel preprocessing, not a hidden route through Vocos.
- Production-weight parity and exhaustive domain-specific speech/vocabulary edge cases are not established. Do not claim 100% feature parity from the current test count alone.
- The real TensorRT runtime remains unvalidated on NVIDIA hardware.

## Next priority

1. Implement optional BigVGAN v2 24kHz/100-band inference as an independent native Rust vocoder backend and its mel configuration, without changing the existing Vocos numerics.
2. Add independent local Python/PyTorch numeric fixtures for BigVGAN convolutional/activation layers; preserve existing DiT/CFM/Vocos parity.
3. Only after functionality and numerical parity, run actual TensorRT NVIDIA engine builds and hardware checks.
