# GigaAM Rust / Burn CubeCL CUDA — Windows x64

This is the native **command-line** inference executable, `gigaam-rs.exe`, not a graphical UI. It uses Burn 0.22.1 with CubeCL CUDA. Flex and ONNX Runtime are not included in this executable.

## Requirements

- Windows x64 and a supported NVIDIA GPU with current drivers.
- CUDA runtime libraries accessible to the process. The build is pinned to CUDA 12.9 (`CUDARC_CUDA_VERSION=12090`) for runtime NVRTC loading. If runtime DLL loading fails, install the appropriate CUDA 12.9 toolkit/runtime and ensure `bin` is on `PATH`.
- PowerShell 5.1 or 7 for scripts (the build runner validates the syntax with PowerShell 7).
- GigaAM checkpoint files for real speech recognition, downloaded separately under their original license. The package contains **only small synthetic weights** for smoke tests; they do not recognize real speech.

## Start here

Extract the **entire** Actions ZIP into a directory. Open PowerShell there:

```powershell
.\Test-GigaAM.ps1 -Device cuda
```

The test exercises CTC, RNN-T, SSL, Emo and CTC parity with bundled synthetic checkpoint files. Output and logs are written under `results\`. A successful run verifies GPU runtime execution but **does not** establish recognition accuracy with official weights.

For a real GigaAM model and your own audio:

```powershell
.\Run-GigaAM.ps1 -Mode transcribe -ModelPath 'D:\Models\GigaAM\v3_ctc.ckpt' -AudioPath 'D:\Audio\MitaKind.wav' -Device cuda
```

To run the GPU smoke suite and your official ASR checkpoint in the same pass:

```powershell
.\Test-GigaAM.ps1 -Device cuda -RealModelPath 'D:\Models\GigaAM\v3_ctc.ckpt' -RealAudioPath 'D:\Audio\MitaKind.wav'
```

For transcription with timestamps:

```powershell
.\Run-GigaAM.ps1 -Mode transcribe -ModelPath 'D:\Models\GigaAM\v3_ctc.ckpt' -AudioPath 'D:\Audio\test.wav' -WordTimestamps
```

For the other heads:

```powershell
.\Run-GigaAM.ps1 -Mode embed -ModelPath 'D:\Models\GigaAM\ssl' -AudioPath 'D:\Audio\test.wav' -OutputPath '.\encoder.f32'
.\Run-GigaAM.ps1 -Mode emotion -ModelPath 'D:\Models\GigaAM\emo' -AudioPath 'D:\Audio\test.wav'
.\Run-GigaAM.ps1 -Mode probe-audio -AudioPath 'D:\Audio\MitaKind.wav'
```

To run only GPU-independent preprocessing/checkpoint checks:

```powershell
.\Test-GigaAM.ps1 -SkipGpu
```

This should **not** be interpreted as a CubeCL GPU pass.

## Punctuation and normalized text (E2E models)

The official `v3_ctc` and `v3_rnnt` checkpoints predict plain transcription text; the CTC character vocabulary does not contain punctuation. For punctuation and text normalization use **`v3_e2e_ctc`** or **`v3_e2e_rnnt`** instead. Their native Rust inference path requires the matching SentencePiece model stored alongside the checkpoint:

- `v3_e2e_ctc.ckpt` + `v3_e2e_ctc_tokenizer.model`
- `v3_e2e_rnnt.ckpt` + `v3_e2e_rnnt_tokenizer.model`

Both files in each pair must have the exact original names; they are not included in the Windows build artifact. To check one E2E model, use:

```powershell
.\Run-GigaAM.ps1 -Mode transcribe -Device cuda -ModelPath '.\v3_e2e_ctc.ckpt' -AudioPath '.\MitaKind.wav' -Profile
```

To compare original CTC against both official E2E variants in the same results archive:

```powershell
.\Test-GigaAM.ps1 -Device cuda `
    -RealModelPath '.\v3_ctc.ckpt' -RealAudioPath '.\MitaKind.wav' `
    -E2ECtcModelPath '.\v3_e2e_ctc.ckpt' `
    -E2ERnntModelPath '.\v3_e2e_rnnt.ckpt'
```

You may specify `-E2EAudioPath` to test a different recording. The suite records `e2e-ctc-config.log`, `e2e-ctc-asr.log`, `e2e-rnnt-config.log`, `e2e-rnnt-asr.log`, and corresponding `summary.json` entries. An E2E test fails if the process returns an error **or** no nonempty transcription line is produced; it does not require every recording to contain punctuation, since that is a model prediction, not an unconditional guarantee.

The tester now reads native stdout/stderr as UTF-8 directly using .NET process streams and writes logs as UTF-8. This avoids Russian text corruption in Windows PowerShell 5.1, including when `gigaam-rs.exe` reports an error.

### Per-stage timing

Add `-Profile` to `Run-GigaAM.ps1` (or `--profile` to `gigaam-rs.exe transcribe`). The CLI reports `model_load_s`, `audio_decode_s`, `transcribe_s`, `audio_s` and `rtf` to stderr. `rtf` is **transcription time divided by audio duration**, excluding model loading and file decoding; transcription time still includes CPU mel preprocessing and GPU transfers, so it is not pure kernel time.

## Reporting failures

Share `results/summary.json`, relevant `results/*.log`, `results/nvidia-smi.txt` when available, and the included `BUILD_STATUS.txt`. If the CLI says the checkpoint configuration is unsupported, capture `inspect-config` and `inspect-checkpoint` output. Never paste private model data or audio if you do not intend to share it.

`gigaam-rs.exe` is a CUDA-only build. `-Device cpu` or `-Device flex` requires a separately compiled backend; it does not silently fall back. GitHub-hosted Windows runners compile the executable but do not provide an NVIDIA GPU for inference verification.

The CTC test suite includes `ctc-v3-mel`, a tiny synthetic checkpoint using official v3 STFT geometry (`n_fft=320`, `center=false`) for a genuine CubeCL CUDA smoke test. With `-RealModelPath`, `Test-GigaAM.ps1` also saves `real-model-config.log` before attempting ASR; errors from a native executable are captured and added to `summary.json` instead of terminating the PowerShell process prematurely. Official checkpoint validation is separate from the synthetic test suite.
