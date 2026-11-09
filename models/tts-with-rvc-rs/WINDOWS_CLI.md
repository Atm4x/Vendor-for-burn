# RVC Rust/Burn CLI for Windows (CPU)

This package contains a native Windows x64 executable. Python and PyTorch are **not required to run it**.

Place the following private files anywhere on your machine (do not upload them to GitHub):
- RVC v2 `.pth` voice checkpoint
- `hubert_base.pt`
- `rmvpe.pt`
- optional FAISS IVF Flat `.index`
- one or more input WAV files

From PowerShell, in the folder containing the executable:

```powershell
.\tts-with-rvc-rs.exe doctor
.\tts-with-rvc-rs.exe convert --model "C:\models\CrazyMita.pth" --hubert "C:\models\hubert_base.pt" --rmvpe "C:\models\rmvpe.pt" --index "C:\models\CrazyMita.index" --input "C:\audio\source.wav" --output "C:\audio\converted.wav" --batch-size 2 --pitch 0 --index-rate 0.75
```

For processing multiple WAV files in one process without reloading weights, use `convert-batch --manifest jobs.json` instead of `convert --input/--output`, where `jobs.json` contains:

```json
[
  {"input":"C:\\audio\\first.wav", "output":"C:\\audio\\first-rvc.wav"},
  {"input":"C:\\audio\\second.wav", "output":"C:\\audio\\second-rvc.wav", "pitch_shift_semitones":2}
]
```

Common controls: `--batch-size 1..32`, `--pitch N`, `--index-rate 0..1`, `--rms-mix 0..1`, and chunk configuration (`--core-frames`, `--overlap-frames`, `--context-frames`). Run `--help` for the full list.

The executable is **CPU-only** (Burn Flex with SIMD/Rayon). This artifact does not contain CUDA kernels. `--device cuda` deliberately rejects the request; it will **not** silently use CPU. The NVIDIA display driver, CUDA runtime or CUDA Toolkit cannot convert this CPU-only artifact to a working CUDA binary. Separate CUDA backend integration, Windows dependency packaging and GPU tests are still required. Do not install CUDA solely to run this release.

You can benchmark your hardware by running the same conversion twice, with batch sizes `1` and `2`. Record warmup and model-load time separately from the inference stage. First-time runs may be slower.
