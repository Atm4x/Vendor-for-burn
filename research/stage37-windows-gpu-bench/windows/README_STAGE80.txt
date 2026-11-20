RVC Stage80 — FIXED original PyTorch CUDA vs Rust (Stage78/77 EXE) benchmark

Extract this ZIP INTO the folder which already contains tts-with-rvc-rs.exe and embed/.
It does NOT overwrite any existing Stage77/78/79 files or weights.

Default Python CUDA: D:\Projects\tts-with-rvc\.venv\Scripts\python.exe
Original Python reference: ./reference_upstream/tts_with_rvc/ (unchanged source from user-provided tts-with-rvc-main (2).zip)
Config is derived using the exact reference API. The previous Stage79 local Config AttributeError is eliminated.
All five original file fingerprints must match the exact uploaded source. Do not point to locally modified D:\Projects\tts-with-rvc as the original-root, otherwise the comparison will fail before benchmarking.

Run from the extracted folder:
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage80.ps1 -Device cuda -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -WarmRuns 2 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -Modes all

Default Rust grid is 300:2,600:1,900:1; change with -Grid "300:2" for a fast smoke test.
Output: RVC_STAGE80_ORIGINAL_COMPARISON_RESULTS.zip, with summary.json and per-case stdout/stderr, model timings, and waveforms.

The PyTorch reference uses the original source and all local model weights; Torch dependencies from your existing Python .venv are loaded normally. PyTorch 2.6+ compatibility uses torch.load(weights_only=False) solely for local, trusted model checkpoints. Do not use untrusted checkpoints.

Stage78 batched TF32x2 was marginally slower on provided RTX4060 results; not promoted. Stage79 Torch FP32/FP16 comparison failed to start due to locally modified Python source. Stage80 corrects the benchmark harness only; not a new GPU speed result.
