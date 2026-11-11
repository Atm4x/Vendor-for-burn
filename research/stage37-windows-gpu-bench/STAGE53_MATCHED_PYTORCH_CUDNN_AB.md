# Stage 53 — matched synthetic NSF: PyTorch/cuDNN vs Burn/CubeCL (Windows CUDA)

**Date:** 2026-11-11. **Scope:** Experimental PR #2, not production. **Result:** benchmark sources and Windows packaging prepared; RTX 4060 measurements pending user-side execution.

## Why Stage 37 cannot be compared directly to Stage 51/52

Stage 37 measured three **individual** Conv1d operators at shapes K3(256→256,240), K11/D5(128→128,2400) and K11(32→32,9600). Its strict FP32 PyTorch/cuDNN reference timings were 0.18395, 0.43870 and 0.28370 ms respectively, whereas experimental CUDA Packed measured 0.20160, 0.85250 and 0.52540 ms. CubeCL was about 1.10×, 1.94× and 1.85× slower, *for those individual convolutions*. Stage 51/52 measured a **complete small synthetic NSF decoder** (including ConvTranspose, condition, source, residual branches, activations and final tanh) with different shapes and a different test harness. Stage 51's 7.97× is the ratio of *the same* Rust ordinary forward vs CUDA Graph replay, not a Torch comparison.

## Stage 53 comparison contract

- Both paths read **the exact same** `synthetic_nsf_decoder.pth` and `nsf_decoder.json`. PyTorch uses `torch.load(..., weights_only=True)`. No actual user voice `.pth`, HuBERT or RMVPE needed.
- Independent PyTorch reference reproduces `NsfWaveformDecoder` in Rust: weight-normalized Conv1d and ConvTranspose1d, `conv_pre`, conditioning, harmonic-source `noise_convs`, 2 upsample stages, 2 ResBlock1 branches each with 3 convolution pairs and dilations `[1,3,5]`, zero-seeded residual sum, final leaky ReLU and tanh.
- Four changing input sequences, source waveforms and conditioning values are recreated for each frame size exactly as Stage 52 input generator. PyTorch uses normal eager as well as `torch.cuda.CUDAGraph` with stable GPU allocations, copying new values CPU→GPU before replay. Both use FP32 with TF32 disabled; cuDNN enabled and `benchmark=False`.
- Each process measures 64/256/1024 frames, minimum 4 dynamic steps; recommended 16, warmup 5. The runner executes PyTorch and Rust baseline/prepacked in forward and reverse process order.
- Both benchmarks separately report (1) eager/ordinary forward + CPU output readback, (2) Graph Replay + CPU output readback, and (3) host-to-GPU input refresh. Cross-backend summary also includes per-iteration refresh+graph latency.
- Rust Stage 52 has an opt-in `--dump-prefix`; exports FP32 output arrays for first four dynamic steps. PyTorch exports identical steps. Full tensor checks include maximum absolute, normalized and RMS errors; a normalized error threshold <= 2×10⁻⁴ is required. Internal Rust Graph/ordinary remains bitwise equal on hardware CUDA.
- No silent CPU baseline, software fallback or CUDA Graph capture failure may be reported as a speedup. The Windows Actions runner only packages tools, it does not run GPU inference.

## Run on RTX 4060

Download Actions artifact `rvc-nsf-stage53-pytorch-cudnn-vs-cubecl-windows-x64`. Unpack and run in a Python environment with CUDA-enabled PyTorch installed (the original Stage37 environment is acceptable):

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-NSF-Torch-Comparison.ps1 -Python python -Iterations 16 -Warmup 5 -Frames "64,256,1024"
```

For a specific venv, pass its executable path via `-Python`. Output is `NSF_TORCH_CUDNN_STAGE53_RESULTS.zip`; `stage53_results/summary.json` includes ratios, parity for both rounds, raw iteration timings, TF32 flags and device metadata. A ratio `torch_over_rust_prepacked_graph_ratio` > 1 means **Rust Graph is faster**, < 1 means **PyTorch Graph is faster**.

## Verification status

- PyTorch CPU fixture parity (synthetic original input, 4 frames): max abs FP32 error **2.98e-8** vs checked-in expected waveform, proving the independent reconstruction at the fixture size.
- PyTorch CPU smoke with changing frames: **PASS** (not a GPU performance measurement).
- Rust 1.98.1 CPU `cargo check --offline --locked` and CPU debug build: **PASS**.
- Rust 1.98.1 CUDA `cargo check --offline --locked`: **PASS**.
- Cross-language CPU smoke (four changing inputs, 4 frames, full 16-element FP32 output): all four samples **PASS**, maximum absolute difference **1.77e-8**. Rust CPU software replay vs ordinary forward: bitwise equality; **not** hardware CUDA Graph.
- CUDA runtime and end-to-end Windows results: **not measured yet**.

## Limitations

This is a small fixed-architecture synthetic NSF, **not** real RVC end-to-end. The two runtimes may use different memory allocation and host integration; it is meaningful to compare host+readback latency but the result does not isolate pure GPU kernel execution. CUDA Graph capture cost is recorded separately; it needs sufficient reuse to amortize. This experiment does not compare `torch.compile`/TorchInductor or TensorRT, and does not validate user voice quality. Stage 52's full dynamic-input parity must be established on real NVIDIA hardware first.
