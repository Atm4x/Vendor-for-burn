# Stage 25 — CPU PyTorch/Burn comparison (preliminary)

Environment: same shared Linux x86_64 runtime (4 CPU quota), real private RVC v2 `CrazyMita.pth`, FP32, 24 model frames (9600 output samples at 40 kHz), no HuBERT/RMVPE/retrieval in this microbenchmark. Torch 2.10 CPU eager, Burn 0.22.1 Flex SIMD/Rayon release. PyTorch `torch.set_num_threads(4)`, Rayon 4 threads. One warm-up followed by five timed iterations. Input: `phone=sin(arange(24*768)*0.013)*0.1`, pitch 128, F0 175 Hz, speaker zero. Both use TextEncoder → reverse Flow → NSF; latent = mean, skipping stochastic posterior sampling. Exact NSF noise sources differ, so this is an **operation-level performance comparison**, not end-to-end numerical parity.

Medians of five measurements:

| Component | PyTorch eager (ms) | Burn Flex release (ms) | Burn/Torch |
|---|---:|---:|---:|
| TextEncoder | 13.703 | 16.986 | 1.24x |
| Reverse Flow | 13.625 | 14.124 | 1.04x |
| NSF-HiFiGAN + F0 source | 132.743 | 208.615 | 1.57x |
| Total | 159.814 | 237.452 | 1.49x |

In this test, PyTorch is **faster** overall. The main gap is NSF. Results can vary with CPU contention and input duration. The times shown exclude checkpoint loading, HuBERT, RMVPE, FAISS, audio I/O and normalization. The original Python wrapper has other overheads; this benchmark avoids including those overheads when comparing neural kernels. Do not extrapolate these numbers to GPUs.

`benchmark_pytorch_synth.py` needs the private model and the extracted original `tts-with-rvc` Python source:

`python benchmark_pytorch_synth.py --original-root PATH_TO_ORIGINAL --checkpoint PATH_TO_PTH --frames 24 --iterations 5 --threads 4`

Rust/Burn benchmark:

`cargo run --release --offline -p tts-with-rvc-rs --bin bench_rvc_synth -- PATH_TO_PTH 24 5`

Subsequent work: compare HuBERT and RMVPE on identical audio windows, test longer shapes and B>1, and profile NSF convolution/transposed-convolution/resblocks on each backend. The full original Python wrapper was not runnable without fairseq and other dependencies in this environment.
