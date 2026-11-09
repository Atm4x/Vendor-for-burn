# Stage 35 — promote and verify the best direct NSF CPU kernel

Current R&D: Stage 34 found a single-threaded AVX-512F output-stationary Conv1d that avoids GEMM, im2col, and per-tap writes. Saved real CrazyMita activations for K3 and two K11 cases. Three pinned-CPU A/B series measured ~1.15x (K3), ~1.11x (K11/D5), and ~1.18x (K11/D1) relative to the same-executable direct GEMM baseline. This is NOT yet full-model speedup and no GPU execution has happened.

Inputs: `research/nsf_fused_register_conv.rs`, `research/profile_torch_nsf_layers.py`, `docs/STAGE34_OUTPUT_STATIONARY_AVX512_RND.md` in the experimental repository. Real weights/audio remain local and private.

Goals:
- Build a clean immutable packed-weight plan for finite FP32 CPU Conv1d (odd K, stride 1, symmetric padding; feature-detect AVX-512F/FMA), and expose it only from the supported RVC NSF model boundary. Do not add a generic legacy fallback or change other legacy scripts.
- Integrate the output-stationary path into the actual NSF decoder and measure whole-NSF A/B after fixing identical synthetic noise and checkpoint. Validate 24 and 128 frames; compare with PyTorch reference.
- Record packed plan construction and amortization, throughput vs original Burn under 1/2/4 CPU threads, and resident forward performance. Be mindful that parallel residual branches and internal GEMM parallelism can contend, but do not treat Rayon as a fundamental arithmetic optimization.
- For CUDA, design and implement a dedicated CubeCL/CUDA equivalent and run it on actual NVIDIA hardware before claiming GPU competitiveness. AVX-512 code is CPU-only.
- Work sparsely: one Cargo invocation at a time, 2 build jobs, no workspace clean, no redundant GitHub GET. Publish new work in draft PR #2 via GitHub Connector only; never upload private .pth/.pt/.index/WAV.
