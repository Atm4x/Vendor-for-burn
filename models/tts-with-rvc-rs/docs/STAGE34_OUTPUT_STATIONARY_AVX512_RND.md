# Stage 34 — output-stationary AVX-512 Conv1d without GEMM

Status: independent CPU R&D, not enabled in production RVC. Repository: `Atm4x/Vendor-for-burn`. All checkpoint / audio data were kept private and excluded from version control.

## Goal and observation

The Stage 33 segmented Winograd K11 kernel reduced scalar multiplication counts, but its four matrix products and input packaging dominated, causing a regression. Stage 34 investigates a *single-pass register-resident direct cross-correlation kernel* optimized for the shapes present in the trained `CrazyMita.pth`. It does not call `gemm`, construct `im2col`, materialize a phase output, or dispatch to Conv3d. This is a model-focused CPU experiment, not a universal replacement for Burn's Conv1d.

The finite-FP32 operator is

`Y[o,t] = sum_(c=0..Ci-1) sum_(j=0..K-1) W[o,c,j] * X[c,t+j*D-P]`, with zero padding outside the input and `P=(K-1)*D/2` (odd K, stride 1).

Choose a block of `V` output channels and `T` output positions. All `(o,t)` in that `V×T` block share the corresponding input scalar for a fixed `(c,j,t)` and the same weight vector for a fixed `(c,j,o)` across every temporal position. Packing the immutable weights as `[out_channel_block, input_channel, tap, output_lane]` makes vector loads contiguous. The kernel holds `T` independent SIMD accumulators in registers, broadcasting each input sample once across `V` output channels and writing each output once after the complete reduction. This is an exact traversal of the convolution sum over real arithmetic, with FP32/FMA rounding differences accepted by a bounded error test.

**Reuse accounting:** compared to a direct elementwise loop, per-block input scalar loads can be reused across `V` output channels, and each vector of weights reused across `T` output positions. Arithmetic complexity remains `O(Cout * Cin * K * L)`; the improvement is memory traffic and SIMD utilization, not fewer mathematical multiplications. This is not a new asymptotic convolution algorithm.

## Attempted microkernel layouts

All tested in optimized single-CPU-thread Rust (AVX2/FMA or AVX-512F/FMA), on the same real model weights/activations. A single `gemm`-based Conv1d implementation from the vendored Burn gemm crate supplies a same-executable baseline. The weight packing is built once outside timed forwards; the execution time **includes** input zero-padding, output allocation, and the actual arithmetic. A separate reference PyTorch 2.10.0 CPU `conv1d` is also measured on a single pinned core with MKLDNN enabled.

- Output-position SIMD, transformed K3 Winograd: slower than GEMM (about 2.6 ms versus 1.0 ms on early K3). The existing Stage 33 Winograd+GEMM kernel remains a better CPU option for K3.
- Output-position SIMD, four output channels: slower for all three measured geometries.
- Output-channel SIMD, AVX2 eight channels: materially better but still generally slower than GEMM.
- Output-channel SIMD, AVX2 16 channels × 4 time positions: approaches parity.
- Output-channel SIMD, AVX-512F 16 or 32 channels × 8 time positions: the first configuration family faster than reference GEMM for all three measured real layers.

## Repeated same-process A/B measurements

Single CPU pinned via `taskset`; three independent sessions, each reports the median of nine measured forwards after warm-up. The three sessions were run sequentially. Numbers below are the medians of the three session medians. Values include output allocation and zero-padding. Benchmark instrumentation is research-grade; timings are subject to shared-host contention.

| Real activation shape | GEMM baseline (ms) | Direct AVX-512 (ms) | Speedup | Winning register tile |
|---|---:|---:|---:|---|
| K3/D1, Ci=Co=256, L=240 | 1.0257 | 0.8930 | 1.149x | 16×8 |
| K11/D5, Ci=Co=128, L=2400 | 9.7805 | 8.7840 | 1.113x | 16×8 |
| K11/D1, Ci=Co=32, L=9600 | 2.6570 | 2.2447 | 1.184x | 32×8 |

For context, the separate pinned single-thread PyTorch 2.10.0 runs on the same real input tensors measured ~0.9747, ~8.4562, and ~2.0684 ms respectively (15 timed runs after warm-up). These are not interleaved cross-framework measurements, so only the within-Rust A/B speedup is an acceptance-grade claim. In the cross-process figures, the new kernel is within roughly 10% of PyTorch on the three samples; it is not demonstrated faster overall.

## Correctness evidence

- Local direct Rust test passed 2,268 combinations of `(Ci,Co,L,K,D)`, including input length zero, odd/even output length, output channels not divisible by 16/32, kernels 1/3/5/11, and dilations 1/2/5. Reference: the previous direct GEMM computation. Both AVX-512 16-channel and 32-channel variants checked. Condition: absolute error `<= max(1e-5, abs(reference)*1e-5)`.
- Saved PyTorch activations from the private checkpoint: maximum difference between direct AVX-512 and GEMM ~6.68e-6 (K3), ~1.526e-5 (K11/D5), ~4.77e-6 (K11/D1). Includes 32-channel variant and no bias, which is accounted for only when comparing to saved PyTorch outputs.
- `f16`, `bf16`, grouped or transposed convolutions, non-finite inputs/weights, non-unit strides, and general padding have **not** been validated. Do not dispatch to this kernel for those cases. The example has no runtime dispatch for other CPU architectures: explicit AVX-512F + FMA support is required.

## CUDA / multi-backend implication

The output-stationary register tiling is independent of the CPU ISA: the corresponding GPU plan is to assign a warp/block to a tile of output channels and time positions, make weights contiguous over output lanes, share each input scalar across output lanes, and accumulate the entire output tile before one write. On CUDA, register pressure, global-memory coalescing, shared-memory staging, and tensor-core alternatives must be evaluated. **No CUDA/WGPU kernel was implemented or measured**. GPU speedups cannot be inferred from AVX-512 timings. A generic native `GEMM` may remain superior for other geometries; the goal here is the user's specific trained model, not universal fallback routing.

## Reproduction (offline)

Source: `models/tts-with-rvc-rs/research/nsf_fused_register_conv.rs` (requires Rust 1.98.1 and the vendored `gemm` rlib for its benchmark/reference); Python reference: `research/profile_torch_nsf_layers.py`. Compile the Rust file with `rustc --edition 2021 -O -C target-cpu=native -L dependency=<target/release/deps> --extern gemm=<matching libgemm.rlib> ...` and run with the directory containing the three private activation cases. The file is deliberately outside Cargo targets and only holds R&D drivers, not an automatically enabled backend path.

## Promotion gates

1. Move the best output-channel register kernel to a separate backend-owned immutable packed-weight plan, created once when loading the model and invalidated with weight changes.
2. Add feature-gated CPU routing only for exactly supported shapes, dtypes and ISA without silent CPU/GPU fallback; preserve canonical weights, validate padding/NaN/FP16 semantic boundaries.
3. Compare the full NSF decoder against the old CPU implementation and original PyTorch with identical latent activations and deterministic NSF noise; compare `B=1`, `B=2`, 24/128-frame inputs and the full short WAV sample.
4. Implement a proper CUDA kernel based on the same output-stationary dataflow, and compare directly with cuDNN on actual NVIDIA hardware. Do not assume the register-tiled CPU code is sufficient for CUDA.
5. Only after all numerical and full-model checks pass, consider promotion to the RVC inference backend. No production files were modified in this stage.
