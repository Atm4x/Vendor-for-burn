# Stage 85 — nonlinear NSF operator fusion R&D (Rust 1.100.0)

Parent: local Stage84 `a9b5d58e`; previous Stage83's attention and Stage84's tiled decoder remain opt-in, not modified here. This is not a production throughput claim.

## Measured motivation and scope

Stage68 synchronized RTX4060 profile for a ~90-second WAV: NSF 5.164s of 14.014s attributed model time; HuBERT 3.109s; RMVPE 3.068s. The measured value is prior-stage diagnostic evidence; it is not a Stage85 measurement. Stage84 tiling addresses peak memory but can add work. Stage85 targets the nonlinear, high-sample-rate NSF residual execution directly.

Four upsampling stages × three parallel ResBlock branches × three sequential convolution pairs = **36 residual pairs per NSF decode call** for the checked RVC-v2 configuration. At 32 overlapping chunks this means 1152 pair executions. In the original source each pair dispatches five distinct tensor operations: LeakyReLU, Conv1d, LeakyReLU, Conv1d, residual addition. The experimental path evaluates the same mathematical expression with two CubeCL kernels, each handling activation as an input epilogue and the second adding the residual as an output epilogue. This can remove up to **3 × 1152 = 3456 separate tensor operations/kernels** per full 32-chunk run, plus their stand-alone activation tensor allocations. These are structural upper bounds, **not measured CUDA launches or runtime savings**. CubeCL may fuse its own operations, and host/kernel dispatch details vary by backend.

Mathematically, for each ResBlock pair with zero-padded same-size convolutions,

`H = Conv1(LeakyReLU(X,0.1), W1, b1)`

`X' = X + Conv2(LeakyReLU(H,0.1), W2, b2)`.

The new implementation applies LeakyReLU to each source value while accumulating each convolution, adds bias after reduction and incorporates the residual in the second kernel. It preserves all three sequential pairs, three parallel branches, source generation, stochastic schedule, speaker conditioning, padding, dilation and weights. Floating-point reduction order and FMA policy can differ: **PCM16 bitwise equality is not assumed**.

## Engineering decisions

- `RVC_NSF_FUSED_PAIR_MAX_CHANNELS=64` enables only pairs whose channel count is at most 64. Earlier wide-channel pairs keep the established implicit GEMM; small high-rate stages are candidates. Only default scheduling is selected; diagnostic flush/sync modes keep the original operator sequence. Unset = entirely unchanged production path. Negative, zero or malformed values do not enable the experiment.
- Direct per-output CubeCL F32 FP32 kernel supports 1D dilated same-padded odd kernels, logical OIK weight shape with arbitrary strides (including temporal or prepacked views), optional bias and optional residual, and arbitrary batch/time dimensions. Checked host arithmetic rejects zero shapes, overflow and storage overrun. Unsupported backend/dtype fails explicitly instead of silently falling back for an opted-in pair.
- The fused kernels use direct scalar channel/tap accumulation. This can be **slower** than CUDA implicit GEMM for high channel counts because tensor cores and GEMM tiling may have much better throughput. The channel cutoff is a test parameter, not a proven optimum.
- No vendor, legacy, Python/ONNX, F0 threshold or model weight modifications. Rust 1.100.0's stable allocator API is deliberately not used: the relevant allocations are device tensors, not host `Vec` instances.

## Verified local evidence (2026-11-14)

- Rust 1.100.0, standalone `rustc --test` run of `nsf_fused_pair_math.rs`: **2 PASS / 0 FAIL**, covering six nonlinear pairs and separate boundary cases.
- `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu,flex,cuda --lib --bin tts-with-rvc-rs --test nsf_fused_pair_math`: **PASS** (type checking only).
- Additional `cargo check` with `--features cpu,flex,cuda,wgpu --lib --bin tts-with-rvc-rs`: **PASS** (type checking, 26.72s on warm dependency cache).
- No CUDA GPU is available in this container. CUDA JIT, actual execution of the fused operator on CubeCL and the full-WAV numerical/throughput comparison are **not performed**.
- No Windows build or Windows PowerShell parser run was completed in this environment. CI workflow supplied but not executed yet.

## Verification gates

1. Pure Rust mathematical comparison on six tensor configurations including positive/negative activation regions, batch size 2, kernels 1/3/5, dilations 1/2/3/9, boundaries and independent weights/biases: must pass.
2. Validate CubeCL/CUDA Rust type-check and invoke generated kernels on real CUDA before considering numerical parity established. Compiler-only check is not GPU kernel validation.
3. Full-WAV ABBA `baseline_1, fused_1, fused_2, baseline_2` under the same input and weights. Compare PCM16 RMSE, per-mode warm medians, independent baseline drift, GPU clocks and JIT cold start. Reject incomplete comparisons; require no voiced/unvoiced changes across source chunks.
4. If fusion is slower than the baseline, do **not** tune more environment-variable combinations as the next default action. Profile kernel occupancy, memory traffic and GEMM utilization; either implement truly matrix-tiled fused convolution or drop the direct-scalar approach.

## Expected long-range follow-on

The major improvement is not an arbitrarily high count of fusion flags. A dedicated NSF execution compiler should match the nonlinear graph to several implementation classes (direct fused low-C, tensor-core tiled high-C, transposed polyphase) with a small offline shape microbenchmark and retain only paths that improve full-WAV latency within an accepted numerical error. This iteration introduces and tests the first low-C execution class. It does **not** provide a generalized compiler yet.
