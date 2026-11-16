# Stage115 — RVC ResBlock transport R&D (Burn/CubeCL, source-only)

Base: `Atm4x/Vendor-for-burn` PR #2, Git Bridge HEAD `e41ce4b2b1d9c5c89b8c4f5cdef7e059817a03bc`. Local branch: `rnd/stage115-resblock-transport`. Rust: 1.100.0. **Experimental, opt-in only; no CUDA runtime benchmark or speedup claimed.**

## The measured evidence, separated from MAC analysis

- For 40k RVC at 128 latent frames, four ResBlock stages execute **56.812 billion MAC**. This is approximately 97.23% of generator MAC, not measured GPU time. Stage arithmetic: stage0 C256 L1280 = 10.570B; stage1 C128 L12800 = 26.424B; stage2 C64 L25600 = 13.212B; stage3 C32 L51200 = 6.606B. Across the three K3/K7/K11 branches, contributions are 3/21, 7/21, 11/21 of ResBlock MAC. Dilation affects addressing, not the MAC count for same-padding.
- The library's Stage102 NCU RTX4060 report captured **Conv2d**, not a ResBlock Conv1d: Direct dense3x3 kernel 456.42 µs, L1TEX LSU throughput 95.32%, SM compute 9.04%, L2 1.85%, DRAM 4.66%. It does **not** prove that ResBlocks have the same bottleneck.
- Stage109–112 NCU results examine polyphase **ConvTranspose1d**, not residual dense Conv1d; do not use their long-scoreboard or timings as if they measured ResBlocks.
- Stage37 isolated CubeCL Conv1d benchmarks on RTX4060 reported packed vs Direct speedups of 1.62×, 2.89×, and 1.63× on three synthetic ResBlock-like geometries. Against matched strict-FP32 cuDNN, packed still required 1.10–1.94× as much synchronized host time. These are not end-to-end or kernel-only results.
- Stage114 PyTorch FFT eager was slower than cuDNN in all four cases (1.22–3.36×), not evidence for custom CubeCL. Neither approach should be selected from FLOPs alone.

## Algorithm implemented

The current decoder normally evaluates each residual pair as separate LeakyReLU, Conv1d, LeakyReLU, Conv1d, add operations. The experimental mode `RVC_NSF_RESBLOCK_TRANSPORT=1`:

1. De-weight-normalizes each Conv1d **once** using the established checkpoint loader, and stores its logical OIK weight in a physically contiguous `[kernel, Cin, Cout]` layout with **Cout innermost**. The layout is converted on the host once at model load rather than repeatedly copied on GPU for each call.
2. Uses a cross-backend CubeCL kernel with two adjacent output positions per invocation, vectorized over output channels. The kernel combines input LeakyReLU with the Conv1d reduction; the second convolution fuses residual addition after its bias. The residual input is converted to contiguous NTC only when its strides require it; later kernels emit NTC storage with logical NCT views.
3. Preserves the existing six Conv1d operations per residual block and all residual branches. No FFT approximation, model retraining, low-rank truncation or FP16 conversion. The mathematical graph is exact; different reduction order and FMA scheduling can change FP32 rounding.
4. Leaves existing production selection unchanged. The mode is incompatible with `RVC_NSF_RESBLOCK_FP16` and intentionally errors instead of silently changing precision. Fused shared-tile and kernel-variant experiments must be disabled during matched A/B.
5. Adds NVTX ranges `stage115_resblock_first` and `stage115_resblock_second` around matched baseline/candidate subgraphs, visible to NCU/NSYS. These ranges are not performance counters by themselves.

The source lives in `vendor/cubek-convolution-0.3.1/src/kernels/forward/direct_preact_residual.rs`, `vendor/burn-cubecl-0.22.1/src/kernel/conv/preact_residual1d.rs`, and two model routing files. Backend is CubeCL; no CUDA-specific code or C++ kernels were introduced.

### Important performance risk

The initial implementation is a transport proof-of-concept, **not a Tensor Core accelerator**. It removes materialization/launch work but still computes C×C×K dense reductions through scalar/short-vector arithmetic. Fusing epilogues can lose if reduced occupancy/register pressure or read multiplication dominate. NCU must explicitly establish performance. If the kernel loses to baseline or implicit GEMM, stop this route and reuse the evidence to design a stronger algorithm (not another unmeasured tile tweak).

## Validation

- Rust 1.100.0 / `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu --lib`, with `DOCS_RS=1`: **PASS**.
- `rustc --edition=2024 --test research/stage115-resblock-transport/transport_reference.rs`: **2/2 PASS**, including the physical weight bijection, negative activations, padding, dilation, optional residual and different loop orders.
- The CubeCL CPU-runtime library test compiled but **could not link**, because this environment does not contain the Tracel LLVM 23.1 offline bundle. `DOCS_RS=1` skips download at build time but does not provide required LLVM C symbols at link time. Do not mark CPU runtime PASS.
- Real NVIDIA GPU, NVRTC kernel JIT, NCU/NSYS result, full checkpoint, full WAV and PCM parity: **NOT TESTED** in this environment.
- CUDA feature `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cuda --lib` with `DOCS_RS=1`: **PASS**. The check compiled Rust and feature gates; it did NOT launch CUDA kernels, validate numerical parity or measure runtime. Windows package and PowerShell script execution: **NOT TESTED** here.

## Experiment protocol on Windows RTX4060

1. Build the patched code with the same Rust/CubeCL versions as PR #2. Supply the local private voice/HuBERT/RMVPE/index/WAV; never upload those private assets into CI.
2. From the packaged executable and stage115 PowerShell runner, compare `baseline`, `gemm` (existing Burn/CubeCL ImplicitGemm control), and `transport`. Use identical input, resident loaded checkpoints, FP32, warmups and sync points. Compare complete PCM WAV with numerical RMSE/maximum error, not just hashes. Default benchmark writes WAV SHA256; different hashes do not by themselves imply failure.
3. Profile with Nsys both the full DecoderNSF and NVTX first/second convolution ranges; then profile representative NCU kernel launches with `--set full` and inspect workload, L1TEX/DRAM traffic, warp-stall attribution, occupancy, registers, and GPU duration separately. Do not compare replay time directly to noninstrumented warm benchmark time.
4. Start with C128, K11 and representative dilations; capture other stages and K3/K7 to verify any improvement is not a special-case regression. Do not conclude MAC share equals runtime share.
5. Only if this improves **GPU device time and matched full-WAV end-to-end latency** without quality regression should production dispatch change. If it does not, retain the benchmark evidence and revert the opt-in kernel.

## Constraints

No obsolete or legacy scripts edited. No .NET/Unity build. All repository source analysis and modifications were performed locally from the PR #2 Git bundle. No GPU timing was invented. No private model assets were included.
