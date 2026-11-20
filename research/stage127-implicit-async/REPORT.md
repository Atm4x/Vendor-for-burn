# Stage127 — Conv1d asynchronous ImplicitGemm R&D (source-only, GPU pending)

Date: 2026-11-17. User explicitly requested a fresh full HEAD bundle from PR #2. Git Bridge run 44137421030 produced SHA-verified bundle pointing to `1e9f23495e97fbbc59fcdeac9c3fde4f637dcfa5`. All source work was local after retrieval.

A new **opt-in** Conv1d route uses `cubek::convolution::ConvAlgorithm::SimpleAsyncCyclic` through the existing CubeCL `conv_gemm_simple_async` for the same shapes and FP32 CMMA arithmetic contract. The sync control remains untouched, and other Conv2d/ConvTranspose code is not rerouted by this switch. This is a pipelined asynchronous staging **hypothesis**. It could reduce global-load stalls but may instead raise shared memory occupancy, fail CUDA JIT or hurt throughput. A 3–4× full-RVC improvement is not proven or forecast from this one kernel change.

The baseline is the **old fast ImplicitGemm full WAV** (~12s), not the pre-NTC slow direct Conv1d that inflated the apparent gain in Stage118. `RVC_NSF_CONV1D_GEMM_ASYNC` does not enable itself in production. Profile both large waveform and actual kernels against the same Rust executable. The shipped PyTorch reference captures the different original Python pipeline, not equivalent mathematical graph work. No scripts with private model/audio weights have been touched.

Source review: Irodori's performance used shape-specific fused algorithms, semantic graph compaction, caching and reducing submission overhead, and 10-warmup+ repeated multi-process checks. Dashmag's CUDA parity requires disabling TensorFloat32 where requested (`CUBECL_CUDA_TF32=0`) and keeping conformance fixtures; its causal Conv1d kernels are **depthwise**, not an adequate substitute for RVC dense C×C Conv1d. RTX 3060Ti Irodori numbers are not extrapolated to RTX 4060.

Stage68 historical synchronized full 90s: NSF residual ~4.473s, HuBERT frontend ~2.612s, RMVPE UNet ~2.439s, TextEncoder attention/FFN ~1.744s, GRU ~0.506s, with nested intervals and synchronization overhead. ImplicitGemm Conv1d alone cannot explain all 12s; profile a matched full timeline before redesigning multiple networks.

Stage127 hypothesis is a low-risk first try with clear A/B and failure gates. Never claim measured win from local `cargo check` alone, and do not promote if NCU counters or warm ABBA reject it.
