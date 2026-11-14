# Stage 70 — real HuBERT CubeCL CPU CpuGemm R&D (2026-11-12)

**Base:** PR #2 at `2963e449`. **Status:** experimental CPU-only opt-in; production/default and CUDA unchanged.

## Reproduction environment

Offline Rust 1.98.1, actual Tracel LLVM 23.1.0-4 bundle, Linux 5-vCPU AMD EPYC container, optimized release, `--features cpu`. Real HuBERT Base checkpoint from user's Library, layer 0, 12 frames, 768 channels, 12 heads, deterministic f32 input. Private checkpoint/index/WAV are NOT part of Git.

## Actual source-level finding

Vendored `cubek-matmul-0.3.1`, with both `multi-level` and `tiled` enabled, routes default `Strategy::Auto` to `SimpleCyclicCmma`, then uses `SimpleUnit` when unavailable on CubeCL CPU. A specialized `tiled::Strategy::CpuGemm` kernel exists but is omitted from that auto path. The experiment selects CpuGemm only when `client.properties().hardware.num_cpu_cores.is_some()` and `CUBEK_EXPERIMENT_CPU_GEMM` is set. Vendored Cargo checksum updated; no default change.

## Benchmarks (real HuBERT weights)

Original backend: cold 77,820.29 ms; second process-internal pass 69,057.60 ms (12 frames, first transformer layer). Diagnostic spans within hot pass: Q/K/V 16,470.9 ms, attention scores 585.4 ms, softmax 17.0 ms, value matmul 537.8 ms, output projection+norm 6,580.1 ms, FFN input+GELU 23,616.1 ms, FFN output+norm 21,240.3 ms.

CpuGemm opt-in, same real checkpoint and input: cold 1,058.29 ms; hot **22.57 ms**, a ~**3060×** improvement for that layer. Hot spans: QKV 4.636 ms, scores 0.281 ms, softmax 0.171 ms, value matmul 0.133 ms, projection+norm 1.234 ms, FFN input+GELU 4.536 ms, FFN output+norm 11.247 ms. These are measured CPU backend sync spans, not CUDA data or full-RVC speedups.

All **9216 FP32 output values are bit-identical**, RMSE = 0, max abs = 0, both SHA256 `0e311250f84d1b950e413dd27667ad5e0a33c0b6c561959d1a7d852bd6bcd2df`.

## Full short real RVC smoke

`infer-wav-segment` with original private CrazyMita.pth, hubert_base.pt, rmvpe.pt, CrazyMita.index, MitaKind.wav for source 0–0.24s, CubeCL CPU + CpuGemm. **PASS**: output 8800 samples at 40kHz in a WAV, wall 12.07s including diagnostic tracing, peak RSS 1,314,196 KiB. HuBERT 12-layer transformer stack 694.6ms; whole HuBERT 1.746s, RMVPE 4.255s, NSF 2.075s, TextEncoder 1.368s, Flow 0.551s. Spans are nested and include explicit sync; don't sum nested phases. Second independent non-traced process **PASS**, wall 14.11s; output WAV SHA256 was identical to traced run: `6b11d18838d4d1722cf8fb9f2facf989cdef0b6c35c48e6833680099058a66f4`. Previous Stage69 CubeCL CPU default did not finish HuBERT within 240s.

## Limits and next step

**Do not extrapolate 3060× from one layer to whole RVC**; exact CPU pipeline accelerated, but no full 89.94s WAV trial or warmed production end-to-end throughput measured. No autotuned globally optimal selection proven for all CPU tensor layouts. Shape test matrix (multiple frame lengths, transposed/batched/irregular strides, dtypes) and proper automatic selection in CubeK are required before changing CPU default. GPU execution must retain its current algorithm.

Full raw data, diagnostic stderr/stdout, original local benchmark source and source patch are available in user-facing `RVC_STAGE70_REAL_HUBERT_CPU_GEMM_PROFILE.zip`. Private weights intentionally excluded.
