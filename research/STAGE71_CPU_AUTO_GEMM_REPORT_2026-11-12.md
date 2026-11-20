# Stage 71 — CPU Auto CpuGemm and real RVC validation (2026-11-12)

PR #2. Parent Stage70: `b6067074d1bdf47e83cc42c01a36c70ea0175bb8`.
Local environment: offline Rust 1.98.1, Tracel LLVM 23.1.0-4, 5-vCPU AMD EPYC Linux, CubeCL CPU, release. **Not CUDA results.**

## Production change

In `vendor/cubek-matmul-0.3.1/src/strategy.rs` CPU devices with pure FP32 (same global/stage/register dtype), nonquantized, untiled inputs, and a contiguous innermost matrix axis on both operands/output now select existing tiled `CpuGemm` automatically. Unsupported dtype/layout still runs the existing `Auto` strategy. CUDA selection is untouched. The Stage70 experimental env flag is unnecessary for compatible CPU FP32 shapes. Updated vendored Cargo checksum. Additional reproducible CLI `models/tts-with-rvc-rs/src/bin/bench_cpugemm_shapes_stage71.rs` validates shape/stride/determinism and sampled numerical reference.

## Real HuBERT layer with original weights

Ran real `hubert_base.pt` layer0 for 12, 100, 384, 500, 900 time frames, each 3 times, with explicit Stage70 CpuGemm, same device/model, outputs finite with stable sum/max. Times in milliseconds, cold / hot1 / hot2:

| Frames | Cold | Hot1 | Hot2 |
|---:|---:|---:|---:|
| 12 | 1103.14 | 24.85 | 34.83 |
| 100 | 1383.08 | 222.54 | 262.19 |
| 384 | 1862.48 | 542.41 | 741.48 |
| 500 | 2362.66 | 1233.75 | 1281.14 |
| 900 | 2737.49 | 1437.89 | 1434.36 |

Stage70's 3060× benefit at 12 frames must NOT be extrapolated to larger shapes; no old SimpleUnit baseline was repeated at every size.

## 11 matrix geometries: PASS with and without flag

Tested QKV [12,768,768] and [900,768,768], HuBERT FFN [100,768,3072], FFN2 [384,3072,768], RVC TextEncoder [27,256,768], RMVPE projection [33,768,1536], both-transposed [13,31,27], irregular [45,769,513] and [67,1025,259], batched [12,17,64]×[12,64,19], and broadcast RHS [12,23,64]×[1,64,21].

Each **11/11 PASS** for correct result shape, *entire-output bitwise repeat*, and 24 sampled output values verified independently by scalar f64 accumulation. Max sampled absolute error 0.0002782345, acceptance 0.002. After *forced* `cargo clean --release -p cubek-matmul` and full offline release dependency rebuild, verified the automatic CPU path with `CUBEK_EXPERIMENT_CPU_GEMM` **unset**. Earlier incremental build had incorrectly reused stale vendored dependencies; final testing did not.

## Real end-to-end RVC with private model assets, not committed

Used CrazyMita.pth, hubert_base.pt, rmvpe.pt, CrazyMita.index and MitaKind.wav from user Library, backend CubeCL CPU:
- 0–0.24s source: auto CPU, no flag, PASS 13.13s, 8,800 output samples at 40kHz, 1,313,964KiB RSS. WAV SHA256 `6b11d18838d4d1722cf8fb9f2facf989cdef0b6c35c48e6833680099058a66f4`, exactly Stage70 flagged CpuGemm output.
- 0–1s source: with explicit CpuGemm and diagnostic tracing, PASS 18.46s, 39,200 samples; CPU auto no flag, PASS 20.72s, same SHA256 `23196342e37dec64116289ae337272ef460beafb684c0e06aeb16881d3840910`. Differences in process wall time reflect distinct tracing and workload; not a controlled speedup measurement.
- One-second diagnostic nested non-additive spans: RMVPE 7.44s (BiGRU 4.46s, UNet 2.53s), HuBERT 3.47s (12 Transformer layers 2.31s), NSF 3.20s (residuals 2.03s, upsample 0.35s), TextEncoder 1.44s, Flow 0.49s, FAISS 0.009s.

Rustfmt/check and git diff whitespace verification passed. Only FP32 compatible CPU matrices have been validated; mixed dtype, quantization, arbitrary tiled layouts, full 89.94s CPU output and CUDA kernel runtime remain out of scope. Full logs and benchmark source are in `RVC_STAGE71_CPU_AUTO_GEMM_PROFILE.zip`.

## Stage72

On actual CPU workload RMVPE recurrent 4.46s is now the top component. Build a proper efficient CPU BiGRU (not Stage67 GPU persistent kernel), validate recurrent math and real audio; then investigate RMVPE UNet and NSF residual Conv1d for original production shapes. Preserve CUDA route and avoid unused legacy scripts.
