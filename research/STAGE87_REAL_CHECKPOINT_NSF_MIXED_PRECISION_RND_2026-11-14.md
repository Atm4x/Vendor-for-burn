# Stage 87 — NSF FP16 implicit-GEMM R&D on the real RVC checkpoint

Base: local Stage86 commit `16b69418bf73676d8c13bc299075318586947dae`, whose tree `177bb9370206c9fb814e03de02209c00f51f2741` matches the separately published PR #2 Stage86 tree. Toolchain: user-supplied Rust 1.100.0. Source-only experimental change. No NVIDIA GPU available locally.

## Core observation: Rust diverged from original Python before the convolutions

The user-provided `CrazyMita.pth` contains **FP16** tensors for the NSF residual weights, e.g. `dec.resblocks.0.convs1.0.weight_v`, and for the decoder upsampling/post convolution weights. The current Rust `RvcCheckpoint::tensor_f32` loads FP16 values as FP32, normalizes weight-normalized Conv1d parameters on the host, and runs the residual convolutions in FP32. The original PyTorch CUDA path normally permits half-precision inference on an RTX 4060. The numerical execution formats therefore differ, independently of convolution algorithmic quality.

Stage86 scalar/shared-memory fusion can reduce launches but does not use matrix tensor cores; its benefit against ImplicitGemm on RTX is unknown. In contrast, a **whole-ResBlock FP16 precision-island** policy tests a different compute regime using the existing CubeCL implicit-GEMM implementation. The implementation is **eligible** for FP16 tensor-core acceleration. Whether CubeCL actually dispatches efficient MMA kernels requires JIT/profiler evidence and must not be assumed.

## Actual model cost atlas (not a timed GPU profile)

From real checkpoint `config[12] = [10, 10, 2, 2]`, three ResBlock branches with kernel sizes `[3, 7, 11]`, three pairs of two Conv1d each, and 384 latent frames entering a full-length local Rust chunk:

| Stage | Channels | Output positions | Estimated ResBlock FLOPs/chunk | Fraction |
|---|---:|---:|---:|---:|
| 0 | 256 | 3,840 | 63.42 G | 18.6% |
| 1 | 128 | 38,400 | 158.54 G | 46.5% |
| 2 | 64 | 76,800 | 79.27 G | 23.3% |
| 3 | 32 | 153,600 | 39.64 G | 11.6% |
| **All** | | | **340.87 G** | **100%** |

Counting 2 FLOPs per multiply-accumulate, excluding activation, padding and memory traffic; real last chunks are shorter. At 32 equal full chunks this would be 10.91 TFLOPs, but **do not multiply this into a measured workload claim**: the final chunk is partial and input frame padding varies. This cost atlas demonstrates why 64/128/256 channels are the first FP16 target (88.4% nominal residual convolution FLOPs); 32-channel high-frequency output remains FP32 by default.

## Implementation and invariants

`RVC_NSF_RESBLOCK_FP16=1` activates FP16 precision islands only for `dec.resblocks.*` Conv1d. `RVC_NSF_RESBLOCK_FP16_MIN_CHANNELS` controls the eligible channel threshold (default 64, positive integer). Stages below the threshold remain untouched. The decision is made at weight load and stored per `Conv1d`, so no runtime switching inside a warm convolution.

For an eligible ResBlock:

1. Load and weight-normalize each convolution's weights in FP32 as before, then convert normalized weights and bias **once at load** to FP16.
2. At the ResBlock entry, convert the prepared FP32 features to FP16 **once**.
3. Execute all six Conv1d operations per ResBlock via the existing `CubeCL ConvStrategy::ImplicitGemm` in FP16; both LeakyReLU and residual additions within this ResBlock remain FP16.
4. At the ResBlock exit, convert its output back to FP32 **once**, before cross-branch aggregation and the next decoder stage.

This is a **precision island**, avoiding per-Conv FP32↔FP16 round trips. For each eligible 3-pair ResBlock it requires two boundary casts rather than twelve (six Conv1d × two casts), an **83.3% reduction in conversion boundaries** compared with the abandoned initial Stage87 proposal. With three eligible stages and three branches each, the number of explicit cast boundaries is 18 rather than 108 per full model decode; this is an operator count, **not measured speed**.

This preserves **topology, branch ordering, dilation, padding, NSF source phase and noise, output length and FP32 cross-branch aggregation**. It does **not** preserve FP32 per-pair residual arithmetic or guarantee bitwise parity with FP32 Rust or Python FP16. Python's whole-model FP16 boundaries and weight-normalization details are not completely reproduced. Explicitly reject simultaneous `RVC_NSF_PAIR_SHARED_TILE` or `RVC_NSF_FUSED_PAIR_MAX_CHANNELS` experiments, which operate on FP32 kernels and would otherwise silently bypass the precision policy. No obsolete/legacy files or vendor dependencies were edited.

Production default without `RVC_NSF_RESBLOCK_FP16=1` remains unchanged. `main.rs` benchmark JSON now records the effective precision policy and threshold.

## Offline evidence: real weights, simulated activations

`research/stage37-windows-gpu-bench/stage87_resblock_fp16_probe.py` uses the **real** checkpoint and reproducible random input activations (one per each of 12 actual residual branches; 35 positions; sigma 0.3; deterministic seed). It executes three variants with CPU PyTorch: original Rust-style FP32, the Stage87 per-block FP16 precision island, and the abandoned per-Conv round-trip variant. This is *not* an actual-audio activation distribution, GPU JIT, or end-to-end quality evaluation.

Maximum block-output RMSE across 12 branches:

- ResBlock FP16 island vs FP32: **0.000536133**.
- Per-Conv round-trip FP16 vs FP32 (rejected intermediate design): **0.000420629**.
- FP16 island vs per-Conv round-trip: **0.000528030**.

All results, including per-stage/branch RMSE, maximum absolute error, relative L2, actual input weight dtypes and cost atlas, are saved as `evidence/stage87_real_weights_probe.json` in the release ZIP. These values cannot predict PCM16 RMSE or timbre quality; errors may compound in the generator.

## Verification and acceptance gates

- Rust 1.100.0 standalone precision-policy unit tests: **3 PASS / 0 FAIL**. Includes default unchanged, 32/64/128 threshold semantics, unrelated layers, invalid-input rejection.
- `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cuda --lib --bin tts-with-rvc-rs` on Rust 1.100.0: **PASS**.
- Extended `cpu,cuda,wgpu` typecheck under Rust 1.100.0, offline/locked: **PASS** (`1m45s` on this host; type checking only).
- Python real-weight CPU probe (12 ResBlock branches): **completed**.
- CUDA JIT, actual FP16 GEMM Tensor Core selection, Windows EXE, full WAV, PCM16 RMSE, listening and throughput: **NOT YET TESTED**.

The Windows runner keeps the existing user command format, performs baseline/island/island/baseline in independent processes, verifies baseline-to-baseline and mixed-to-mixed WAV determinism, reports normalized RMSE and warm inference, and saves errors. `-MinChannels 64` is the default; use 32/128 in separate experiments only after the first quality/speed gate. CI compiles and packages the Windows executable but has no guarantee of access to an NVIDIA GPU.

## Decision rule and next R&D

**Reject** Stage87 mixed precision if CUDA runtime fails to select an efficient FP16 GEMM route, if full audio quality degrades materially, or if A/B/B/A speed improvement is within baseline drift. Do not claim a speed-up merely because FP16 is supported by hardware.

If successful, the next architectural step is to extend precision islands across whole NSF decoder stages and reduce remaining inter-stage conversion boundaries, and compare against original PyTorch FP16 on identical geometry. If unsuccessful, investigate the actual CubeCL implicit-GEMM kernel selection (MMA, tile, occupancy) using the profiler rather than adding more scalar fusion layers.

Rust 1.100 stable custom allocators are intentionally not used here: Conv1d GPU intermediate storage and workspace are managed by Burn/CubeCL, not host Vec allocation.
