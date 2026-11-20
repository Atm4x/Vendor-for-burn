# Stage 100 — Stage 99 integration, BN A/B verification and Winograd F(2×2,3×3) reference

Date: 2026-11-15. Branch: `research/stage100-spatial-bn-local` (local only). Upstream: `Atm4x/Vendor-for-burn`, PR #2. No Stage100 code was pushed to GitHub.

## Git provenance

- One CI-trigger commit to existing workflow (comment only): `e0e4da4958dab860bbb20ab90fed90556a0f7220`, parent PR2 head `bd3c4041e3d3bc1521223478dd98af4924760954`.
- Git Bridge workflow run `43649507132`, success, artifact `14135831700`.
- Full-history Git bundle size `98,021,927` bytes, SHA-256 `5569328ebd55153faae663db6abf43b69f16f6064cc923b754a64fe9ba5c0dda`.
- `git bundle verify` succeeded inside cloned repository; bundle contains main, feat/rvc-burn-parity-stage1, and experiment/conv-kernel-rnd-stage33-20261110 refs and a complete history.
- Rust 1.100.0 installed locally from the user-provided offline toolchain.

## Integration and corrections

Stage99 patches could not be applied by `git apply --check` because patch context did not match real source files. The intended changes were reconstructed on top of the exact PR2 source checkout and compiled, not copied into a synthetic source tree.

1. Added `vendor/cubek-convolution-0.3.1/src/kernels/forward/direct_spatial_quad.rs`, a 2×2 spatial register-tiled forward Conv2d kernel; connected it through vendored `burn-cubecl` into `models/tts-with-rvc-rs/src/cubecl_conv.rs` and RMVPE execution. Opt-in flag `RVC_RMVPE_UNET_SPATIAL_QUAD=1`.
2. Replaced unsupported CubeCL `continue` syntax with a guarded block, allowing macro expansion and crate typecheck.
3. Added BatchNorm folding to Conv2d and ConvTranspose2d at RMVPE model load. Kept unfused normalization for the input BN and for all reference paths; independent opt-in `RVC_RMVPE_BN_FOLD=1`. For ConvTranspose2d weight layout `[in_channels,out_channels,kH,kW]`, scaling takes place along dimension 1, with an output bias.
4. Added new `rmvpe_unet_conv_execution=spatial_quad` and `rmvpe_unet_bn_folded` markers to full WAV benchmark JSON, matching execution flags.
5. Created Stage100 long-WAV Windows A/B runner and analyzer, covering **five distinct configurations** in symmetrical order with two samples of each: `reference_direct`, `candidate_direct`, `fused_direct`, `fused_spatial`, `fused_tf32x2_batched`. The analyzer rejects absent/mismatched BN execution and incompatible chunk boundaries, shapes or runtime settings.
6. Updated local vendored Cargo checksums for the modified `burn-cubecl` and `cubek-convolution` crates. No legacy scripts changed.

## Build and validation

- Rust 1.100.0: `DOCS_RS=1 cargo check --offline -p tts-with-rvc-rs --no-default-features --features cpu --lib --bin tts-with-rvc-rs` — **PASS**.
- `DOCS_RS=1` is strictly a **local offline typecheck workaround**: `tracel-llvm-bundler` otherwise tries to download LLVM (DNS unavailable). It does not validate actual LLVM compilation/linking or CUDA execution.
- Stage99 standalone Rust math checks executed correctly as standalone binaries (`rustc -O`, **not** `rustc --test`): 6 Conv2d spatial variants PASS, 2 BatchNorm algebra checks PASS.
- Stage99 Python analyzer tests: 4 PASS. Stage100 Python analyzer tests: 5 PASS (including new BN-fold-guard).
- `git diff --check` PASS.
- Stage100 original Windows executable: **not built**; CUDA/WGPU/Windows compatibility and RTX 4060 speed **not measured**.

## New Winograd R&D result

The standalone CPU reference `research/stage100-conv2d/winograd/reference.rs` computes forward 3×3, stride1, dilation1, pad1 convolution using:

- Weight prepacking once: `U = G g Gᵀ` (16 coefficients instead of 9 per input/output channel pair).
- Input transform per spatial tile and input channel: `V = Bᵀ d B`; the same 16-element `V` is shared across all output channels in a group.
- Pointwise tile-space matrix accumulation: `M[t, oc, k] = Σ_ic U[oc, ic, k] × V[t, ic, k]`, `k=0..15`.
- Inverse transform: `Y = Aᵀ M A` plus bias, trimming odd spatial tails.
- Grouped and depthwise cases, even/odd spatial sizes, batch >1, and different input scales are handled. Seven independent deterministic numerical tests PASS.

### Numerical results

| Shape | Max abs FP32 vs Direct | SNR (dB) |
|---|---:|---:|
| N1, 1×1, 1→1 | 0.00000004 | 130.70 |
| N1, 2×3, 3→7 | 0.00000048 | 138.47 |
| N2, 7×9, 8→8, groups=2 | 0.00000143 | 135.33 |
| N1, 13×15, 8→8, depthwise | 0.00000072 | 138.23 |
| N1, 32×32, 16→32 | 0.00000954 | 131.22 |
| N1, 64×64, 32→32 | 0.00001717 | 128.72 |
| N1, 5×7, 3→5, scale 100 | 0.01171875 | 136.40 |

### Local CPU comparison (indicative ONLY)

Rust 1.100.0 `rustc --edition 2024 -O`, single-thread, warm median of nine measurements, `black_box` prevents elimination. Comparison is versus **this specific naive scalar Direct reference**, not an optimized GPU or PyTorch operator. Weight preprocessing excluded from timed Winograd inference.

| Configuration | Naive scalar Direct | Winograd prepacked CPU | CPU-only reference ratio |
|---|---:|---:|---:|
| 32×32, Cin16 Cout32, 3×3 | 8.320 ms | 0.320 ms | 25.96× |
| 64×64, Cin32 Cout32, 3×3 | 68.384 ms | 2.420 ms | 28.26× |

These large ratios include advantages from tile reuse, loop organization, fewer boundary checks, and optimization differences. They do **not** establish comparable gains versus optimized Direct/GEMM/CUDA kernels. The arithmetic lower bound is 36→16 multiplications for four 3×3 outputs per channel pair, i.e. 2.25× fewer multiplications in the tiled product, but transforms cost arithmetic and memory; transformed weight storage grows by 16/9=1.78×. GPU throughput must be measured.

## Source comparison against provided `pytorch-main.zip`

Extracted exact archive members:
- `pytorch-main/aten/src/ATen/native/Convolution.cpp` (`_convolution`, around line 1648–1674): chooses backend, delegates regular CUDA Conv2d to `at::cudnn_convolution` and transpose to `at::cudnn_convolution_transpose`.
- `pytorch-main/aten/src/ATen/native/cudnn/Conv_v8.cpp` (around lines 580–645, 923–1000, 1161–1298): builds candidate engine configurations through cuDNN frontend heuristics and uses a plan cache / benchmark path.
- This source archive **does not contain the cuDNN engine's proprietary CUDA implementation**. Stage97 profiler independently shows selected cuDNN CUTLASS/TC kernels in PyTorch and substantially slower CubeCL Direct kernels in Rust. No claim is made about exactly which cuDNN algorithm executes each RMVPE shape.

## Benchmark goal and remaining blockers

Hard acceptance is warm RVC/Rust <=1.20× original PyTorch FP16 on >=60s audio, comparable segmentation and settings, no CUDA Graphs, with numerical and listening gates. **Not yet measured for Stage100**. The known Stage98 12s result (RTX4060) is still Batched TF32×2 1344.751ms, PCM16 RMSE 1.499, max abs 40, versus Direct 1693.340ms. The Stage97 12s Torch FP16 (~675ms) and historical long-PyTorch (~2656ms) are context, not matched Stage100 acceptance baselines.

For next GPU cycle:

1. Build a distinct Windows Stage100 EXE with Rust code from the local Stage100 bundle/patch; do not rename Stage94 EXE.
2. Run `research/stage100-conv2d/windows/Run-RVC-Stage100-Long-CUDA.ps1` with the real >=60s WAV, model, HuBERT, RMVPE and index; verify five A/B modes and their JSON flags, paired chunk boundaries and PCM/SNR. Measure warm medians, Stage94 vs candidate direct first, then BN-only, BN+spatial, BN+batched TF32×2.
3. Add model-level RMVPE F0 parity and listening validation. Folding can change floating-point operation ordering and CPU algebra alone is insufficient.
4. GPU Winograd design: implement a distinct 3-stage pipeline, with prepacked `GgGᵀ` weights at model load, one input-transform tensor per tile/input channel, 16-way batched Matmul/GEMM over `(groups, transform_k, spatial_tiles, Cin)` by transformed weight `(groups, transform_k, Cin, Cout)`, then inverse transform+bias+activation. Benchmark transform overhead and materialized memory; do not instantiate a per-output-channel input transform or claim tensor core speed before profiling.
5. Compare Winograd only for dense 3×3 stride1 dilation1; grouped/depthwise and odd/tiny matrices should have their own measured model-stage breakdown. Keep all variants as experiments until real CUDA evidence.
6. Only once Conv2d measured, tackle GPU recurrent BiGRU (~205ms in Stage97 multi-pass profile) and high-frequency elementwise masks/GEMMs if bottleneck remains.

No changes have been uploaded to PR #2 beyond the one-shot CI workflow comment used for obtaining the full Git bundle.
