# Stage 86 — NSF nonlinear operator R&D: shared-memory time tiles and real-weight compression test

Base: verified local Stage85 commit `406bf9c3eee023af302a035106aa3dda4094b2b9` from `RVC_STAGE85_SOURCE_FROM_STAGE78.bundle`. Toolchain: the user-provided stable Rust 1.100.0. No NVIDIA GPU or Windows host involved. All changes are **source-only, opt-in**. Do not merge or promote to production on CPU-typecheck evidence.

## Why this stage targets residual math, not transpose or another cache

The synchronized Stage68 profile on RTX 4060 for the full ~89.94s recording identified NSF 5.1644s, of which 4.4727s occurred in the residual aggregation stages; upsampling was only 0.2188s. Separate HuBERT frontend (2.6124s) and RMVPE UNet (2.4385s) remain large. Diagnostic GPU synchronization affects absolute times, and these stage measurements must not be added to overlapping parent spans. A polyphase ConvTranspose1d optimization would target a small measured fraction of NSF latency. It was intentionally not pursued here.

The Stage85 candidate replaced two `LeakyReLU → Conv1d → LeakyReLU → Conv1d → residual-add` steps with two independent direct CubeCL kernels. Stage86 replaces **both convolutions of each supported residual pair with one kernel** using temporal blocking and a shared-memory intermediate, without storing the intermediate first-convolution result in global GPU memory. That is an alternative operator execution algorithm, not an activation cache between audio chunks.

## Operator and implementation

Exact mathematical target for each ResBlock pair (same-channel symmetric zero-padded Conv1d):

- `H[t,c] = b1[c] + Σ_(ic,k) W1[c,ic,k] * leaky(X[t + k*d1 - p1,ic])`
- `Y[t,c] = X[t,c] + b2[c] + Σ_(ic,k) W2[c,ic,k] * leaky(H[t + k - p2,ic])`

Out-of-range input `X` and **out-of-range H** are separately treated as zero. This distinction matters when `b1` is nonzero. The CubeCL kernel computes `H` for an expanded temporal tile in workgroup shared memory, synchronizes once, computes `Y` for the central time tile, and writes only `Y` to global memory. Each `(batch,time_tile)` occupies one CUDA workgroup of 128 threads. Every output is owned by exactly one workgroup; no atomics or cross-workgroup synchronization.

- `RVC_NSF_PAIR_SHARED_TILE=1` enables the experimental route, leaving production unchanged when unset.
- Tile length 32 output samples, second-convolution halo at most 5 samples on each side. Scratch budget `(32+10)*64*4 = 10,752 bytes` shared memory per workgroup.
- Eligible shapes: `1 <= channels <= 64`, second kernel odd and `<=11`, second dilation exactly 1. Actual loaded conv weights and biases determine the shape. Wide layers retain the existing implicit-GEMM route by design; no Python/native implementation switching.
- First-convolution odd-kernel dilation and symmetric padding are preserved; actual tensor strides, optional bias and batch dimension are handled. Hosts reject unsupported dtypes/layouts, out-of-bounds strides, indexing overflow and invalid geometry.
- All remaining stages retain current weight normalization, source phase, per-chunk RNG and global-F0 settings. Sequential residual aggregation remains the production policy.

**Execution risk:** this is a scalar/direct reduction rather than a tensor-core matrix tile. It removes one dispatch and a large intermediate global tensor at the cost of recomputing H in neighboring tile halos. At second-kernel lengths 3/7/11, ideal interior first-conv duplicate-work factors are 1.0625/1.1875/1.3125 for 32-sample tiles. Faster memory traffic does not guarantee faster inference; the shader might lose badly to implicit GEMM. An honest benchmark must reject it if that happens.

## Quantitative cost bounds, not measured speed

For the default 300-frame core + 40-frame context on each side + 4-frame tail, the model receives 384 frames per full-sized chunk. At 32 chunks and ×200/×400 decoder time expansion, a *rough full-chunk shape estimate* for the last two NSF stages is 2,457,600 sample positions at 64 channels and 4,915,200 positions at 32 channels. Nine residual-pair intermediates per stage imply around **11.3 GB of intermediate H writes and 11.3 GB of H reads** that can potentially be removed. This assumes all 32 chunks have identical full length and all supported pairs execute. Actual last chunk geometry is smaller.

This is not a claim of seconds saved. At a hypothetical 272 GB/s full utilization, 22.6 GB is only ~0.083 seconds of memory bandwidth. Launch overhead and low efficiency may matter, but the dominant opportunity is high FLOP throughput and efficient channel tiling. An eventual tensor-core-tiled implementation is necessary if this scalar fused route loses.

## Real checkpoint SVD: negative result for naïve low-rank Conv1d

Read `CrazyMita.pth` with PyTorch CPU `weights_only=True`, reconstructed weight normalization along output channels, and analyzed 54 actual NSF residual convolutions using singular values of matrices `[Cout, Cin * K]`. The rank retains a given percentage of squared singular-value energy; this is a **weight-space** diagnostic, not an activation-audio quality metric.

| Channels | Layers | Mean rank retaining 99% energy | Theoretical multiply reduction at 99% | Theoretical multiply reduction at 95% |
|---:|---:|---:|---:|---:|
| 32 | 18 | 22.28 / 32 | 1.263× | 1.747× |
| 64 | 18 | 50.00 / 64 | 1.099× | 1.451× |
| 128 | 18 | 103.0 / 128 | 1.074× | 1.390× |

Reduction formula for a rank-`r` spatial-feature flattening is `(Cout*Cin*K)/(r*(Cin*K + Cout))`, counting a factorized channel-kernel conv plus a following 1×1 projection. The additional kernel launch, activation allocations and numerical error are not included. The accurate low-rank case barely reduces nominal arithmetic on 64/128-channel layers. **Reject simple weight SVD as the main acceleration plan** absent strong post-training signal-conditioned structure or additional model-quality evidence. Reducing 5% energy in each weight may compound unpredictably; it is not an acceptable production premise.

`stage86_weight_spectrum.py` reproduces this on the user-supplied `.pth` without changing it, and writes aggregate results. No model weights or audio are bundled.

## Tests and evidence

- Stable Rust **1.100.0** directly from supplied toolchain: standalone `rustc --test` mathematical reference on 8 synthetic parameter/time/batch cases; exact bitwise FP32 parity for sequential vs time-tiled CPU arithmetic, plus halo/coverage test: **2 PASS / 0 FAIL**.
- PyTorch **2.10 CPU** on `CrazyMita.pth` original weight-normalized tensors: **15 PASS** across five actual residual pairs, time lengths 17/31/64, batch two, 32-sample time tiles. Maximum absolute error vs original PyTorch Conv1d implementation: **5.9604645e-8**.
- `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu,flex,cuda --lib --bin tts-with-rvc-rs --test nsf_pair_tile_math` on Rust 1.100.0: **PASS**. `git diff --check` and Rustfmt: PASS.
- Real CUDA kernel JIT, actual `cuda` inference, Windows CI, whole-WAV audio parity, peak VRAM and speed: **NOT TESTED**.

The real-weight Python verifier and the mathematical Rust tests execute the **algorithm's formulas**, not the CubeCL device kernel. They do not prove device execution, numerical parity or speed. The development environment has no NVIDIA GPU; a separate Windows CI job compiles the executable and provides the familiar `Run-RVC-Full-Stage86.ps1` A/B/B/A runner, but its hosted runner does not supply the user's RTX 4060.

## Acceptance, stop conditions and next algorithmic direction

Stage86 runner holds NSF GEMM, persistent BiGRU, Mel reuse, model inputs, exact RNG seeds, batch/core frame parameters and the SHA256 of the WAV comparable. It alternates baseline/shared/shared/baseline. Acceptance requires all four outputs and deterministic repeated baseline, no voiced-frame flips, bounded PCM delta and a significant warm speed benefit beyond A/B drift. Failed runs remain in result ZIP.

**Stop** if the scalar fused pair is slower than ImplicitGemm or if register/shared-memory pressure causes kernel compilation failure. Do not perform arbitrary tile-size tuning in lieu of diagnosis. The next structural R&D should explicitly target tensor-core utilization on the 64- and 128-channel NSF convolutions, or investigate model-wide chunk scheduling while establishing parity with the original Python FP16 inference. These may offer larger improvements than eliminating a single intermediate tensor.

The stable allocator API in Rust 1.100 was not used: device tensor scratch is controlled by CubeCL, not the host `Vec` allocator.
