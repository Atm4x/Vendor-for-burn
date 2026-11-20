# Stage 84 — mathematically bounded NSF waveform execution and Rust 1.100

Parent: local source-only Stage83 `c5c5d4b4` (which includes Stages 79–81). Stage82 is intentionally excluded. Experimental source only; default decoder logic and legacy code are unchanged.

## Question

Can a single long latent sequence be decoded into an identical waveform by executing finite, analytically chosen NSF tiles, rather than replicating the full model context for every tiny chunk? Can this unlock a global RVC scheduler without allocating full high-rate decoder activations? This is a **memory/scheduling** algorithm, not a claim of faster arithmetic.

## Exact signal model

RVC v2 NSF is a sequence of a pre-Conv1d, repeated transposed upsample + harmonic-source Conv1d + parallel **nonlinear** ResBlocks, followed by a post-Conv1d and tanh. Each residual block has three sequential pairs of dilated Conv1d and ordinary Conv1d, with leaky ReLU and residual additions. There is no global attention in the decoder; all receptive fields are finite. SourceModuleHnNSF, by contrast, integrates F0 phase over time and must be evaluated on the full sequence before output tiling. Stochastic phase/noise inputs must also be held identical to the reference.

For a requested half-open output interval `[a,b)` a standard convolution of kernel K, dilation d, stride s, left padding p has required input interval:

`[a*s-p, (b-1)*s-p+(K-1)*d+1)`.

For ConvTranspose1d with dilation one, the required input interval is:

`[ceil((a+p-(K-1))/s), floor((b-1+p)/s)+1)`.

A sequential two-convolution residual pair of odd symmetric kernels contributes radius `((K1-1)/2)*d1 + ((K2-1)/2)*d2`; three pairs add radii. Parallel branches use the maximum branch radius. Work **backwards** through the decoder and separately union the harmonic-source dependencies induced by each noise-convolution path. Convert harmonic indices back to the original latent frame grid and clamp the result only at the full-sequence edges. The final input interval is the smallest conservative frame-aligned interval obtained from these formulas; interior tiles run the unchanged real decoder and crop only the requested output samples.

No approximate context, interpolated source, relaxed F0, modified weights, Graph replay, or new normalization is introduced. `RVC_NSF_TILE_CORE_FRAMES=<positive integer>` opts into tiled execution. With the flag unset, the existing decoder remains the production path. Speaker conditioning is broadcast or sliced depending on whether its time dimension is one or equal to latent length.

## Verified results in this cycle

- Uploaded `rustc 1.100.0 (0ef1bc386 2026-11-09)` and `cargo 1.100.0`, LLVM 23.1.5 installed from supplied tarball.
- Verified with a compilable stable source that `std::alloc::Allocator`, `Global` and `Vec::<u8,Global>::new_in(Global)` are usable without nightly features. No custom host allocator was added to the GPU pipeline: Burn/CubeCL manages device allocations separately.
- Pure Rust numerical tests of interval coverage, transposed-stride reverse support by brute force, and **full nonlinear two-stage synthetic signal vs tiled**: 3 PASS, 0 FAIL. The latter includes two nonlinear residual branches, upsampling, shared harmonic input, source convolutions, and a post-convolution with output differences below `1e-7` for tile sizes 1,3,7,17,31,64.
- Rust 1.100.0 `cargo check --offline --locked --no-default-features --features cpu,flex,cuda --lib --bin tts-with-rvc-rs --test nsf_tiled`: PASS (type checking, no CUDA JIT or model-inference execution).
- The full Burn synthetic-weights integration test exists in `models/tts-with-rvc-rs/tests/nsf_tiled.rs`, but has **not** been run to completion. CPU test linking was manually stopped after unusually long LLVM/CubeCL linking; do not count it as PASS. The Windows CI gate has explicit execution of this test.

## Duplication lower bounds and stop criterion

For an **illustrative** ×400 RVC geometry with rates `[10,10,2,2]`, kernel sizes `[2*s]`, maximum residual radius 60 per stage, and a 6000-frame input, estimated total decoder-input footprint under different core tile sizes is:

| Core frames | Tiles | Sum of expanded frame counts | Added frames vs once |
|---:|---:|---:|---:|
| 32 | 188 | 10114 | +68.57% |
| 100 | 60 | 7298 | +21.63% |
| 300 | 20 | 6418 | +6.97% |
| 600 | 10 | 6198 | +3.30% |
| 900 | 7 | 6132 | +2.20% |
| 1500 | 4 | 6066 | +1.10% |
| 6000 | 1 | 6000 | 0% |

These are analytic *input frame counts*, not measured FLOPs, memory consumption, runtime, or a claim about the actual CrazyMita checkpoint's exact kernels. A separate executable reference program produced these numbers. Tiling adds work and kernel launch overhead; it can only beat monolithic execution if the **memory/scheduling benefit** outweighs this overhead. Do not promote the candidate on a theoretical speed claim.

**Current integration boundary:** `RvcV2Inferencer::infer_16khz_batch_with_f0` still enforces 0.2–10 seconds of 16-kHz audio per batch input, and the outer full-WAV chunk scheduler is unchanged. Therefore this Stage84 code tiles *inside an existing RVC chunk*. It does **not** yet run an entire 90-second latent sequence through one global NSF decode, nor does it remove the repeated HuBERT/RMVPE/context work. The 6000-frame table is a hypothetical stress case for planning a future long-window route, not a currently executable end-to-end scenario. An exact global scheduler requires relaxing those limits and separately establishing HuBERT, RMVPE, attention and flow window semantics; none of those changes is included in Stage84.

## Next test after Windows CI

The CI workflow `.github/workflows/rvc-stage84-nsf-tiling.yml` uses Rust 1.100.0 and bundles the Windows EXE and `Run-RVC-Full-Stage84.ps1`. Run full-WAV `baseline_1, tiled_1, tiled_2, baseline_2` (ABBA) with the same voice, F0, source and NSF implicit-GEMM configuration. `summary.json` records warm latency, hashes and PCM16 max delta/RMSE. Reject if shape or output mismatches, if output drift is significant, or if the memory goal is not achieved. No automatic enabling in production.

## Larger speed hypothesis beyond Stage84

Real Stage68 synchronized CUDA profile for a roughly 90-second source reported NSF 5.164 s, HuBERT 3.109 s (frontend 2.612 s), RMVPE 3.068 s (UNet 2.439 s), TextEncoder 1.830 s, Flow 0.528 s. Even infinite NSF acceleration alone is limited to around `14.014/(14.014-5.164) = 1.58x` for that workload. Large end-to-end speedups must change the HuBERT+RMVPE+NSF combined execution strategy, not just a single Conv2d kernel or host allocator.

Promising follow-on is one *fused high-rate NSF execution engine*, holding a small time/channel tile in GPU registers/shared memory through nonlinear residual pairs, reducing intermediate tensor writes and launch count. That demands its own numerical reference and code generation; it is not implemented or benchmarked here. For global long-window operation, separately prove which layers genuinely depend on per-chunk inputs and compare against the original Python FP16 pipeline (Stage80 reference crash recovery); do not assume simple global concatenation reproduces chunked attention or stochastic flow.
