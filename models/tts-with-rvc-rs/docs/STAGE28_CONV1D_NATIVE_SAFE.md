# Stage 28 — Safe native Conv1d experiment (Burn Flex 0.22.1)

## Scope

Experimental isolated implementation inspired by tracel-ai/burn PR #6322. The root `main` and upstream PR are unchanged. Only the vendored Burn Flex `conv1d` path is modified; `conv2d`, `conv_transpose1d` and other operators remain unchanged.

## Three regression fixes

1. `f16`: inputs and weights are promoted to `f32`, all convolution contributions are accumulated in `f32`, bias is applied in `f32`, and the final result is converted to `f16` once. This avoids partial-result `f16` overflow when kernel positions cancel. This variant uses extra allocations and has not been performance tuned.
2. IEEE non-finite padding: missing padded contributions of non-finite weights propagate `NaN` in both the native depthwise and GEMM paths.
3. Empty channel reduction: `gemm_k == 0` no longer forms out-of-bounds pointers; output is zero plus any bias.

The optimized `f32/f64` paths use direct strided GEMM or native depthwise accumulation without pseudo-3D expansion, and parallelize over disjoint output groups/channels with Rayon when enabled.

## Verified tests

- `cargo test -p tts-with-rvc-rs --offline --locked`: all suites PASS, 54 total tests (15 unit, 2 CLI, 7 new Conv1d regressions, 30 parity).
- 7 new Conv1d tests: cancellation in `f16`, grouped/depthwise NaN propagation with padding, zero input channel with bias, `f64`, grouped stride/dilation/padding, and 80 deterministic randomized `f32` configurations.
- Burn Flex `--no-default-features` compiled successfully in an isolated temporary workspace with the same offline dependencies.
- New native Conv1d and the legacy 1D-as-3D path returned matching values to seven decimal places in the 6 benchmarked finite `f32` shapes.
- `git diff --check` PASS.

## A/B benchmark

Conditions: Linux EPYC container, 4 vCPU cgroup, debug Rust binaries, 4 Rayon workers, same process and inputs, after two warmups, six timed calls per variant, including output materialization. These are NOT release measurements and cannot be compared directly with the upstream author's optimized Apple M4 results.

| Shape | Legacy conv1d as conv3d (ms) | Native conv1d (ms) | Legacy/native |
|---|---:|---:|---:|
| TCN dilated [8,64,160], 64 out, k5, dilation2, pad4 | 41.603 | 34.849 | 1.194x |
| Audio padded [8,64,500], 64 out, k5, pad2 | 127.610 | 105.068 | 1.215x |
| RVC short [1,256,24], 256 out, k3, pad1 | 30.941 | 28.520 | 1.085x |
| RVC long [1,256,128], 256 out, k3, pad1 | 42.801 | 36.698 | 1.166x |
| Wav2vec stride [1,64,100], 32 out, k3, stride2 | 2.323 | 2.190 | 1.060x |
| Depthwise [2,64,128], k7, dilation2, pad6 | 1.524 | 0.762 | 2.000x |

## Remaining gates

- RELEASE benchmark: BLOCKED. The full release rebuild of Burn Flex and downstream Burn crates exceeded two execution limits. The old optimized binary and new debug executable are NOT comparable, so no full RVC speedup is claimed.
- Full `CrazyMita.pth` synthesis validation: a debug warmup forward produced 9600 samples; subsequent timed execution exceeded the sandbox limit. This is not a complete end-to-end real-checkpoint A/B result.
- Mixed-precision testing beyond the targeted `f16` cancellation regression remains open.
- Upstream publication and merge into the user-facing feature branch are deliberately deferred until optimized RVC parity and performance are measured.
- RMVPE Conv2d remains unchanged and cannot benefit directly from this Conv1d patch.

## Reproduction

- Run `cargo test -p tts-with-rvc-rs --offline --locked` with the repository's offline Rust toolchain.
- Run `cargo run -p tts-with-rvc-rs --offline --locked --bin bench_conv1d_stage28` for the in-process old/new kernel microbenchmark.
- For a release benchmark, build both variants at identical optimization settings and compare the same `bench_rvc_synth` inputs, prewarmed and with CPU thread count pinned.
