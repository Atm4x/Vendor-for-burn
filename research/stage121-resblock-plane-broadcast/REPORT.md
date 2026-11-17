# Stage121 — RVC ResBlock C128/K11: NCU-guided plane-shared input

Date: 2026-11-17. Branch `experiment/conv-kernel-rnd-stage33-20261110`, PR #2. Toolchain Rust 1.100.0.

## Stage120: rejection based on uploaded RTX 4060 evidence

Source: `20261116T234547207Z.zip`, consisting of the benchmark manifest, original NCU `.ncu-rep`, exported 1679-field NCU CSVs, and complete NSYS SQLite timelines. No GPU measurement was simulated.

### PCM and full-RVC results

| cycle | baseline ms | control ms | prefetch ms |
| --- | ---: | ---: | ---: |
| 1 | 942.9366 | 345.9464 | 343.4304 |
| 2 | 886.51795 | 335.85705 | 336.2987 |

Stage120 prefetch produces a repeatable WRONG output: 119054 / 119200 samples differ from baseline, max abs PCM error 25966 LSB, SNR only 0.227 dB. Control differs from baseline by just 231 samples, max 1 LSB and SNR 96.68 dB.

### Matched C128/K11 NCU data

| metric | control first / second | prefetch first / second |
|---|---:|---:|
| NCU duration (ms) | 5.635 / 5.735 | 8.174 / 5.659 |
| registers/thread | 63 / 60 | 54 / 55 |
| Long Scoreboard, average stalled warps per issued instruction | 2.732 / 2.611 | 3.920 / 3.966 |
| global-load SASS instructions | 53.616M / 53.665M | 63.937M / 63.991M |
| L1/TEX throughput elapsed % | 55.11 / 52.59 | 49.15 / 51.66 |
| L1 hit rate % | 88.15 / 88.02 | 90.76 / 89.09 |
| local loads | 0 / 0 | 0 / 0 |

NSYS isolated geometry (`gridX=932, blockX=32`, 6 calls per compiled variant): control combined GPU time 39.1574 ms; prefetch 49.4198 ms, +26.21%. NSYS confirms the prefetch kernel regression even though wrapper time differed by less than one percent.

**Verdict: Stage120 prefetch REJECTED** — numerical failure and resource regression. The implementation and environment flag are removed from active production code; history is preserved in Git.

## Bottleneck → precise candidate

The remaining unmodified transport control uses 53.6M SASS global-load instructions for two sampled C128/K11 kernels, with Long Scoreboard ~2.6–2.7 stalled warps per issued instruction, and L1/TEX utilization around 53–55%. Not a DRAM saturation diagnosis.

Existing kernel scheduling at Cout=128 and output vector width=4 assigns `vectors_per_position=32` consecutive CUDA units to different groups of output channels of **the same time tile**. All 32 lanes load identical input scalar `input[N,t,ci]` for each of four temporal positions, but different weight vectors. That is a structural duplication in source code. Warp-level hardware can already coalesce identical global addresses, so reducing per-thread load instructions is **not** guaranteed to improve transactions or time.

Stage121 adds opt-in `RVC_NSF_RESBLOCK_PLANE_BROADCAST=1`: when Cout=128, K11, temporal tile4, output vector width4, the first lane loads each shared input scalar and broadcasts through CubeCL `plane_broadcast`; when `PLANE_DIM != 32`, it falls back to ordinary loads. Weight loads, FMA order, activation and fused residual remain unchanged. The older corrupting prefetch has been removed.

Expected effect to test, **not claim**: fewer global-load SASS instructions and fewer Long Scoreboard stalls, weighed against the extra plane shuffle cost. Accept only on matched CUDA wall time + GPU kernel time, strict PCM bit parity versus control, NCU counters and no adverse occupancy/register effect. If no speedup, reject the stage and select a new bottleneck.

## Validation gates

- Rust 1.100.0 `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu,cuda,wgpu --bin tts-with-rvc-rs --bin stage121_plane_broadcast_probe` — PASS.
- Independent bitwise mathematical reference: 2/2 PASS for C128, 32 lane alignment, K3/7/11, D1/3/5, tails, residual.
- Full CubeCL CPU runtime probe: local Linux linking FAILED because `DOCS_RS=1` skips the native Tracel LLVM library; unresolved LLVM symbols at final link. This is an environment/toolchain limitation and does **not** establish backend correctness. On Windows CI the CPU probe runs as an additional non-blocking diagnostic. The user GPU probe is a mandatory runtime gate, executed automatically by the Stage121 runner.
- GPU JIT, PCM equivalence of actual CUDA, Nsight counters and end-to-end timing: require user RTX 4060 run; no fabricated acceleration.

## Delivery contract

One Stage121 GitHub commit after Rust edits and local verification, one current Windows workflow and one CI artifact with Rust CLI, CPU/CUDA probe, Stage121 PS1, automatic NCU CSV extractor, report and workflow contract. Older automatic Stage120 workflow disabled; historical workflows remain preserved. No model files, WAV or indexes in CI artifact. Use the fixed Python `D:\Projects\tts-with-rvc\.venv\Scripts\python.exe` for NCU CSV extraction only; NVTX DLL from the installed conda torch2.6.0 `torch\lib` directory.
