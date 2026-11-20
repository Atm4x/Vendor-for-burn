# Stage123 — Bottleneck-derived activation relocation (RVC/Burn/CubeCL)

Date: 2026-11-17. Source: user-uploaded Stage117/Stage121 NCU CSV and Stage121 full NSYS SQLite GPU timeline; exact C128/K11 geometry, NOut4 × temporal tile4. Scope: only the pair of Conv1d operations at 128 channels/K11; no algorithm chosen speculatively.

## Why not prefetch, shuffle, or NOut2?
Stage120 prefetch was rejected: ~99.88% PCM mismatch, up to 25,966 PCM LSB, ~19% increased global loads and ~26% GPU kernel regression.
Stage121 plane broadcast was rejected: PCM bit-exact, but full RVC ~14.6% slower, 53.6M to 178.5M global loads, L1/TEX throughput ~91%, and massive instruction divergence. NOut2 was proposed without proof; predicted doubled lane work and at least +20% L1 sectors, so not implemented.

## Raw evidence for fast control C128/K11
| NCU counter | first Conv1d | second Conv1d |
|---|---:|---:|
| Instrumented GPU Duration | 4.241 ms | 5.239 ms |
| Registers per thread | 48 | 64 |
| Issue active | 74.20% | 68.26% |
| FMA pipe active, elapsed | 48.13% | 44.25% |
| ALU pipe active | 46.29% | 43.24% |
| L1/TEX throughput | 57.62% | 52.98% |
| Long Scoreboard, average stalled warps/issued instruction | 2.22 | 2.74 |
| Global load instructions | 53.616M | 53.665M |
| L1 global-load sectors | 211.006M | 211.560M |
| All warp instructions | 552.847M | 553.524M |
| Divergent branch targets | 0 | 0 |
| Local loads/stores (spills) | 0 | 0 |

### Quantitative dataflow model
For T=29,800, NOut=4, tile4, C128, K11: 7,450 warp groups, 1,408 K×Cin iterations, each with 1 vector weight load and 4 scalar input loads. Predicted global loads: `7450×1408×5 = 52.448M`, measured 53.616M (+2.23%). Predicted L1 sectors: `7450×1408×20 = 209.792M`, measured 211.006M (+0.58%). The hardware is already coalescing identical input addresses across lanes. Further lane broadcast is not justified by these data.

### Warm critical path
Using event-level NSYS, the warm second interval from stage102_rust_hubert to stage104_rust_nsf_readback is 341.363 ms. CUDA kernels consume 321.176 ms, with 19.417 ms gaps. 94.1% GPU busy during warm inference means eliminating the global launch overhead isn't the dominant goal. The C128/K11 set with grid932/block256 costs ~38.08 ms across two runs, or 19.04 ms per pass, ~5.6% of the warm timeline. Second kernels alone cost ~9.89 ms (~2.9%). This is the Amdahl upper bound for work limited to their computation.

## Proven repeated work
Existing mathematical pair:
```
h = Conv1d(LeakyReLU(x), W1) + b1
y = x + Conv1d(LeakyReLU(h), W2) + b2
```
First intermediate h is private to immediate second Conv1d; x is the true skip. Algebraically equivalent:
```
h_active = LeakyReLU(Conv1d(LeakyReLU(x), W1) + b1)
y = x + Conv1d(h_active, W2) + b2
```
The compiler may reduce or restructure the work, so this is a **static dataflow count**, not a speed claim: a scalar intermediate of the second convolution can be reactivated for 32 output-channel groups × 11 taps = 352 lane-level evaluations instead of one when stored by first Conv1d. Approximate count for 29,800 positions: 1,342,668,800 versus 3,814,400 activation evaluations (interior approximation). No extra GPU launch or buffer, no changed weight/input loads, no reordered FP32 FMA reduction. Addition of b1 takes place before the single moved activation; original skip x is never activated.

## Concrete implementation and safety gates
- Opt-in `RVC_NSF_RESBLOCK_ACTIVATION_PLACEMENT=1`, C128/K11 transport only; no production default changes.
- New CubeCL compile-time booleans `input_activated` and `output_activated` in direct preact-residual kernel. If `output_activated`, residual must be absent.
- First Conv1d writes LeakyReLU(biased sum); second receives already-activated input and adds original x unchanged. Existing caller API remains backward compatible.
- Rust independent full pair reference test 3/3 PASS (C1/3/4/7/16/32/128; K3/7/11; dilation1/3/5; short/edge inputs). Offline Rust 1.100.0 feature check cpu,cuda,wgpu of project CLI and probe PASS; does not imply CUDA JIT pass.
- Actual CUDA CubeCL probe must return bitwise identical output at T3/17/71 and dilation1/3/5 before any expensive RVC benchmark.
- Full RVC must preserve PCM SHA against control; otherwise candidate rejected and results archived.
- Compare the paired NCU first+second geometry GPU durations and instruction counters (ALU/FMA, global-load count and L1 sectors, registers, warp issue, Long Scoreboard). A saving in second kernel is not useful if it increases first kernel more.
- Compare warm median and complete CUDA/NVTX NSYS SQLite timeline, not raw aggregated NVTX host times or NCU replay wall time.
- Reject on correctness regression, GPU regression, or unaccounted instruction effects; don't expand geometry without evidence.

### Known limitations
The broad `cargo check --tests` locally hits obsolete Stage30 binary benchmarks calling removed `Device::flex()`, unrelated to these changed crates. The targeted cargo check for Stage123 bins passes. Old benchmark/obsolete scripts deliberately not edited.

### Not yet claimed
No RTX 4060 CUDA compilation/JIT, end-to-end PCM or GPU speedup measured on Stage123. All numerical performance conclusions require the Windows artifact executed on the user's GPU.
