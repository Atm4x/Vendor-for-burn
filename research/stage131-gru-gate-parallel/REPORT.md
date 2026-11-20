# Stage131 — Gate-parallel persistent RMVPE BiGRU (R&D)

Scope: `models/tts-with-rvc-rs/src/rmvpe_persistent.rs` and route setup. Parent: Stage127, includes experimental Stage129 coalesced and Stage130 split-K routes. Source is based on Git Bridge PR #2.

## Why this algorithm

Persistent GRU processes time sequentially (the recurrence is inherently dependent). Current baseline launches `2 × batch` CTAs, each with 256 lanes, and each lane accumulates three independent 256-element FP32 dot products in sequence. Stage130 split-K divides each dot product across 2 or 4 shards but introduces shared-memory reduction and changes FP32 summation order. Both approaches use 4 CTAs at batch=2; neither can saturate an RTX4060's SMs.

Gate-parallel instead maps `(gate ∈ {reset,update,candidate}, output ∈ 0..255)` to 768 lanes per CTA. Each lane executes the same original 256-element FP32 summation in the same order; a single CTA barrier after gate projection makes the three 256-element vectors visible. The first 256 lanes update the recurrent hidden state; a second CTA barrier commits state before the next time step. Both weight reads and state loads are coalesced across each gate's neighboring lanes. One-time `[2,768,256] → [2,256,768]` weight prepack is performed at model load when opted in. Input projection is unchanged.

Static work comparison per `(batch, direction, frame)`:
- Baseline: 256 threads × (3 × 256 FMA per lane); one shared hidden array; two barriers per recurrent step.
- Gate-parallel: 768 threads × 256 FMA per lane; shared hidden + shared gate sums; two barriers per step; same mathematical arithmetic and FP32 reduction order.
- Split-K2: 512 threads × (3 × 128 FMA per lane) plus reductions; shared partial arrays; FP32 reduction order changes.
- Split-K4: 1024 threads × (3 × 64 FMA per lane) plus reductions; shared partial arrays; FP32 reduction order changes.

The gate-parallel 768-thread block might still lose due to register limits, CTA scheduling and low occupancy. **No speedup is claimed until RTX4060 NSYS/NCU verifies it.**

## Opt-in routing

Set `RVC_RMVPE_GRU_PERSISTENT=1`, `RVC_RMVPE_GRU_WEIGHT_COALESCED=1`, `RVC_RMVPE_GRU_GATE_PARALLEL=1`, and unset `RVC_RMVPE_GRU_COLUMN_SPLIT`. The route refuses incompatible settings. Original persistent kernel unchanged; GPU operation semantics remain FP32. Benchmark JSON records layout, split and gate-parallel selection.

## Acceptance protocol

`Run-Stage131-GRU-AB.ps1 -Runs 3 -Cycles 2 -Profiler all` runs 5 configurations in alternating order: control, coalesced, gate_parallel, split2, split4. Inputs, model checkpoint, index rate, core frames and tensor batch remain identical. It requires bit-exact PCM for coalesced and gate_parallel, >=70 dB PCM SNR for split-K routes, synchronized warm medians, Nsight Systems warm CUDA event coverage and targeted Nsight Compute counter capture. Profiling uses 90s WAV NSYS and short-audio NCU replay as separate measurements. Missing target GPU kernels and nonzero tooling exits fail closed; it stores a ZIP without WAV/models.

The Stage127 actual RTX4060 warm trace shows ~559.081 ms GPU kernel time for 17 GRU launches (`implicit_sync`) and ~492.116 ms for 17 launches (`ntc_combined`). Even perfect GRU elimination would save <=0.56s of 12.48s, so this experiment is bounded. Main full-RVC R&D remains NSF ResBlocks (~3.68s warm GPU kernels), not GRU-only microbenchmark vanity.

## Local proof / limits

- Rust 1.100.0 standalone mathematical model: 4/4 tests PASS; gate-parallel matches sequential accumulation bitwise on f32 synthetic cases.
- Python route and NSYS/NCU parser contract: 4/4 PASS.
- GPU correctness/throughput on RTX4060: **not measured locally**.
- CPU CubeCL JIT kernel execution: separate gate; the bundled LLVM 23.1 was required to link; results must be recorded before further claims.
- No automatic promotion to default until Windows CUDA JIT, audio parity and warm NSYS complete.
