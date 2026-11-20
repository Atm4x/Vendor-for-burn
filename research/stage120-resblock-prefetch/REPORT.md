# Stage120 — ResBlock C128/K11 prefetch R&D

## Source evidence
Stage117 uploaded ResBlock NCU CSV and Stage118/119 full NSYS timeline: Long Scoreboard 4.355/4.811 stalled warps per issued instruction; Math Pipe Throttle 2.457/2.387; registers/thread 39/40; achieved occupancy 97.30%/97.65%; L1/TEX 53.14%/52.44%; DRAM throughput 7.74%/8.78%; L1 hit about 87.8%; zero local spills. This supports investigating dependent memory-load latency, not a confirmed DRAM bandwidth bottleneck.

## Implementation
For Cout=128, K=11 only, add compile-time CubeCL mode RVC_NSF_RESBLOCK_WEIGHT_PREFETCH=1. Fetch first vector of weights before the input-channel reduction loop and fetch next channel vector before MACs on the current channel. Keep [K,Cin,Cout] packed weight layout, NTC input and tile4; same FP32 MAC order and fused LeakyReLU/residual-add.

## Verification
- Rust 1.100.0 offline cargo check (cpu,cuda,wgpu) PASS.
- Independent mathematical reference tests 2/2 PASS for K3/K7/K11, dilation 1/3/5, tails and residuals.
- GPU runtime JIT, register/occupancy response, hardware NCU counters, matched PCM16 and end-to-end RTX timings: NOT YET MEASURED.

## Acceptance and delivery
One PR #2 commit combines Rust/CubeCL and an up-to-date Windows CI with one independent Stage120 PowerShell runner. Old Stage112 and Stage116 automatic push triggers are disabled; historical code is not removed.

Report full warm timings, PCM16 difference versus both baseline and control, NCU GPU Duration/Long Scoreboard/L1/DRAM/registers/occupancy and full NSYS kernel/API timeline. Reject candidate on regression. Do not interpret NCU replay wall time as warm inference.
