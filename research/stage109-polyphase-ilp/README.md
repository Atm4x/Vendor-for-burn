# Stage 109 — Software-pipelined Polyphase ConvTranspose1d

RTX 4060 Stage 108 found no useful compute-kernel speedup from packing NCT→NTC, and modest or mixed impact of KIC alone. Long Scoreboard stalls remain 12–16 per issued instruction, with FP32 FMA utilization near 30%.

This stage adds a four-channel software pipeline to the specialized FP32 polyphase kernel, with two distinct accumulation policies:

- `pipe4`: load four independent input/weight pairs before four sequential FFMA updates. Keeps the original channel accumulation order.
- `ilp4`: same memory loads, but four independent accumulator chains, combined at the end. This changes floating-point reduction order and must pass full-output parity tests before any production use.
- `pipe4_kic`, `ilp4_kic`: variants using weight layout `[K,Cin,Cout]` prepacked once.
- `single` and `kic`: controls from Stage 108.

Run `Run-RVC-Stage109-ILP.ps1 -Backend cuda -Profiler ncu -Runs 12 -NcuLaunchCount 2 -PythonExe 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe'` in the packaged artifact. For more source-level evidence, add `-SourceCounters` (may increase replay time).

The benchmark checks every output element against stock Burn `ConvTranspose1d` on identical inputs. Key NCU metrics: `gpu__time_duration`, Long Scoreboard, LSU wavefronts, eligible warps, FMA throughput, registers and local-memory traffic. Weight prepacking happens once at initialization and is excluded from per-inference timing, consistent with production model loading. Input copy costs are not artificially excluded for an NTC variant; NTC is deliberately not retested here because it lost on both shapes.

The result ZIP includes only benchmark logs/JSON, `.ncu-rep`, comparisons, and diagnostics. No WAV or models. New code is opt-in in probe, not enabled in production decoder. CUDA performance must be measured on an actual GPU; CI does not have one.
