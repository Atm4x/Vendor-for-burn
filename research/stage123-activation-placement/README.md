# Stage123 — RVC40 ResBlock LeakyReLU activation placement

Stage123 is a narrow, **NCU/NSYS-derived** Burn/CubeCL R&D experiment on the C128/K11 ResBlock pair in the RVC NSF decoder (Rust 1.100.0). It is **not** a proven speedup and is disabled by default.

## Mechanism
The existing GPU second Conv1d recomputes LeakyReLU of the first convolution's intermediate for every K×Cin tap and each output-channel group. This same intermediate is private to the second convolution. Stage123 computes its LeakyReLU when the first convolution writes it; the second convolution treats it as already activated. The residual skip uses the untouched original x, and FP32 accumulation order, layout, weights, kernel launch count, and buffer allocation remain unchanged.

Flag: `RVC_NSF_RESBLOCK_ACTIVATION_PLACEMENT=1` (only with transport, Cout128, K11). Control and default behavior remain unchanged. No Stage120 prefetch or Stage121 plane-broadcast is enabled.

## Evidence
The supplied fast-control NCU two kernels execute about 53.6 million global loads each and ~211 million L1 sectors, within +2.2% and +0.6% of their modeled dataflow limits. Their measured ALU utilization ~43–46%, eligible warp issue ~68–74%, and other instruction pressure motivate removing provably redundant activation work, not increasing loads or reducing vectorization. The second kernel in the warm NSYS timeline costs about 9.89 ms out of 341.36 ms warm inference (~2.9%), which is an upper bound on overall gain for this second kernel.

## Build and run (one CI ZIP)
Download `rvc-stage123-activation-placement-windows-x64` from the successful Stage123 Windows CI run. Unzip it and put the existing `embed/` folder alongside the two EXEs. No models/WAV are distributed in the build ZIP.

Preflight: `./Run-Stage123-Activation-Placement.ps1 -CheckOnly`

Full benchmark: `./Run-Stage123-Activation-Placement.ps1 -Cycles 2 -Runs 5 -Profiler all`

The runner first executes **actual CubeCL CUDA JIT bitwise parity** in `stage123_activation_placement_probe.exe cuda` (K11, C128, dilation 1/3/5, time 3/17/71). If it fails, benchmark/profilers never run. The runner then alternates baseline, control (transport NTC tile4), and candidate across independent warmed processes; requires **bitwise identical PCM to control** before NCU and NSYS profiling; produces NCU reports and CSV, full NSYS timeline SQLite, JSON and logs in one results ZIP excluding audio/weights. Submit this single results ZIP for inspection.

Configured tools: NCU 2025.1, NSYS 2024.6.2, CUDA 12.8, installed `nvToolsExt64_1.dll` from Conda PyTorch (DLL only). Only executable Python for Nsight export and PyTorch reference: `D:\Projects\tts-with-rvc\.venv\Scripts\python.exe`. No new installs. Historical workflows remain manually triggerable; current CI is Stage123 alone.

## Verification
Mathematical pair-level Rust reference 3/3 PASS, local Rust 1.100.0 offline cargo check with cpu,cuda,wgpu on the CLI and CUDA probe PASS. **No Windows CUDA JIT, RTX 4060 metrics, or full output PCM has yet been measured for Stage123**. Do not promote unless GPU measurements prove a benefit.
