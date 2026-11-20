# Stage 63 continuation — hardware parity and native worker

Continue `Atm4x/Vendor-for-burn` PR #2 after Stage 62. Use the latest Git Bridge bundle/commit and source-only ZIP. Rust 1.98.1, vendored Burn/CubeCL. Never publish private model weights, FAISS index or WAV. Do not modify legacy benchmarks. Keep a single user-facing Windows EXE with runtime CPU, CUDA and WGPU selection; no compile-time type aliases or implicit CPU fallback.

Stage 62 changes: shared `burn_cubecl::Cube` backend type, default `cpu,cuda,wgpu`, WGSL NaN compile fix via scaled inverse-square distances, actual synchronized GPU-IVF profiler, `RVC_GPU_F0=1` experimental no-host-salience path, and `infer-wav-segment-benchmark` reporting load/first/warm inference in the same process. Stage62 PowerShell runner uses `./embed` assets and compares CPU FAISS / GPU IVF / GPU IVF+GPU F0. Only count compile/test/hardware evidence actually present in report.

Priority gates:
1. Obtain `RVC_STAGE62_RESIDENT_RESULTS.zip` from a physical NVIDIA GPU on CUDA and WGPU; confirm JIT shader compilation and all outputs. If failed, repair root launch error, do not just catch final `into_data` panic.
2. Diff intermediate GPU/CPU IVF center IDs, 8 neighbors, distance weights and feature tensors; diagnose 9 PCM16 max error and refine precision/tie policies. Establish tolerance over multiple real chunks, not one 0.7 s segment.
3. Verify F0 GPU against CPU FP64 at frame boundaries, nonfinite input, thresholds, coarse pitch crossing; treat `RVC_GPU_F0` as opt-in until real WAV comparisons and error propagation are addressed.
4. Profile true warmed end-to-end, CUDA Events, WGPU timestamps, graph capture eligibility and GPU memory. Long-lived worker must avoid reloading ~195 MB IVF per request.
5. Final architecture for NeuroMita Rust: independent typed TTS service worker with explicit device selection, lifecycle/cancel/shutdown and model reload; one EXE, no adapter chains, no implicit fallbacks.
6. If possible physical AMD hardware: WGPU/D3D12 path, explicit adapter/device identity and feature checks; ROCm separately only if required, not installed by default.

Output one source-only ZIP, one factual MD report, one NEXT prompt. Publish after verification by fast-forward to PR #2, never force push.
