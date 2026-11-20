# Stage 61 — Native RVC next agent

Start with published Stage60 from `Atm4x/Vendor-for-burn`, experiment PR #2. Use CI Git bundle to restore Git refs instead of repeated GitHub GET calls; verify source tree SHA. Rust 1.98.1, vendored Burn/CubeCL; Linux LLVM available. Do not edit unused legacy scripts or publish private CrazyMita/HubERT/RMVPE/.index/WAV.

Stage59 RTX4060: 6/6 full-RVC CUDA runs, exact WAV SHA across Dense/Compact/Fused. Median RMVPE Dense=863.5ms, Burn Compact=942ms, Fused=868ms (two rounds, not significant). Full cold time ~4.04s independent of mode. Dense stays default.

Stage60 adds CUDA-only opt-in `RVC_RMVPE_PLANE_READBACK=1`. One warp/plane per F0 frame uses plane_max/min/any, preserves first-maximum ties and 10-FP32 compact output. CubeCL CPU/LLVM JIT cannot lower plane operations; CPU mode must fail explicitly. Host simulated lane reduction matched actual CPU sequential fused and bitwise Dense F0 (35/35 library tests); real RMVPE salience on 36/71 frames also matched. CUDA all-target compile check passes. **No RTX runtime for plane kernel**; do not claim actual GPU parity or speedup.

Priorities:
1. Run Stage60 Windows artifact `rvc-stage60-rmvpe-plane-windows-x64` on RTX 4060 (real CrazyMita `.pth`, HuBERT, RMVPE, FAISS, MitaKind WAV). Test full WAV SHA parity Dense/Compact/Fused/Plane with `Run-RVC-RMVPE-Plane-Stage60.ps1` at 0.35,0.7,2s; index_rate 0 and 0.75, reversed rounds. Save output ZIP. If plane throws JIT errors, diagnose GPU-only kernel; do not enable fallback.
2. Build a matched isolated CUDA Events/NSight test of salience readback versus full RMVPE and end-to-end resident model inference to identify actual bottlenecks; two-round cold timings are insufficient.
3. Optimize only if real GPU measurements support it. Higher leverage may be remaining HuBERT→FAISS→TextEncoder D2H/H2D and multi-stage GPU dispatch, not just F0 readback.
4. Keep CPU/LLVM numeric tests, CUDA compilation, release exact WAV and private model outputs separate from published source-only ZIP, technical report and NEXT prompt. Publish via verified Git tree and fast-forward into PR #2, not main. Respect NeuroMita one worker per **service type**; no per-model worker explosion.
