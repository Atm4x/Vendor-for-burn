# Stage 60 — continuation prompt (native RVC)

Start from the published Stage 59 in `Atm4x/Vendor-for-burn`, experiment PR #2. Restore current Git history via CI Git bundle, not dozens of GitHub GETs. Rust 1.98.1, offline LLVM/CubeCL and vendored dependencies are provided. Do not publish private voice `.pth`, HuBERT, RMVPE, FAISS index, or WAV. Legacy/obsolete scripts are read-only references.

Stage 59 introduced optional `RVC_RMVPE_FUSED_READBACK=1`: one fused CubeCL kernel performs all-frame finite check + first strict argmax + nine-neighbor gather, returns 10 FP32 values vs dense 360 (36x fewer D2H bytes), leaves CPU FP64 F0 calculation unchanged. The previous Stage 58 Burn Compact remains via `RVC_RMVPE_COMPACT_READBACK=1`. Default remains Dense. CPU LLVM tests **33 passed**, **4 intentionally ignored**; standalone fused CPU JIT parity on mixed dimensions, ties, edges, bad values and sliced strides passed. Real RMVPE outputs from `rmvpe.pt` and `MitaKind.wav` (0.35s/0.7s, 36/71 frames) bitwise equal F0 vs Dense, all four thresholds. CubeCL CPU benchmark: fused much faster than Burn Compact, slower than Dense; **no CUDA runtime result**. CUDA all-targets compile check passed. Stage59 report and source-only diff in repo.

Priorities:
1. RTX 4060 Windows artifact `rvc-stage59-rmvpe-fused-windows-x64`: compare Dense, Burn Compact, Fused, with exact WAV SHA parity and stage profiling; 0.35/0.7/2 seconds and index_rate 0/0.75, stable order-reversed rounds. Capture output ZIP.
2. Before auto-enabling anything, separate GPU kernel time from host dispatch/readback via CUDA Events or equivalent. The one-thread-per-frame fused kernel might underutilize RTX 4060; if so, develop warp/block-per-frame with full finite/tie/F0 parity.
3. Examine FAISS D2H/H2D boundary; no approximate search or changed `nprobe` without explicit proof. Stage55 exact FAISS is baseline.
4. Continue native service integration with NeuroMita's **one shared TTS worker**, without per-model workers or Python/native switching. Use actual worker protocol; don't modify Core blindly.
5. Keep reproducible Rust CPU/CUDA tests and all perf baselines, not just synthetic NSF. Publish source-only ZIP, report and NEXT prompt, safe fast-forward into PR #2. Do not merge to main.
