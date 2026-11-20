# Stage 59 — one-dispatch fused CubeCL RMVPE F0 salience compact readback

Date: 2026-11-11. Repository: `Atm4x/Vendor-for-burn`, experimental PR #2. Baseline: published Stage 58. Rust 1.98.1, vendored Burn 0.22.1 / CubeCL 0.11.3, Tracel LLVM 23.1.0. This is an **opt-in R&D path, not a production speedup claim**.

## Implementation

Stage 58 compact readback computes the same nine-bin F0 neighborhood through multiple Burn tensor operations (`max_dim`, equality/masking, `min_dim`, `is_finite`, `all_dim`, `gather`, `cat`). Stage 59 adds `rmvpe_fused.rs`, a **single-dispatch CubeCL kernel** that independently processes each 360-bin frame:

1. Sequentially inspect all 360 FP32 bins, recording the **first** strict maximum (`>` resolves ties toward the lowest index) and rejecting NaN, +Inf, or -Inf anywhere in the frame.
2. Write center and the nine neighboring raw FP32 values, clamping out-of-range coordinates exactly like Stage 58.
3. Transfer 10 FP32 elements per frame, 40 bytes rather than 1440 bytes (36× fewer logical returned bytes). The original CPU FP64 weighted-centroid F0 conversion and per-item threshold semantics remain unchanged.
4. Preserve input buffer strides, including time slicing from a 32-frame-padded RMVPE output. Validate input backend, DType F32, non-tiled layout, buffer bounds, and overflow before unsafe kernel dispatch. The same CubeCL client handles launch and blocking output readback.
5. **No implicit backend switch:** setting `RVC_RMVPE_FUSED_READBACK=1` on an incompatible Flex tensor returns an explicit error; it does not silently fall back to the dense or Stage 58 compact implementation.

Selection: unset `RVC_RMVPE_FUSED_READBACK` and `RVC_RMVPE_COMPACT_READBACK` for normal Dense; set `RVC_RMVPE_COMPACT_READBACK=1` for Stage 58 Burn Compact; set `RVC_RMVPE_FUSED_READBACK=1` for Stage 59 fused CubeCL. Do not enable fused or compact in production by default before GPU measurements.

The current kernel uses **one sequential thread per frame**. One dispatch eliminates cross-kernel launch overhead but does not guarantee good CUDA occupancy or coalesced memory accesses. A warp-per-frame kernel may be worthwhile after hardware profiling.

## Verified checks

- `cargo test --release --offline --locked -p tts-with-rvc-rs --features flex,cpu --lib`: **33 PASS, 0 FAIL, 4 ignored by design**. The ignored items are two performance benchmarks and two input-restricted experiments.
- `cargo check --offline --locked -p tts-with-rvc-rs --features flex,cuda --all-targets`: **PASS**.
- Standalone native CubeCL CPU/LLVM execution (not just compilation): direct fused results match Stage 58 Burn Compact elements and original Dense FP64 F0 bit-for-bit for B=1/2/4; T=1/7/33; thresholds 0/0.03/0.5/1/2; edge bins, ties and zero frames.
- Noncontiguous `[2,7,360]` time slice from `[2,32,360]`: exact Dense-vs-Fused F0 parity. NaN / +Inf / -Inf in non-peak bins: correctly rejected. Flex primitive: explicit backend error.
- Actual `rmvpe.pt` on the real `MitaKind.wav` input. RMVPE executed through Flex, its real salience passed through native CubeCL CPU fused decode; F0 matched Dense bit-for-bit for 0.35 s at 0 (36 frames), 0.7 s at 22.4 s (71 frames), thresholds 0, 0.03, 0.5, 1.0. Separate opt-in test **PASS**. The private model and audio are **not** committed.

**Not performed:** actual RTX 4060 CUDA runtime execution, three-way WAV parity on Windows, full end-to-end RVC on CubeCL CPU, and production enablement. Compilation and CPU numerical proof are not hardware performance evidence.

## LLVM / CubeCL CPU microbenchmark

Release, separate opt-in test, alternating three-mode order with three warm-up rounds and 12 measured rounds per mode. All modes include blocking readback of their own result; they differ in bytes transferred and number of operations. These are one local CPU measurement series, not a statistical performance guarantee.

| Batch | Frames | Dense (ms) | Burn Compact (ms) | Fused CubeCL (ms) |
|---:|---:|---:|---:|---:|
| 1 | 35 | 0.04601 | 0.49408 | 0.08698 |
| 1 | 70 | 0.04274 | 0.71198 | 0.14175 |
| 1 | 200 | 0.05110 | 1.01214 | 0.09943 |
| 4 | 35 | 0.04181 | 0.91043 | 0.09515 |
| 4 | 200 | 0.06952 | 1.90204 | 0.13023 |

**Interpretation:** fused is 5–15× quicker than multiple Burn Compact dispatches on this CubeCL CPU, but 1.9–3.3× slower than plain Dense CPU readback. Keep Dense as the default. The 36× figure is **bytes**, never execution-time speedup.

## Windows CUDA package

`Run-RVC-RMVPE-Fused-Stage59.ps1` alternates Dense, Burn Compact and CubeCL Fused per round, checks process exit codes, stage readback records, logical byte counts and **full WAV SHA-256 equality**, and emits `RVC_STAGE59_RMVPE_FUSED_RESULTS.zip`. This script consumes the user's own `CrazyMita.pth`, `hubert_base.pt`, `rmvpe.pt`, `CrazyMita.index`, and WAV locally, without copying any of these into GitHub.

`.github/workflows/rvc-rmvpe-fused-stage59.yml` builds a CUDA-enabled Windows Rust executable, checks PowerShell parser errors before packaging, and publishes source-only runnable Windows tooling. CI runs without the user's private weights or RTX hardware. A successful Windows workflow is a compilation/package proof only.

## Next

1. Run Stage 59 on the RTX 4060 for 0.35/0.7/2 seconds, `index_rate=0` and `0.75`, multiple rounds. Compare stage times, warm latency, memory and the complete WAV hashes across all three modes. Do not claim improvement from logical bytes alone.
2. If fused is slower on CUDA, profile a warp-per-frame/coalesced reduction and controlled stream policy; only change the algorithm when finite/tie/boundary/F0 parity remains exact.
3. Investigate FAISS `index_rate>0` D2H/H2D boundary separately; preserving the exact `nprobe` nearest-neighbor order and blend semantics is mandatory.
4. Continue native package / worker integration through the **existing shared worker for the TTS service type**; do not add one worker per model or a Python/native worker-selection layer.
