# Stage 59 — next-agent prompt (native RVC, PR #2)

Continue from published Stage 58 in `Atm4x/Vendor-for-burn`, experiment PR #2. Recover the current Git branch via existing CI Git bundle; do not repeatedly GET whole repository through GitHub. Use uploaded Rust 1.98.1, offline Tracel LLVM and vendored dependencies. Do not upload private voice weights, Hubert, RMVPE, FAISS index or WAV; legacy scripts are reference-only.

Stage 58 introduces **opt-in** `RVC_RMVPE_COMPACT_READBACK=1`: device-side exact first-maximum + 9-neighbor selection and finite validation, only 10 vs 360 FP32 elements read back per RMVPE frame. Rust Flex release tests 30 PASS, 2 ignored benchmarks; CUDA all-targets check PASS. Full RVC WAV compared byte-for-byte for 0.35 and 0.7-second real clips with index rate 0 and 0.75. Important: CPU benchmark demonstrates compact mode is slower, so the default remains dense; real RTX CUDA performance is NOT TESTED. Windows Stage58 runner prepared and packaged via GitHub Actions when published.

Next priorities:

1. If RTX 4060 A/B ZIP is available, analyze paired dense/compact RMVPE stage time, logical D2H bytes and exact WAV SHA; ensure correct hardware and same configs. Avoid claims of 36x *speedup* — only 36x smaller readback.
2. Develop a **fused CubeCL kernel** for RMVPE salience decode, combining first-argmax, full finite validation and nine-bin gather in one dispatch per frame. Preserve exact FP32 center selection, tie order and FP64 host F0 centroid. CPU tests for B=1/2/4, shapes, all-zero, NaN/Inf, edge maxima, thresholds; CUDA runtime test when possible. Keep behind experiment gate until measured and proven.
3. Measure whether pure CPU preprocessing is preferable on CPU and whether aggressive GPU fusion outperforms many Burn launches; compare per-stage with comparable workloads.
4. Investigate FAISS `index_rate=0.75` device boundary without changing nprobe=1 IVF Flat exact-neighbor result. Benchmark against Stage55 exact retrieval baseline.
5. Integrate native model via *existing shared worker for the TTS service type*, not per-model Python/native worker switching. No generic event bus, adapter sprawl or model-specific worker forks. Do not change NeuroMita Core without checking the actual shared worker protocol.

Publish source-only ZIP (diff + MD) and next-agent prompt; all tests PASS/FAIL/NOT TESTED explicitly. Commit only relevant source files; safe fast-forward to the existing PR #2 after comparing the actual remote branch HEAD and local expected Git tree. Do not touch main.
