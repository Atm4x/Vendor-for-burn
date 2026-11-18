# CONT13 handoff — Fish Speech 1.5 Rust/TensorRT-RTX

Source of truth: `FISH_SPEECH_CONT12_MODELS_FISH_SPEECH_SOURCE_ONLY_2026-11-18.zip` (see `docs/CONT12_REPORT.md`); original Fish Speech 1.5 Python archive, original design spec, vendored `trtx` 0.9.1 RTX 1.7, Rust 1.100.0. Keep work local; do not edit legacy Python. Do not assume a real CUDA driver/SDK is installed.

Actual CONT12 improvement: `ar_stream_plan.rs` partitions Slow/Fast AR around paired physical KV writes; `trtx_lowering.rs` uses real `add_cast(kBF16)` at segment outputs; `kv_device_owner.rs` owns stable GPU cache abstractions and monotonic cursor; `ar_stream_executor.rs` tests same-stream sequence and retains in-flight buffers. Two-layer AR fixture: 3 network partitions, 2 K/V pairs, 4 copies. **No production GPU driver or TensorRT engine execution exists.**

Tests: Core 186/186, outer 3, exporter 5, fmt PASS, `tensorrt-api` check PASS; tensorrt-mock not completed (slow trtx-sys). No GPU tests. No GitHub commits.

NEXT: (1) Audit `ArStreamPlan` for output binding, alias, memory lifetime correctness, arbitrary layer count and mixed Slow/Fast reset, with fault injection. (2) Introduce a strict, bounded device buffer arena/reuse strategy that preserves ordering without per-token CPU sync. (3) If real Windows TensorRT-RTX 1.7 SDK and CUDA runtime/runner become available, implement a concrete stream backend with `cudaMemcpy2DAsync`, a one-layer prefill/decode smoke and exact metadata-driven engine cache. If not available, do not invent GPU results; improve Linux contracts/tests. (4) In parallel, verify authentic checkpoint conversion, tokenizer IDs, and reference waveforms when artifacts arrive.

Do not call `KvUpdate` a native TRT primitive: the mutable cache update is an external CUDA operation. Do not compile one engine per token; no full-capacity K/V clone/concat; GQA reads only active-prefix; BF16 conversion must happen before copy; Fast cache resets per Slow step; no host synchronization every token. Explicitly report actual work remaining across R1–R6 of CONT12 report.
