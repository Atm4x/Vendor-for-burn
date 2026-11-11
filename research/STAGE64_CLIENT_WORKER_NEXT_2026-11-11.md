# Stage 64 continuation — NeuroMita native TTS worker/client integration

Start from PR #2 after Stage63. Consume the latest source bundle; do not re-run unrelated Conv1d microbenchmarks. Rust 1.98.1 / Burn 0.22.1. Use one CPU/CUDA/WGPU EXE, no fallback and no Python interpreter for native RVC. User files (voice PTH, HuBERT, RMVPE, FAISS, WAV) are private and must never be committed.

First inspect the Stage63 full-WAV acceptance results archive (`RVC_STAGE63_FULL_WAV_RESULTS.zip`) from the physical RTX 4060 and report the actual warm whole-recording RTF, output duration, PCM parity, memory use, and failures. Gate CPU-FAISS/F0 baseline on full audio; GPU resident alternatives opt-in until numerics, quality and timing are validated. Note graph_replay=false.

Then implement integration **through one typed TTS worker** owned by the NeuroMita application. The worker owns a long-lived `RvcV2Inferencer`; client-facing commands load model with explicit backend, infer PCM16/WAV or audio stream, report progress per chunk, cancel before next chunk, reload, unload, shutdown. Do not create event bus, capability registry or adapter pyramids. The app/service orchestration sits above this worker; model identity and chosen backend are explicit, and error propagation never silently reroutes to CPU or Python. Avoid a separate EXE per backend. Keep output semantics and device ownership clear, with lifetime and cancellation tests.

If user-provided full-WAV benchmarks prove GPU-IVF overhead dominates, keep CPU FAISS as the default rather than blocking the client on speculative optimization. Move real NSF graph capture, kernel-level profiling, AMD hardware validation and higher-precision F0 into independent follow-up R&D tasks, with reproducible criteria. Ask for specific new GPU/long-audio traces only when needed. No claim of final CUDA Graph speedup without source-level integration and physical measurement.

Artifacts: one source-only ZIP, one factual report, next prompt; commit verified changes fast-forward into PR #2; never publish private binaries or weights.
