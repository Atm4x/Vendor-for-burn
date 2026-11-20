# Stage 57 next-agent prompt — native RVC complete-route R&D

Repository `Atm4x/Vendor-for-burn`; continue the **same** experimental PR #2 from the Stage 56 published HEAD. Obtain one up-to-date Git Bridge bundle if available; avoid repeated GitHub GET/file dumps. Use Rust 1.98.1 and offline vendor packages. Never commit private weights, FAISS index, or WAV. Do not edit obsolete/legacy scripts.

Read `research/STAGE54_NATIVE_RVC_FULL_ROUTE_2026-11-11.md`, `research/STAGE55_NATIVE_CPU_ALGORITHM_RND_2026-11-11.md`, and `research/STAGE56_NATIVE_RVC_RELEASE_AND_CHUNK_RND_2026-11-11.md`.

Verified Stage 56: actual CrazyMita CPU release full pipeline PASS; real-weight baseline/prepacked release WAV parity; CLI duplicate full-audio highpass removed with binary-identical output; expanded 90-s mock chunk tests; 25/25 release library tests; FAISS k/nprobe real-index parity. Attempted exact top-k and seeded IVF search did **not** improve release timing and were not shipped. Do not re-add these variants without new matched evidence.

Work priorities:

1. First inspect any incoming RTX 4060 `RVC_STAGE54_RESULTS.zip`: real-weight full CUDA correctness, per-stage performance and WAV comparison against CPU. Windows CI packaging is compile evidence, not hardware evidence. No silent CPU fallback.
2. Measure roundtrips of HuBERT→retrieval→encoder and RMVPE→F0 and identify a **real** algorithmic opportunity. Propose GPU-resident intermediates only when safe, semantically equivalent and backed by actual measurements. Avoid indiscriminate `Quad` or raw CUDA Graph integration into unrelated layers.
3. Validate longer production chunks, random input offsets, multiple speakers/index rates and variable batch lengths with real inference only when resource/time budgets permit. Preserve deterministic per-chunk seeds and exact sample count.
4. If working on CLI, retain the invariant that high-pass is performed **once** and source RMS matching receives precisely the same filtered PCM as inference.
5. Prepare independent long-lived native RVC process for NeuroMita service-worker integration (discover/load/warmup/infer/status/cancel/unload/shutdown) without coupling individual models or tensor APIs to NeuroMita Core. Avoid overgeneralized event buses and global capability services.

Deliver source-only ZIP, technical MD report distinguishing PASS/NOT TESTED, and next-agent prompt; publish to same experimental PR only after tests and verified Git tree. Python/PyTorch not required for Rust inference; reference Python can be used solely as an independent numerical oracle.
