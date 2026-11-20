# Continue native RVC research after Stage 54

Work in `Atm4x/Vendor-for-burn`, experimental PR #2. First retrieve the latest Git bridge bundle; keep GitHub GET calls minimal and work locally. Never commit private `CrazyMita.pth`, `CrazyMita.index`, `hubert_base.pt`, `rmvpe.pt` or `MitaKind.wav`.

Read `research/STAGE54_NATIVE_RVC_FULL_ROUTE_2026-11-11.md`, the native Rust inference source, and the Stage51–53 GPU A/B results. Do not conflate synthetic NSF Graph with actual full RVC.

The first priority is genuine hardware evidence from `RVC_STAGE54_RESULTS.zip`: check that `--device cuda` processes real HuBERT→RMVPE→FAISS→TextEncoder→Flow→NSF, inspect generated WAV and stderr stage times. Compare baseline/prepacked on the same audio/model and run PyTorch baseline if available. Fix real runtime failures and numerical mismatches. Preserve explicit backend selection without CPU fallback.

Then optimize the measured bottleneck. CUDA Graph replay needs shape-bounded persistent input buffers and safe stream/memory ownership; test changing data across multiple forwards and compare against normal forward. Host-side Conv1d prepack was generalized for NSF, HuBERT frontend, Flow; selected temporal packed/quad kernels remain NSF-specific. Retrieval blending was parallelized frame-wise with deterministic semantics. Avoid coupling models to a central event bus or giant generic inference API.

Deliver source-only ZIP, MD report, next-agent prompt, and verified GitHub commit/CI information. No new dependencies on PyTorch/CUDA Toolkit for end users. No unsupported performance claims.
