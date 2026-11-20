# Continue Stage 56 — native RVC, prioritized by real bottlenecks

Repository `Atm4x/Vendor-for-burn`, experimental PR #2, use latest full Git bridge bundle and keep GitHub GET calls minimal. Read `research/STAGE54_NATIVE_RVC_FULL_ROUTE_2026-11-11.md` and `research/STAGE55_NATIVE_CPU_ALGORITHM_RND_2026-11-11.md`; use locally available model/audio assets only, never publish `.pt/.pth/.index/.wav` weights or voice samples.

Already proved on CPU in Stage55: grouped HuBERT positional Conv1d Prepacked parity; 22/22 Rust tests; fast, exact IVF centroid selection for `nprobe=1`; binary-identical RVC output on the 0.35s CrazyMita input; combined Flex+CUDA compile. Do not equate debug CPU timings to GPU.

First, inspect any incoming `RVC_STAGE54_RESULTS.zip` produced on RTX 4060; confirm the full CUDA route, stderr stage profiles, PCM output characteristics, errors and baseline vs Prepacked parity. In the absence of hardware data, continue LLVM/CPU algorithm tests on varying real audio offsets and chunk lengths, but cap expensive debug NSF full-inference runs. A 0.35-second segment may take tens of seconds on virtualized debug CPU.

Follow measurement, not speculative optimizations: audit CPU readbacks between HuBERT, RMVPE, FAISS and encoder; investigate NSF true GPU-only execution vs host dispatch. On CPU, run a properly matched release/optimized baseline vs bounded-IVF experiment before asserting a stable relative speedup; consider `nprobe>1` only if a real supported index needs it. For long audio use production chunked path and verify overlap/batching correctness across boundaries, ideally with deterministic reference outputs. Do not attempt to reuse NSF Quad kernels on incompatible HuBERT/RMVPE convolutions.

Future architecture: independent, long-lived native model executable; explicit backend selection, no invisible fallback; versioned IPC with load/warmup/infer/status/cancel/unload/shutdown; Core only orchestrates. Packaging must not force the end user to install Python, PyTorch or CUDA Toolkit. Keep runtime GPU libraries versioned and compatible.

Deliver source-only ZIP, MD report with separated PASS/NOT TESTED statuses and a next-agent prompt; publish changes to existing experimental PR with verified Git tree and CI. Do not alter obsolete scripts.
