# Next agent — Stage 52 hardware proof, then Stage 53

You are continuing `Atm4x/Vendor-for-burn`, PR #2, branch `experiment/conv-kernel-rnd-stage33-20261110` with clean offline Rust 1.98.1. Do not edit obsolete scripts or production NSF kernels. The attached source-only Stage 52 bundle is incremental to remote SHA `e9134c27a796117de75d30c03100a67c28735509`. See `STAGE52_DYNAMIC_CUDA_GRAPH_RND_2026-11-11.md`.

1. Verify `git bundle verify`, import via `git fetch`, preserve the local new commit and the original parent, then publish fast-forward to the experiment branch only after confirming the remote branch has not moved. Never force-push. If it moved, reconcile before publication.
2. Ensure GitHub Actions `RVC NSF Stage52 dynamic CUDA Graph proof package` compiles. Resolve issues in the Stage 52 source only; do not fix by deleting correctness assertions or by silently treating CPU software replay as CUDA hardware capture.
3. Get the resulting Windows ZIP and have the user run `Run-NSF-Dynamic-Graph.ps1` on RTX 4060. Analyze raw results from both rounds, especially `is_hardware`, exact FP32 parity, stale-buffer detection, `refresh_ms`, full replay time, capture cost and stream ordering.
4. If failures are from pitched storage, engineer a stride-aware same-allocation data update preserving pointer identity. Inspect CubeCL `Client::write` and CUDA stream semantics, validate output against independent normal Burn forwards, keep full readback synchronization while proving correctness.
5. After proof, research static-shape graph caching, memory/VRAM cost, break-even number of inferences, real model weight compatibility and warm long-lived worker behavior. No need for user voice `.pth`, HuBERT or RMVPE until starting an actual end-to-end voice conversion test.
6. Maintain a self-contained report and source-only archive per iteration, with exact PASS/FAIL and not-tested accounting. Never claim RTX results from a Linux compile.
