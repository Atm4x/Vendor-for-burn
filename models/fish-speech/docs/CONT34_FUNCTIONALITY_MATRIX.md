# Fish Speech 1.5 — CONT34 functionality / operability

The scope is the original fixed 22 source-level capabilities for the supplied Fish Speech 1.5 implementation. DONE describes Rust functionality and available local tests, **not** GPU checkpoint parity.

## Progress

- Functionality: **21 DONE, 1 PARTIAL, 0 MISSING / 22 = 97.73%** (historical `DONE + 0.5×PARTIAL` index).
- Detailed conditions: **35/36 = 97.22%**. CONT33 was 32/36.
- Operability: historical **7/11 = 63.64%**, unchanged. Real Windows/RTX, NVRTC kernel launch, TensorRT runtime memory accounting, checkpoint and audio parity were not tested.

| Item | Status | Conditions | Exact remaining work |
|---|---|---|---|
| F01 model config/Firefly | DONE | 5/5 | — |
| **F08 autonomous audio codecs** | **PARTIAL** | **4/5** | Decode MP3/AAC/M4A/WMA without FFmpeg or additional libraries |
| F11 Slow AR prefill and CUDA facade | DONE | 6/6 | — |
| F12 bucketed Slow AR decode | DONE | 7/7 | — |
| F13 Fast AR / semantic GPU sampling | DONE | 6/6 | — |
| F14 Naive AR TensorRT | DONE | 5/5 | — |
| **F22 GPU lifecycle** | **DONE** | **7/7** | — within defined Rust source contract |

### F22 acceptance

- [x] 4 pre-existing CUDA session, stream, KV lifetime and shared-budget conditions.
- [x] One shared GPU device-memory budget for source-controlled CUDA buffers, managed TensorRT context memory and configured TensorRT Builder/Runtime allocator allocations; no double counting of manually owned context workspaces.
- [x] Explicit `warmup_request` / `unload` model lifecycle with bounded per-request contexts and safe retirement.
- [x] Bounded, persistent scratch pool, exact-size reuse across requests, stream-ordered safety, budget retention while idle and verified source-level behavior.

### Remaining operability gates

Actual RTX tests must verify every TensorRT callback, GPU memory footprint, Windows driver compatibility, NVRTC behavior, numerical reference parity and the latency impact of synchronous allocator retirement. These belong to **operability**, not the fixed 22-functionality index; no claim of hardware PASS is made.

F08 cannot be fully implemented from the vendored codec set without writing additional standards-compliant decoders or accepting external codec dependencies.
