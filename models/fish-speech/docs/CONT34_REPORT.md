# Fish Speech 1.5 — CONT34: F22 GPU memory/lifecycle closure

## Scope and result

F22 reaches **7/7 source-level acceptance conditions**. This does **not** claim hardware, checkpoint, numeric parity or performance readiness.

- Functional status: **21 DONE + 1 PARTIAL / 22 = 97.73%** by the fixed historical 0.5-PARTIAL rule.
- Detailed fixed conditions: **35/36 = 97.22%**, up from 32/36.
- Operability: **7/11 = 63.64%**; no RTX, TensorRT-RTX, NVRTC or Windows run in this environment.
- Remaining functional item: F08, autonomous MP3/AAC/M4A/WMA codecs without external FFmpeg/libraries.

## Changes

### F22-1. Unified TensorRT / CUDA memory budget

- Extended vendored `trtx` 0.9.1+RTX 1.7 to expose `ICudaEngine::getDeviceMemorySizeV2()` and `createExecutionContext(kUSER_MANAGED)`; synchronized vendor checksum metadata. The existing static-context API remains unchanged.
- Context device memory is allocated through `CudaWorkspace`, which owns a `DeviceMemoryLease` from the same `DeviceMemoryBudget` that already counts model weights, KV and CUDA output buffers. A context's workspace is reserved **once**; TensorRT is not also asked to allocate it internally.
- `BudgetedTrtAllocator` implements the existing `trtx::interfaces::AllocateGpu` interface for TensorRT Builder and Runtime allocations. It tracks CUDA device pointers, requested sizes, alignment/padded size and budget leases; supports reallocation, conservative synchronous retirement and failure cleanup. Budget exhaustion fails the allocation instead of silently exceeding the limit. Idle scratch is evicted before retry.
- The new `with_resident_engine_sessions_managed` owns engines, contexts, workspace buffers and runtime allocator for the entire request, synchronizes before destruction, and preserves request-local context isolation. The legacy static cached path remains deprecated and is **not** the production CUDA model route.

**Limits:** CUDA primary-context overhead, graphics-driver allocations and any TensorRT allocations that bypass its configured `IGpuAllocator` are outside the explicit budget. This is an API-level accounting contract, not measured total VRAM usage. Builder/runtime allocator callbacks and `setDeviceMemoryV2` were not exercised on physical hardware here.

### F22-2. Warmup and unload

- `CudaSemanticGenerator::warmup_request(prompt, max_new_tokens)` validates the request and weights, prepares the plan cache, compiles and loads NVRTC samplers, builds or loads serialized TensorRT engines, and instantiates then retires request-scoped, budgeted contexts.
- `CudaSemanticGenerator::unload()` releases idle CUDA scratch storage after synchronization, clears prepared plans, evicts in-memory serialized engines and resets chunk indexing, then checks for outstanding budgeted allocations. Disk engine files are intentionally preserved for cold restart speed. Active inference is excluded by mutable generator ownership.
- Added memory/scratch usage accessors for lifecycle diagnostics.

### F22-3. Bounded CUDA scratch reuse

- Introduced `BoundedScratchPool<T>` and `CudaScratchPool`, which reuse exact-size transient CUDA output buffers within and across requests.
- Resident cache capacity defaults to `min(device_memory_limit_bytes / 8, 64 MiB)`, with FIFO eviction under pressure. Cache entries **retain** their device-memory leases so idle storage remains budgeted.
- Same-stream reuse needs no host synchronization: CUDA stream ordering suffices. Cross-stream reuse synchronizes the source stream; evictions and unload synchronize before physical release. On a synchronization fault, unsafe reuse is rejected and the unreleasable allocation is intentionally leaked rather than prematurely freed.
- On budget pressure, idle buffers can be drained before retrying CUDA allocations. GPU workspaces remain distinct from general-purpose pooled scratch.

## Verification

| Gate | Result |
|---|---|
| Rust Core unit tests | 202 PASS / 2 ignored |
| Rust Core integration suites | 28 additional PASS (230 total Core PASS) |
| Main Rust crate | 58 PASS / 2 ignored |
| TensorRT mock | 58 library PASS / 2 ignored; 4 resident-context tests PASS, other integration suites PASS |
| CUDA + TensorRT feature tests | 80 library PASS / 2 ignored; all feature integration suites PASS |
| `cargo check --offline --locked --features 'cuda-driver tensorrt-api'` | PASS |
| Rust formatting, `git diff --check` | PASS |
| Physical RTX allocator callbacks / stream launches / actual VRAM accounting | NOT TESTED |

New regression tests cover bounded scratch retention/eviction, leases remaining charged when cached, same-request plan-cache reuse and unload invalidation, mock resident-session lifetime and final synchronization, and checked TRT device allocation alignment.

## Remaining work

1. F08 remains PARTIAL (4/5) pending autonomous MP3/AAC/M4A/WMA decoders; intentionally not implemented with outside dependencies.
2. Operability gate: run the Windows/RTX runner with actual 1.5 checkpoint, verify TensorRT RTX availability, real `IGpuAllocator` callbacks, `kUSER_MANAGED` buffer sizes, cross-stream buffer reuse, GPU budget/high-water accounting, successful unload, and PyTorch parity.
3. Performance: benchmark the synchronous TensorRT callback allocator against the native allocator and, if relevant, implement stream-ordered deferred frees with leases retired by CUDA events. Do not weaken the memory-lifetime contract to chase latency.

No dotnet or Unity builds. No new Git Bridge export needed when the local repository history is already current.
