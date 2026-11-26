# CONT18: bounded TTS admission and AR scratch ownership

## Runtime scope

`request_queue::FishRequestQueue` is a bounded FIFO admission queue for one mutable `CpuFishModel`. It is **not** a second supervisor and does not create a worker thread. The service/host calls `process_next_cpu` serially, using `OwnedSynthesisRequest` so waiting requests do not borrow caller memory.

- Queue limits: number of pending requests and accounted payload bytes (text, reference text, acoustic index payload). These are admission estimates, **not** a process RSS guarantee or a bound on generated PCM.
- `process_next_with` marks one active request; reentrant or concurrent processors receive `Busy` without removing other requests. The queue mutex is released before model execution.
- Callers may clone the queue handle to submit or cancel from other threads. Active cancellation uses an atomic flag, which existing `SemanticGenerator::generate_with_cancel` polls at semantic/acoustic boundaries. Pending cancellation removes the request and releases its admission budget.
- `close()` cancels pending work and marks the active job cancelled; closure of the model is a separate operation after its active call returns.
- `pause_for_replacement()` succeeds only when idle. While its guard is held, enqueue and dequeue are rejected. `reload_cpu()` uses this guard around the existing atomic model replacement and resumes admission on success/failure.
- If a worker panics, the active slot is released but the queue is `Faulted`; pending jobs are cancelled. Only successful `reload_cpu()` clears the fault. Worker errors are returned as outcomes and do not poison the queue automatically.

## Stream scratch

`ar_stream_executor::ScratchLimits` bounds retained network-output allocations and number of in-flight submissions. `enqueue()` performs an output-size preflight before touching the device. It returns a budget error without poisoning a healthy KV stream if a limit is reached. `retire_completed()` synchronizes the same backend stream, releases executor-owned in-flight buffer handles and clears the scratch accounting **without resetting logical Slow/Fast KV lengths**. `reset_request()` synchronizes and also resets logical KV lengths.

The scratch accounting excludes persistent KV buffers, model weights, internal TensorRT engine workspaces, backend-specific allocations, external caller-owned bindings, and output handles the caller retains. The output size is computed from the typed graph plan, not GPU measurement. This is a resource safety gate, not a CUDA memory pool or a claim of constant total VRAM.

## Remaining production work

The Linux prototype still has no CUDA `ArStreamDevice`/`KvStreamDevice`, no real engine enqueue, no stream-ordered scratch allocator or CUDA event-based retirement, no GPU warmup, and no pressure-aware PCM/reference-cache budget. The host must call `retire_completed()` before the scratch budget is exhausted; when integrating CUDA, confirm buffer lifetimes, stream completion and caller-held output handles.
