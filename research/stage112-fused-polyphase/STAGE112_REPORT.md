# Stage 112 — Fused two-tap polyphase kernel (R&D)

## Evidence-driven motivation

- Stage 110 `ilp4` was measured on RTX 4060 at 0.0636 ms (stride 8) and 0.1000 ms (stride 10) synchronized call median.
- Stage 111 TF32×3 explicit phase GEMM regained numerical accuracy but measured 0.2039 ms and 0.2240 ms, respectively. The NCU gather and split kernels alone consumed >80 µs for stride 8.
- Reusing values *within one thread* (Stage 110 tile2) decreased CUDA blocks but raised LSU utilization and long scoreboard stalls. Stage 112 reuses values *cooperatively within a CTA* and retains 128 threads per CTA.

## Algorithm

Two taps per output stride phase follows exactly from `K=2*stride`. For output phase p, phase-time q, and output channel o, the source indices are `(p+padding)/stride+q` and the predecessor. Weights are `[Cin,K,Cout/groups]` prepacked once. Each block handles `[16 temporal positions × 32 output channels]` with 4 warps. One reduction iteration stages `[2,16,32]` input values (4KiB) and `[2,32,32]` weights (8KiB) in CUDA shared memory, with `sync_cube()` around use. Each lane accumulates four independent output times, and bias is fused into the final output store. No im2col, gather or scatter buffers are allocated.

## Scope and tests

- Rust 1.100, `cpu,cuda,wgpu` cargo check of Stage112 probe (pending or result supplied in release report).
- Standalone Rust mathematical reference tests `reference.rs` prove eight shapes, including grouped convolution and odd channel counts.
- `compare_stage112.py` rejects missing cases or mismatched parity.
- Windows CI performs CPU parity for grouped2 and odd2, which specifically exercise shared-memory boundaries.
- RTX 4060 kernel latency / correctness and matched cuDNN numbers **not yet measured**. The new kernel is strictly experimental; no production switch.

## Runtime caveats

`cargo check` verifies Rust/CubeCL types but does not ensure successful CUDA JIT or shader barriers on hardware. The runner checks all modes before NCU. If a GPU runtime parity failure occurs, do not increase the tolerance: inspect the actual output mismatch and the emitted CUDA code.
