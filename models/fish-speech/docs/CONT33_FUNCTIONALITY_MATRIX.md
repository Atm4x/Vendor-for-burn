# Fish Speech 1.5 — CONT33 functionality and operability matrix

Evaluation basis: actual supplied Fish Speech Python 1.5 source and original fixed 22-component scope. DONE means Rust source-level functionality and available local tests, **not** proof of real hardware execution.

## Progress

- Source functionality: **20 DONE, 2 PARTIAL, 0 MISSING / 22**; historical equal-weight DONE + 0.5×PARTIAL index **95.45%** (CONT32 93.18%).
- Detailed fixed acceptance conditions: **32/36 = 88.89%** (CONT32 29/36 = 80.56%).
- Operability: historical **7/11 = 63.64%**, unchanged. Windows/RTX, NVRTC launch, TensorRT engine build on real hardware, actual trained checkpoints, audio/latency and PyTorch numerical parity remain unverified.

| Item | Status | Gates | Exact remainder |
| --- | --- | --- | --- |
| F01 Firefly / model config | DONE | 5/5 | — |
| F08 offline audio codecs | PARTIAL | 4/5 | Autonomous MP3/AAC/M4A/WMA decode without FFmpeg or separate libraries |
| F11 Slow AR prefill and public CUDA facade | DONE | 6/6 | — |
| F12 Slow AR decode / bucket profiles / lifetime | DONE | 7/7 | — |
| F13 Fast AR and GPU semantic sampling/EOS | DONE | 6/6 | — |
| **F14 Naive AR** | **DONE** | **5/5** | — at source-contract level |
| F22 GPU resources and lifecycle | PARTIAL | 4/7 | Engine workspace/unmanaged allocations; warmup/unload; bounded scratch pool |

## F14 closure

- [x] NaiveTransformer's independent codebook logits tensor and its weight/output shape contracts, without creating Fast AR weights or Fast KV (pre-existing baseline).
- [x] Source-specific Naive semantic/acoustic sampling and EOS/cancellation semantics on CPU (pre-existing baseline).
- [x] Resident TensorRT Naive Slow prefill/bucketed decode, retaining semantic and acoustic outputs and GPU KV ownership for a request.
- [x] Public `CudaFishModel` factory accepts Naive checkpoint/config, validates required weights before accessing the GPU, and uses the common TTS streaming/synthesis facade.
- [x] Device-side semantic sampling, codebook-local GPU sampling from contiguous acoustic logits with checked offsets, EOS completion after the last codebook and cancellation, plus source-compatible history handling.

## F22 outstanding gate breakdown

- [ ] Account for TensorRT workspace and unmanaged CUDA memory in the same device budget (without double accounting).
- [ ] Explicit reliable model and engine warmup/unload lifecycle.
- [ ] Bounded scratch-buffer pooling and proven cross-request reuse with safe stream retirement.

## Verification limitations

Linux Rust `cargo check` and host C++ simulation verify type contracts and kernel mathematical behavior only. The generated CUDA kernel launch ABI and execution, engine compatibility, checkpoint-backed inference and parity remain NOT TESTED and belong exclusively to the operability tracker. F08 remains PARTIAL rather than substituting FFmpeg for autonomous codecs.
