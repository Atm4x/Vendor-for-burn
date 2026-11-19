# Fish Speech 1.5 — CONT31: device-resident Fast AR sampler

## Source provenance

- Local Git starting tree: `0a587a65afa75f18cfef5d2aba8d9a762d527e80` (published GitHub CONT30 commit `379b1a1ba10bce78ec4cf0d7432dc0f462fca899`).
- Source sampling contract: original uploaded `Fish-speech-pipeline-main (1).zip`, particularly `fish_speech_lib/fish_speech/models/text2semantic/inference.py`.
- Everything was developed locally; Git Bridge workflow was neither necessary nor rerun.

## Implementation

1. `core/src/ar_generation.rs`: `DualArBackend::sample_fast_codes` is a backend-specific extension of the existing, single DualAR generation controller. The default CPU path preserves the old per-codebook sampler, validation, Fast step ordering, EOS-completion semantics, cancellation and trace. Invalid sampled codes are rejected before submitting later codebooks.
2. `src/cuda_fast_sampling.cu`: CUDA kernel with shared-memory bitonic sorting and stable tie-break by token ID, once-only repetition penalty on the codebook history mask, source-compatible descending top-p cutoff before temperature, and independently seeded exponential-race categorical selection. Limited deliberately to 4096 acoustic entries to cap dynamic shared memory at 32 KiB. Rejects nonfinite inputs by setting an invalid output-column sentinel while using safe token 0 for later network input.
3. `src/cuda_fast_sampling.rs`: NVRTC compile-once PTX cache, request-context CUDA module, bounded launch configuration, GPU history mask, device token allocation, final contiguous column parse and validation. Uses the existing `cudarc` package with `nvrtc` feature; runtime availability of `libnvrtc` is required and is not validated here.
4. `src/cuda_generation.rs`: Fast AR `decode_fast` is bypassed by the GPU specialized implementation. It enqueues Fast TensorRT networks and sampling kernels on the same CUDA stream, passes the GPU I32 sampled token to the next Fast graph, and downloads the complete acoustic column only once after all Fast codebooks. Per-codebook cancellation checks remain on CPU without synchronizing logits.
5. `src/cuda_trtx.rs`: extend the session with retained CUDA buffers until `retire_completed`, preventing use-after-free of scratch/history/output buffers in asynchronous kernels. `CudaDualArBackend::Drop` retires pending CUDA work while the sampler function/module is still alive, including error/unwind paths.
6. `tools/test_cuda_fast_sampling_host.py`: emulate the unmodified algorithm of the `.cu` kernel with one CPU thread and compare against independent Python sampling mathematics. This tests logic, but not CUDA driver/NVRTC execution or multi-thread scheduling.

## Validation performed

- Core full test suites: 228 PASS, 2 ignored (200 unit + 28 integration).
- Outer Rust crate: 53 PASS, 2 ignored.
- CUDA + TensorRT features: 65 PASS, 2 ignored; compile PASS.
- TensorRT mock: 53 outer unit PASS, 2 ignored, eight integration/blueprint/resident PASS.
- Host kernel sampling parity: 102/102 cases PASS (small/tied logits, negative penalties, top-p variants, acoustic vocab sizes up to 4096).
- Rust formatting and Git whitespace validation: PASS.
- Real NVIDIA execution, NVRTC compilation and native PyTorch CUDA parity: NOT TESTED.

## Progress and remaining scope

F13 advances from 3/6 to **5/6**, but is still PARTIAL because the semantic EOS decision continues to use the CPU Slow sampler. The fixed acceptance matrix improves from 26/36 to **28/36 (77.78%)**. Overall 22-item status remains 18 DONE + 4 PARTIAL = **90.91%**. Operability remains 7/11 = **63.64%**.

The codebook-size limit applies to the new Fast GPU sampler and accommodates original Fish Speech 1.5 (1024 acoustic entries). There is no hidden fallback to CPU Fast sampling when NVRTC is unavailable; instead the CUDA initialization returns an explicit error.

Next: scalable GPU-resident Slow semantic sampler and EOS termination, or move to F14 if real-GPU sampling would require hardware-specific benchmarking first. Do not claim end-to-end speedup without a Windows RTX run.
