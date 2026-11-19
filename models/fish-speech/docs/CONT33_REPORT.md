# Fish Speech 1.5 — CONT33: native Naive AR TensorRT path (F14)

## Evidence and source

- Exact clean Git Bridge baseline `dd4c6f2e4f77639f6fb0e81d7fdbafe6a0223b5e`, recovered because the available CONT26 bundle was older than GitHub HEAD. The new archive's `source-head.txt` matched that baseline; SHA-256 was verified. No further bridge export is necessary after publishing because the full local history is available.
- Original Python source: attached `Fish-speech-pipeline-main (1).zip`, particularly `models/text2semantic/llama.py` (`NaiveTransformer`) and `inference.py` (`decode_one_token_naive`, `generate`).
- `NaiveTransformer.forward_generate` returns semantic logits and a *single* `codebook_logits` tensor `[1, 1, num_codebooks × codebook_size]`. The semantic sampler uses fixed temperature `0.1`, top-p `0.1`, penalty `1.0`; each independent acoustic codebook uses the request's sampling parameters and its own history. Even when semantic EOS is selected, finish the whole acoustic column before stopping.

## Source changes

1. `src/cuda_naive_generation.rs`: `PreparedNaiveArPlans` compiles one Slow prefill plus bounded, bucketed Slow decode, with both semantic and acoustic outputs retained through lowering. Uses the same resident TensorRT and KV session as DualAR, but **never** constructs a Fast AR transformer or Fast KV. Actual decode positions are supplied via dynamic RoPE and attention-mask inputs.
2. `src/cuda_naive_generation.rs`: Naive generation path dispatches the Slow graph once per semantic/acoustic column, samples the semantic token and every acoustic codebook on the GPU, returns only the small column and EOS result. Per-codebook history is source-compatible (skip first generated column, take the last 16 tokens and pad with zeros). Cancellation is checked before the Slow step and between codebooks. Sampled token bounds and EOS consistency are checked before appending.
3. `src/cuda_fast_sampling.{rs,cu}`: accepts checked logits offsets, allowing codebook-local GPU sampling from one contiguous Naive acoustic logits tensor. The existing DualAR Fast path supplies offset zero and remains source compatible.
4. `src/cuda_semantic_sampling.{rs,cu}`: optionally writes the semantic token directly into the device-resident output column without requiring a CPU round trip before sampling the acoustic codebooks. The original DualAR path remains supported.
5. `src/cuda_model_runtime.rs`: public `CudaFishModel` now routes both DualAR and Naive through the common `SemanticGenerator`/`FishModel` facade, shared engine registry, memory budget and bounded plan cache. Validates Naive weights from all prepared TensorRT plans before CUDA initialization. Preserves Python Naive behavior for `max_new_tokens=0` and truncated overlong generation, subject to configured GPU-profile cap. `src/lib.rs` exposes the new module only under CUDA+TensorRT features.
6. Both host-emulation scripts updated for the extended kernel signatures and device semantic-column output. These compile the actual `.cu` mathematical algorithms as CPU C++ shims and compare their outputs with independent Python reference math.

## Verification performed locally

| Gate | Result |
| --- | --- |
| Rust toolchain | 1.100.0, installed from supplied tar |
| Rust default crate `cargo test --offline --locked --lib` | 53 PASS; 2 ignored |
| Rust core package full `cargo test --offline --locked -p fish-speech-core-prototype` | 230 PASS; 2 ignored across unit/integration |
| CUDA+TensorRT `cargo check --offline --locked --features cuda-driver,tensorrt-api --lib` | PASS (`LIBCLANG_PATH` set to available Clang 17) |
| CUDA+TensorRT `cargo test --offline --locked --features cuda-driver,tensorrt-api --lib` | 73 PASS; 2 ignored |
| `cargo test --offline --locked --features tensorrt-mock` | PASS, including blueprint/resident-context suites |
| Fast CUDA kernel C++ host emulation/reference | 102 PASS |
| Semantic CUDA kernel C++ host emulation/reference | 65 PASS, including semantic token stored into output column |
| Rust formatting / `git diff --check` | PASS |

**Explicit unverified capabilities:** no authentic NVIDIA GPU available here. NVRTC compilation of the edited kernels on the target GPU, actual TensorRT RTX engine build/enqueue, trained checkpoint end-to-end TTS, acoustic quality, latency, memory usage under load, and statistical/numerical equivalence with PyTorch CUDA are **NOT TESTED**. Host C++ emulation and Rust API compilation do not establish real CUDA execution or numerical parity.

## Status and boundaries

- F14 Naive AR source functionality: **5/5 DONE** (previously 2/5). Covers native TensorRT plan and device session, public CUDA model facade, GPU codebook and semantic sampling, EOS and cancellation.
- F01, F11, F12, F13 already DONE; F08 remains PARTIAL pending built-in codecs; F22 remains PARTIAL pending TensorRT workspace accounting, warmup/unload and scratch pooling.
- Overall equal-weight source functionality: **20 DONE, 2 PARTIAL / 22 = 95.45%** (up from 93.18%). Detailed conditions: **32/36 = 88.89%** (up from 29/36).
- Operability historical **7/11 = 63.64%**, unchanged. Nothing about GPU operability was silently promoted.
- The public API still deliberately refuses unsupported `SelectionMode::Greedy` for native GPU generation; source-level GPU gate covers sampled generation like the preceding DualAR backend. A zero-token Naive request means generation up to available context, but GPU-profile limits remain enforced to avoid uncontrolled builds.

## Next

F22: account for TensorRT engine workspaces and untracked GPU memory, provide explicit warmup/unload, then bounded scratch-buffer reuse. F08 built-in MP3/AAC/MP4/WMA decoding remains outside this cycle without additional codec dependencies.
