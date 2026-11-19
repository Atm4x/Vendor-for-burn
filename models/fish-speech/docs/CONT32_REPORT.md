# Fish Speech 1.5 — CONT32: GPU semantic sampling and EOS, F13 closure

## Baseline

- Started from local CONT31 tree `3661a503f2392abfe9dcefdd93f94a4a48a955f8`, identical to published branch commit `a9d9ce9a60769b049b645cce419a678eb0ba7e94`. No Git Bridge re-export was needed.
- Source sampling contract: `Fish-speech-pipeline-main (1).zip`, specifically `fish_speech_lib/fish_speech/models/text2semantic/inference.py`. It applies repetition penalties, sorted top-p truncation before temperature, then exponential-race categorical sampling.

## Changes

1. `core/src/ar_generation.rs` and `core/src/cpu_weighted_ar.rs`: typed `SlowArLogits::{Host,DevicePending}` result and `DualArBackend::sample_semantic` extension, preserving CPU API behavior while allowing a backend-owned device result. Enforce agreement between sampled token and EOS flag *before* entering Fast AR; still complete the final acoustic codebook column on EOS as in the original source.
2. `src/cuda_semantic_sampling.cu`: bounded large-vocabulary device semantic sampler. A single CTA performs score adjustment, stable FP32-key top-p threshold search in 32 passes, tie handling, and exponential race, producing a two-int `{token, eos}` result. No quadratic pairwise rank comparison or bitonic sort over the full semantic vocabulary.
3. `src/cuda_semantic_sampling.rs`: CUDA NVRTC/PTX cache, kernel invocation, explicit history bounds and vocabulary validation, 8-byte result parsing including EOS mismatch and failure sentinel.
4. `src/cuda_generation.rs`: capture Slow logits as a GPU buffer (no logits download), launch sampler in the resident stream, retain buffers until retirement and read back only the compact 8-byte control pair. The Fast GPU codebook loop remains unchanged and device resident.
5. `tools/test_cuda_semantic_sampling_host.py`: compile the actual `.cu` algorithm as one-thread C++ and compare against independent Python source-order probability/sampling math. Tests cover 1 to 155,000 token vocabularies, ties, penalties, top-p 0.01 through 1, temperature and EOS flags.
6. Core regression tests cover GPU semantic EOS on the first step, completion of all Fast codebooks and rejection of an inconsistent token/EOS pair before Fast execution.

## Verification and limits

- Rust Core: 230 PASS, 2 ignored; default crate: 53 PASS, 2 ignored.
- CUDA+TensorRT: 67 PASS, 2 ignored; feature check PASS.
- TensorRT mock: PASS; semantic C++ host-emulation/reference: 65/65 PASS.
- Rustfmt and whitespace check PASS.
- No NVIDIA GPU in this local Linux environment: NVRTC kernel compilation/load, actual launches, backend-specific race behavior, latency, trained checkpoint execution and PyTorch CUDA parity remain unverified. Do not report GPU performance improvements before real hardware measurement.
- Source functionality F13 is 6/6 DONE, although CPU control still synchronizes a small 8-byte token/EOS transfer on each generated semantic column. This is far smaller than the previous full `vocab_size * 4` Slow logits transfer; it is not zero-synchronization autoregression.

## Progress

- F items: 19 DONE, 3 PARTIAL / 22, equal-weight historical index **93.18%** (previous 90.91%).
- Fixed gates: 29/36, **80.56%** (previous 77.78%).
- Operability: historical 7/11, **63.64%**, unchanged.

## Next

Implement F14 Naive AR resident TensorRT inference and integrate public CUDA model facade, then F22 resource management. Do not run Git Bridge unless the published source history or bundle is unavailable or mismatched.
