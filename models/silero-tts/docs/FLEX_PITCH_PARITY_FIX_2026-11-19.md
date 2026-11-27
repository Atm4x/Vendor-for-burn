# Silero V5 RU — long-utterance PCM parity correction

Date: 2026-11-19. Original model: Silero v5_ru.pt (user-supplied offline archive). Rust 1.100.0, Burn 0.22.1, release Flex CPU, four Rayon threads, virtualized AMD EPYC Linux with four-core quota.

## Verified root cause

The original TorchScript model calls update_pitch_coef on the ordinary text path, including default coefficient 1. That function sets predicted pitch values with absolute magnitude below 0.001 to exactly zero before acoustic decoding. Rust TextToMel::infer_with_controls implemented this logic, but the ordinary TextToMel::infer did not.

On six repetitions of the text "Привет, мир!" (77 characters, aidar voice), 38 of 91 predicted pitch values are suppressed. Independently reconstructing the original PyTorch pipeline without that threshold reproduces approximately 0.06977 maximum PCM difference against apply_tts. Restoring the threshold in the manually executed PyTorch pipeline makes its PCM identical to apply_tts (max difference zero). Two independent apply_tts runs are also bitwise identical.

Rust now applies the equivalent native tensor operations abs().lower_elem(0.001) and mask_fill(mask, 0.0) before acoustic inference. This preserves a device-resident path for CubeCL while sharing the scalar threshold used by SSML. A new unit test covers the threshold boundary and checks Flex tensor/scalar agreement.

## Direct end-to-end validation

Same original PyTorch checkpoint, text and voice in both processes; exact PCM sample-count agreement.

| Sample rate | Samples | Max absolute PCM error | RMSE |
|---|---:|---:|---:|
| 8 kHz | 43,800 | 0.000092760 | 0.000003646 |
| 24 kHz | 131,400 | 0.000081286 | 0.000002813 |
| 48 kHz | 262,800 | 0.000105500 | 0.000003644 |

Before correction: 8 kHz max absolute error approximately 0.06989, RMSE approximately 0.00159. After correction: max error 0.0000928, RMSE 0.00000365. All three tested rates pass max error 0.003, without claiming universal input parity.

Release warmed Flex latency on this workload was approximately 190–199 ms, RTF approximately 0.0347–0.0363 for 5.475 seconds of audio. PyTorch CPU latency varied from approximately 188 to 203 ms on the same shared host. Exact GELU gave approximately 332 ms with only a minor PCM-error reduction; the fast GELU remains preferable on this benchmark.

Rust release library tests: 39 passed, 0 failed. Model fixture-gated tests were not exercised without SILERO_PARITY_DIR; full end-to-end Rust versus PyTorch comparison was run independently using the actual model.

## Performance investigation

Timing of eight ConvNeXt blocks per long-utterance inference: first pointwise projections approximately 42.3 ms, second projections approximately 48.5 ms, fast GELU approximately 18.8 ms, layer normalization approximately 3.8 ms and depthwise convolution approximately 2.5 ms. Total vocoder median approximately 152 ms; ISTFT approximately 4.2 ms; PQMF approximately 2.7 ms.

Further optimization should focus on the 512-to-1536 and 1536-to-512 CPU GEMM projections, intermediate memory traffic and activation fusion. Current Flex GEMM already uses Rayon. A new implementation must outperform it in a controlled end-to-end benchmark and maintain PCM parity.

## Reproducibility

Build: cargo build --offline --release --no-default-features --features reference-flex --bin silero-tts

Use tools/benchmark_release_flex.py with --binary, --model-dir, --reference-pt, --sample-rate, --repeats, --threads and --max-pcm-error 0.003. The max-pcm-error flag makes the benchmark fail on a violated strict maximum-error threshold; PyTorch remains optional and is not used by runtime inference.

The local 432 corrected vendored checksums required by the restricted environment are build-only artifacts and are not included in this commit.
