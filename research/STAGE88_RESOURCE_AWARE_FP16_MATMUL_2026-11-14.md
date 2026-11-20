# Stage 88 — Budget-aware FP16 Tensor Core GEMM (RTX 4060 regression)

Base: Stage 87 PR #2 (`adba20adb0ea51612777aab64c79ec8e300dede9`); experimental branch only. Toolchain: Rust 1.100.0. No local NVIDIA GPU is available.

## Original error and measured baseline

The user-supplied `RVC_STAGE87_FULL_WAV_RESULTS.zip` reports a full 89.939 s WAV (3,597,570 PCM16 samples, 32 chunks, CUDA) with two successful FP32 cases and two failed FP16 cases:

- `baseline_1` warm 12,369.3005 ms; `baseline_2` warm 11,940.3354 ms. Both WAVs have the same SHA-256 and zero PCM difference.
- `mixed_1` and `mixed_2` both fail before producing PCM: `Shared memory budget exceeded: kernel requests 319488 bytes, hardware allows 101376 bytes per cube.` This is 3.1515x the available SMEM.
- No FP16 vs FP32 audio-quality or speed result can be reported from Stage 87: mixed precision never ran.

## Failure localization

`decoder.rs` selects FP16 `Conv1d::forward`, which calls `cubecl_conv::conv1d`. Burn `conv_forward(..., ConvStrategy::ImplicitGemm)` invokes `cubek::convolution::Strategy::Inferred { SimpleSyncCyclic, Cmma }`. The actual selection path is `cubek-matmul-0.3.1/src/multi_level/routines/batch/simple.rs` -> `selector::plane::infer_blueprint_plane`; the similarly named selector in `cubek-convolution` is **not invoked** by this path.

Before Stage 88, the plane selector constrained tile geometry for vectorization and plane occupancy but not per-cube shared-memory capacity. The actual kernel validator (`partitioned_matmul/setup.rs`) includes full operand stages, a writer's output stage and *an additional accumulator reader stage for bias*. Thus a valid tensor-core instruction could still generate an impossible full stage.

## Architectural fix

`vendor/cubek-matmul-0.3.1/src/multi_level/routines/selector/plane.rs` now calls `fit_plane_tiling_to_smem` after selecting the hardware MMA instruction and row/partition configuration, but before building the matmul blueprint.

The planner computes the same stage footprint as the launch validator: `lhs_stage + rhs_stage + writer_output_stage + optional_bias_accumulator_stage`, with stage buffering and per-dtype element sizes. It reads the device's `max_shared_memory_size`. For an oversized stage, it reduces the most resource-expensive axis among rows-per-plane, number of M partitions, number of N tiles, and K tiles until the footprint fits. It never changes the tensor-core instruction or `ConvStrategy`, and it explicitly returns `SharedMemoryTooBig` if even the minimum tile cannot fit.

This is a general change to the GPU matmul tiling planner, not an ad hoc RTX 4060 case or a direct-conv fallback. The original FP16 weight normalization, precision island, NSF network topology, and production FP32 behavior are unchanged.

The source's vendored `.cargo-checksum.json` entry is updated to keep offline Cargo integrity checks valid. No unused legacy scripts are edited.

## Verification as of packaging

- `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cuda --lib --bin tts-with-rvc-rs`, Rust 1.100.0, `DOCS_RS=1`: PASS. This compiles the modified vendored `cubek-matmul` library and RVC code without executing CUDA JIT.
- `research/stage88_smem_selection_test.py` extracts the exact two planner function definitions from the vendored Rust source, compiles them using `rustc --test`, and runs a stand-alone resource model: 5 PASS / 0 FAIL. Covers the RTX 4060 budget, bias overhead, already-valid shapes, impossible shapes, and FP16/FP32 across six GPU budgets (including buffered stages). These are isolated numerical tests, not CUDA tests.
- `cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu,cuda,wgpu --lib --bin tts-with-rvc-rs`, Rust 1.100.0, `DOCS_RS=1`: PASS.
- `git diff --check`: PASS.
- The Windows workflow includes a PowerShell syntax check, isolated selector source tests, unified `cpu,cuda,wgpu` check, release EXE build, doctor and artifact packaging. Workflow results are pending until GitHub Actions completes.

## Important remaining gates

1. **CUDA JIT runtime**: Stage 88 Windows EXE must execute the first mixed-FP16 convolution on RTX 4060 without a shared-memory panic. The planner is modeled after the validator, but actual kernel resource usage is not proven until a real GPU run.
2. **Numerics and quality**: record PCM16 RMSE, max difference, and listen to the mixed-FP16 output. This mode does not promise bitwise parity with FP32.
3. **Speed**: A/B/B/A baseline/mixed/mixed/baseline with same WAV and warmed runs, including baseline drift. A smaller tile may trade throughput against occupancy; *zero performance gain is possible even if the crash is fixed*.
4. **Generalize**: if successful, compare FP16 tensor-core tile execution with original PyTorch CUDA FP16, and inspect selected MMA instructions and kernel timings before broadening the mixed-precision policy.

### Windows command (run only with the Stage 88 EXE)

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-RVC-Full-Stage88.ps1 -Device cuda -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -WarmRuns 2 -CoreFrames 300 -TensorBatchSize 2 -IndexRate 0.75 -Modes all -MinChannels 64
```

Results archive: `RVC_STAGE88_FULL_WAV_RESULTS.zip`. Preserve the existing `embed/` folder; do not use the Stage 87 EXE for this experiment.
