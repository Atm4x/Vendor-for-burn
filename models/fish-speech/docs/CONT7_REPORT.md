# Fish Speech 1.5 Rust / TensorRT-RTX — CONT7

Date: 2026-11-17. Stage: Linux, offline static TRT lowering + original-PyTorch CPU geometry parity. **No SDK-backed engine build or GPU inference.**

## Source and provenance

- Local base: Git Bridge artifact 14347327522 of workflow 44126115177, `codex/fish-speech`, commit `e89871a0`. GitHub branch HEAD was `218f920a` when checked during CONT6. The older bridge cannot be presented as current HEAD.
- CONT5/CONT6 source snapshots from user were integrated locally under `models/fish-speech/`; CONT7 changes stay local. No GitHub remote writes or additional repository GET loops.
- User-supplied modified Fish Speech 1.5 Python pipeline is the source for the numerical fixtures. Legacy/obsolete files were not edited.
- The `trtx-0.9.1+rtx1.7`, `trtx-sys`, and generated bindings are resolved from the already vendored offline Cargo tree. Rust 1.100.0, `LIBCLANG_PATH=/usr/local/swift/usr/lib`.

## Implemented

1. `core/src/firefly_trtx_conv.rs`: typed `FireflyConv2dPlan` for FishConv1d, direct `Conv1x1`, `FishTransConv1d`, and the **ConvNeXt channel-last `nn.Linear` projections**, the latter lowered as 1x1 Conv2D without data/weight reordering. TensorRT inputs/outputs use `[B,C,1,T]`; checked original `[out,in/groups,k]`, `[in,out,k]` and `[out,in]` weight shapes. Validates grouped/depthwise counts, asymmetric zero pad, stride/dilation, right-side transpose crop and FP32 input bytes.
2. `src/trtx_lowering.rs`: typed `lower_firefly_conv2d(...)` calls real `trtx` API for `addPaddingNd`, `addConvolutionNd`, `addDeconvolutionNd`, explicit `setStrideNd`/`setDilationNd`/`setNbGroups`/`setPrePadding`/`setPostPadding`, owned checked FP32 constants, source/expected output shape validation. `lower_firefly_conv2d_from_checkpoint(...)` binds directly to validated SafeTensors from `LoadedWeights`, rejecting wrong source weight rank and non-F32 dtype. **Compile-checked FFI, not GPU executed.**
3. `tools/generate_trtx_convolution_fixtures.py` + golden `.txt`: independent original-PyTorch results from the exact user's `firefly.py`, including grouped/dilated, odd strided, short depthwise, stride-2/8 transpose and channel-last linear projection. Six cases pass the Rust CPU reference at max error <= `2e-5`; TRT network outputs themselves remain untested.
4. `core/src/main.rs --firefly-trt-coverage`: grounded static coverage report over the complete declared Firefly DAG. Coverage is architectural (supported conv/linear node families), not a real engine or kernel execution count.
5. No modifications to arbitrary old scripts or the NeuroMita runtime. There is no silent mock inference fallback.

### Firefly static lowering coverage

| Stage | All nodes | Conv/Linear nodes with TRT code | Remaining nodes | Checkpoint requirements |
|---|---:|---:|---:|---:|
| Reference encoder | 156 | 66 | 90 | 218 |
| Waveform decoder | 264 | 105 | 159 | 232 |
| **Total** | **420** | **171** | **249** | **450** |

Remaining node families: LayerNorm (channel-first and channel-last have distinct semantics), GELU/SiLU/Tanh, residual/sum, parallel resblocks, gamma layer-scale, active masks, STFT/log-mel, grouped residual FSQ encode/decode, and input/output orchestration. TRT engine network construction needs orchestration over the DAG; per-layer lowering is a building block only.

## Verified results

| Test | Outcome | Caveat |
|---|---|---|
| `cargo fmt --all --check` | PASS | Rust 1.100 |
| `cargo test --locked --offline --manifest-path core/Cargo.toml` | **143 PASS**, 0 failed | 120 unit + 12 Firefly reference + 4 FSQ encode + 2 FSQ decode + 4 spectrogram + 1 six-case cross-language fixture |
| `cargo test --locked --offline` | 3 PASS | manifest/checkpoint synthetic |
| `cargo test --locked --offline --features tensorrt-mock` | 7 PASS (3 + 3 + 1) | Does not build real TRT network |
| `cargo check --locked --offline --features tensorrt-api` | **PASS** | Header/binding/Rust compilation, not link/infer on Windows |
| Python exporter `unittest discover` | 5 PASS | Offline trusted source fixture |
| PyTorch original `FishConvNet`, `FishTransConvNet`, Linear numerics | 6 cases PASS with error <= 2e-5 | CPU reference, **not TRT GPU parity** |
| Original Firefly `state_dict` key/name/shape correspondence | 450/450 | Existing CONT6 meta shape audit, not numerical parity |

Unproven (`NOT_TESTED_GPU`): Windows x64 build/link/load; SDK 1.7 actually accepting layers, engine build/serialize/load; RTX execution correctness; original weighted Fish full TTS / voice cloning; GPU-side sampling/EOS; TorchInductor performance and packaging size.

## Overall progress estimate (subjective, not a measured GPU completion metric)

About **34% end-to-end Rust-native TTS engineering completeness**. Structural Firefly graph lowering coverage is **171/420 (40.7%)**, a separate, non-runtime percentage. These must never be equated. GPU acceptance gates remain 0% tested.

## Next priority order

1. Close remaining straightforward Firefly native graph nodes (SiLU, GELU_ERF, Tanh, Add, ParallelMean, gamma, masks) using the exact `trtx` 1.7 API and native NCHW broadcasting, with CPU parity. Then both LayerNorm variants, FSQ & log-mel boundary; build a complete topologically sorted network for encoder and decoder.
2. Obtain exact `config.json`, `tokenizer.tiktoken`, semantic and Firefly checkpoints (source license); validate real SafeTensors and architecture parameters, complete full weighted CPU goldens. Do not invent weights or claim sound reproduction.
3. Full Slow/Fast AR TRT lowering with real active-prefix KV, per-step Fast reset, device sampling, EOS, cancellation and safe buffer/stream ownership; connect to Firefly.
4. Windows SDK/GPU gate: minimal GEMM engine execution → real Conv/TransConv → Firefly + AR → complete TTS; serialisation/caching, f32 first then BF16/FP16. Avoid mock as a production backend.
5. Performance gate: compare warmed same-version original Fish Speech 1.5 `torch.compile(mode="reduce-overhead")` to native at same weights, GPU, precision, input. Profile CUDA launches/syncs, KV, Firefly, VRAM, RTT/RTF, cold/warm build and package footprint.

## Reproduce

```bash
cd models/fish-speech
export PATH=/path/to/rust-1.100.0/bin:$PATH
export LIBCLANG_PATH=/usr/local/swift/usr/lib
cargo fmt --all --check
cargo test --locked --offline --manifest-path core/Cargo.toml
cargo test --locked --offline
cargo test --locked --offline --features tensorrt-mock
cargo check --locked --offline --features tensorrt-api
python -m unittest discover -s tools -p 'test_*.py' -v
cargo run --locked --offline --manifest-path core/Cargo.toml --bin fish-proto -- --firefly-trt-coverage
```

The fixture generator needs the original user's Python `firefly.py` and PyTorch only in development. Shipping Rust runtime does not require Python. Input/model artifacts remain unavailable for real TTS.
