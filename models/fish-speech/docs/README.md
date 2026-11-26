# Fish Speech 1.5 native Rust prototype — G0/G1/G2 partial

Source snapshot: user-provided Fish-speech-pipeline-main(1).zip and trtx-rs-main.zip (17 November 2026).

## What actually runs

`core/` is a pure Rust 1.100.0 dependency-free library and CLI with executable CPU contract tests. No weight loading, BPE implementation, TensorRT engine or model speech synthesis is claimed. A new CPU STFT/log-mel implementation is tested against original Python.

```
cd core
cargo test --offline
cargo run --offline --bin fish-proto
```

`trtx-adapter/` is a separate, intentionally non-executable-until-dependencies-are-available smoke probe using the vendored `trtx 0.9.2+rtx1.7` mock feature. It only creates a ReLU network to test API wiring, **not** a Fish Speech engine.

```
cd trtx-adapter
cargo run --offline
```

The adapter currently cannot resolve `autocxx` from the local Cargo cache. Do not interpret failure to resolve Cargo dependencies as a TensorRT API test failure.

## Components

- `core/src/config.rs`: model-specific transformer GQA/FFN dimensions and validation. Synthetic test config is **not** checkpoint data.
- `core/src/graph.rs`: separate non-executable typed plans for Slow prefill/decode, Fast decode, Firefly mel->FSQ and FSQ->HiFiGAN; shape and output validation.
- `core/src/kv.rs`: physical capacity vs logical prefix, Slow vs per-step Fast cache cursors, poisoned-tail CPU test.
- `core/src/sampling.rs`: Fish penalty/top-p/temperature order, exponential-race categorical distribution, session RNG. Not PyTorch RNG or bit-identical sampling.
- `core/src/session.rs`: request lifecycle, exact Fast-step count, prefill input not included in previous-token window, EOS/reset.
- `core/src/conversation.rs`: message/VQ channel packing via a tokenizer abstraction. Fixture encoder is not real Fish BPE.
- `core/src/tokenizer.rs`: special-token IDs derived from count of mergeable ranks; semantic/acoustic mapping. Not a production tokenizer.
- `core/src/audio.rs`: Fish padding/transposed-convolution and Firefly length arithmetic, CPU lengths and padding; see spectrogram_cpu.rs for numerical STFT and mel.
- `core/src/math.rs`: CPU FP32 RMSNorm, SwiGLU, RoPE, small causal GQA attention fixtures.

Read `SOURCE_AUDIT.md`, `DEPENDENCIES.md`, and `TEST_REPORT.md` before continuing.

## Restrictions

The Python originals are included only as unmodified user-provided sources in the working area, not in this artifact. No NeuroMita integration was attempted. Do not use the structural mock as a runtime backend; fail explicitly without the real GPU backend. Final deployment must not require Python, Torch or LibTorch.

## 17 November continuation: connected transformer SSA IR and numeric CPU reference

The newly added `core/src/graph_ir.rs` is **not** a thin rename of the original `NetworkPlan`. It declares value IDs and edges, typed input/weight origins, explicit per-layer K/V outputs, checkpoint keys, layout/dtype constraints, inference of node shapes, and a lowering audit that distinguishes candidate TensorRT primitives from unvalidated GPU operations.

`core/src/transformer_ir.rs` constructs fully connected *Slow prefill*, *Slow token decode*, and *Fast codebook decode* computational graphs for the synthetic fixture. This graph is not yet a TensorRT `NetworkDefinition`: dynamic bucket profiles, mutable GPU K/V binding, exact checkpoint parsing and lowering still have to be implemented.

`core/src/cpu_ir.rs` is a small CPU mathematical interpreter for those operations, for **synthetic FP32 fixture testing only**. BF16-tagged cache data remains F32 in this fixture, so no GPU or BF16 accuracy parity is claimed. It can verify absolute-position RoPE, Fish embedding gating, causal GQA, cache prefix isolation and equivalence of whole-prefill vs prefill+cached-decode on matching synthetic inputs. The AR interpreter does not execute Firefly; separate numerical Firefly reference kernels are now available.

To rerun checks, use the toolchain in `.toolchain/bin` or your own Rust 1.100.0, then run `cargo test --offline` and `cargo test --offline --release` inside `core/`. `cargo run --offline --release --bin fish-proto` now prints both old structural plans and the new wired transformer IR audit.

For status across all components read `PROTOTYPE_PROGRESS.md`. Before calling this TensorRT-Rust inference, a real backend adapter and real checkpoint/golden parity must pass.

## CONT3 — Fast AR source-correct wiring and preliminary TRT lowering

Read `CONT3_REPORT.md` and `PROTOTYPE_PROGRESS.md`. New modules: `core/src/trtx_blueprint.rs` (strict TRT candidate primitive expansion, not executable TensorRT), `core/src/shape_profiles.rs` (KV active-prefix bucket planning, not SDK profiles). Added CPU tests and source-correct Fast step 0 / later-step input semantics. `97/97` tests PASS on Rust 1.100.0 offline. Real `trtx` mock cannot yet compile without the offline Cargo transitive closure; Windows GPU completely untested.

## CONT4 — numerical Firefly CPU foundation

Added `core/src/firefly_cpu.rs` (Fish Conv1d/ConvTranspose1d, ConvNeXt, residual/parallel blocks, composable small HiFiGAN), `core/src/spectrogram_cpu.rs` (radix-2 FFT, reflect pad, STFT, Slaney logmel), `core/src/fsq_cpu.rs` (grouped residual FSQ index decode and learned output projection) and exact allocated/valid Firefly audio lengths.

Run `cargo test --offline`; fixtures in `core/fixtures` are generated with the unmodified original `firefly.py` / `spectrogram.py` where possible, and with explicitly marked equation-level FSQ reference because the third-party package is unavailable locally. `132/132` tests PASS in Debug/Release with Rust 1.100.0. **These are CPU numerical primitives, not model inference.** See `CONT4_REPORT.md` for verified and unverified areas.
