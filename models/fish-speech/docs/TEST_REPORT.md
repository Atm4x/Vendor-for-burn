# Rust 1.100.0 prototype test report — 2026-11-17

## Verified (local Linux CPU)

| Check | Status | Evidence |
|---|---|---|
| Rust 1.100 toolchain from user tar | PASS | `rustc 1.100.0 (0ef1bc386 2026-11-09)` |
| `cargo fmt --check` core | PASS | no diff/error |
| `cargo test --offline` core | PASS | 58 passed, 0 failed |
| `cargo test --offline --release` core | PASS | 58 passed, 0 failed |
| `cargo run --offline --release --bin fish-proto` | PASS | all 5 structural NetworkPlans validated |
| Slow AR active-prefix and reset | PASS_CPU_CONTRACT | contiguous position, full capacity, NaN-poisoned tail |
| Fast KV reset and sequence length | PASS_CPU_CONTRACT | 8 codebook positions in synthetic fixture |
| Sampling operation order and distribution | PASS_CPU_CONTRACT | penalty->top-p->temperature->softmax; exp-race RNG distribution |
| RoPE / RMSNorm / SwiGLU / GQA | PASS_CPU_FIXTURE | tiny F32 arrays, local math only |
| Firefly shapes and message packing | PASS_CPU_CONTRACT | synthetic data and test tokenizer |
| Actual TRT bindings compilation | BLOCKED_ON_CRATES | missing `autocxx` registry package in offline Cargo cache |
| Real TensorRT Network Definition build | NOT_TESTED_GPU | no SDK/GPU |
| Model weights loading / true tokenizer | BLOCKED_ON_ARTIFACT | no weights/config/tokenizer |
| Speech output / TorchInductor parity | NOT_TESTED_GPU | no model/GPU |

## Structural CLI output

```
SlowPrefill: inputs=3, outputs=2, ops=21, kv_len=Some(32)
SlowDecode: inputs=3, outputs=2, ops=21, kv_len=Some(33)
FastDecode: inputs=3, outputs=2, ops=20, kv_len=Some(1)
FireflyEncode: inputs=1, outputs=1, ops=26, kv_len=None
FireflyDecode: inputs=1, outputs=1, ops=14, kv_len=None
STRUCTURAL_MOCK_PASS; GPU_INFERENCE_NOT_TESTED; MODEL_WEIGHTS_UNAVAILABLE
```

## Exact trtx adapter blocker

```
error: no matching package named `autocxx` found
location searched: crates.io index
required by package `trtx v0.9.2+rtx1.7`
```

The adapter is an *unverified source probe*, not a PASS. The core structural mock does not depend on trtx and must not be presented as a TRT mock.

## Limitations and next gates

G0 source audit and G1 dependency design done. G2 **partially implemented/tested**: CPU math, sampling, tokenizer special IDs, conversation packing, state machine, KV, audio length contracts. Real tokenizer regex/BPE and exact config handling are pending. G3 weights, G4 golden parity, G5-G8 Windows GPU gates are not complete.

## Continuation 2026-11-17: 78 tests

Test results supersede the prior **58**-test snapshot above; the earlier section is retained as historic evidence, not the current count.

| Check | Result |
|---|---|
| Rust 1.100.0 Debug `cargo test --offline` | **78 PASS / 0 FAIL** |
| Rust 1.100.0 Release `cargo test --offline --release` | **78 PASS / 0 FAIL** |
| `cargo fmt --check` | PASS |
| `fish-proto` CLI release | PASS, 5 structural plans + 3 wired transformer DAGs |
| Transformer DAG validation (typed edges, checkpoint weights, layouts/dtypes, active-prefix, topological order) | PASS_CPU_CONTRACT |
| Synthetic numeric Slow prefill, Slow cached decode, Fast codebook decode | PASS_CPU_FIXTURE |
| Synthetic full-prefill vs prefill+cached-decode logits (`abs <= 1e-5`) | PASS_CPU_FIXTURE |
| NaN-poisoned unused KV tail | PASS_CPU_FIXTURE |
| Explicit Fish embedding gating/codebook offsets, RoPE and causal GQA values | PASS_CPU_FIXTURE |
| Actual `trtx` mock compilation | BLOCKED_ON_CRATES: offline `autocxx` missing |
| Windows TensorRT-RTX, original model, audio synthesis | NOT_TESTED_GPU / BLOCKED_ON_ARTIFACT |

Synthetic CPU weights are deterministic fixture arrays. BF16-tagged cache values are represented as F32 for CPU testing; their numerical precision is not matched to PyTorch/CUDA. Existing structural `NetworkPlan` for Firefly does **not** yet have executable edges. The new fully wired transformer IR does not yet lower to a TensorRT network.

## CONT3 validation (2026-11-17)

- Rust 1.100.0 Linux x64: `cargo test --offline` **97/97 PASS**, `cargo test --offline --release` **97/97 PASS**, `cargo fmt --check` PASS.
- New CPU tests: Fast position-0 Slow context, Fast position>=1 acoustic embedding lookup, mismatched `fast_dim` projection with bias, rejected invalid token embeddings, TRT candidate decomposition numerical equality with typed CPU interpreter, exact KV logical-length bucket boundaries, deterministic first codebook offset and cache prime protocol.
- Candidate TRT-blueprint has 26 deferred operations in Slow prefill/decode and 25 in Fast step0 for synthetic 2-layer model. These are **not** proven TensorRT-RTX compatible. `trtx-adapter cargo check --offline` still stops at missing `autocxx` crate.
- No model weights, real TensorRT SDK or GPU used. No inference, accuracy, speed or VRAM claim. See `CONT3_REPORT.md`.

## CONT4 verification

- Rust 1.100.0 `cargo test --offline`: **132 PASS**, 0 FAIL (114 core unit + 12 original Firefly/PyTorch numeric + 4 original spectrogram/PyTorch numeric + 2 FSQ synthetic equation tests).
- Rust 1.100.0 `cargo test --offline --release`: **132 PASS**, 0 FAIL.
- Rust 1.100.0 `cargo fmt --check`: PASS.
- `cargo run --offline --release`: PASS. This CLI prints structural/IR/blueprint audit, not model inference.
- Tested original Python source for Conv1d, ConvTranspose1d, ConvNeXt, HiFiGAN ResBlock and a two-stage toy HiFiGAN; original LogMelSpectrogram against four waveforms. FSQ values compare to independently reconstructed tagged-source equations, NOT the actual installed third-party implementation.
- Actual TensorRT-RTX mock FFI, full Fish Speech model, checkpoint weights and Windows GPU: UNTESTED.
