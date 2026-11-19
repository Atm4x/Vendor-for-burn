# Fish Speech 1.5 CubeCL — CONT44 report

Date: 2026-11-19. Development branch: `codex/fish-speech`. Remote baseline: `790813976b978308ef2e2bf266eb69b43bae4174` (CONT43).

## Scope

The last C04 acceptance gate (real checkpoint CubeCL semantic generation combined with CPU Firefly trace comparison) remains OPEN. Neither the source ZIP nor the restored local workspace provides actual semantic + Firefly weight checkpoints. It would be misleading to count existing synthetic-zero AR graphs as real checkpoint parity.

Continued with C05: `src/cubecl_backend/fsq.rs` adds a TensorRT-independent, device-executed `CubeClFsqExecutor::{encode,decode}` with the existing `GroupedResidualFsqEncoder` and `GroupedResidualFsqDecoder` descriptors as the parameter source (no CPU fallback).

- Decode: one CubeCL work item per output coefficient, mixed-radix FSQ `[8,5,5,5]`, grouped residual quantizers, powers of `(level-1)` and `-1` padding semantics; BCT output.
- Encode: 3 ordered CubeCL kernels for grouped project-in + first FSQ bound, per-dimension residual quantization with ties-even rounding, and mixed-radix code packing; BCT output.
- Inputs, weights, quantizer geometry and invalid indices are preflighted before launch. Projected non-finite values/digits are reported through a device error buffer. Extent checks bound a single allocation to 512 MiB. API is deliberately synchronous and correctness-first; weights are uploaded on every call.
- Uses existing numerical reference fixtures. The "Python golden" files contain deterministic Python-source-formula evaluations; the original `vector-quantize-pytorch` runtime was **not** executed in this cycle.

## Verification

| Test | Result |
| --- | --- |
| Decode vs Rust CPU and Python-formula fixture cases | PASS |
| Encode vs Rust CPU and Python-formula fixture cases | PASS |
| 27 combinations of groups (1,2,8), quantizers (1,2,4), frames (1,3,7), batch=2 | PASS |
| Padded -1, code 0/999, malformed indices, NaN input and width mismatch | PASS |
| `cargo test --offline --locked --features cubecl-cpu --lib` | 94 passed, 2 ignored |
| `cargo test --offline --locked --features cubecl-backend --lib` | 72 passed, 2 ignored |
| `cargo test --offline --locked --lib` | 58 passed, 2 ignored |
| `cargo test --offline --locked --manifest-path core/Cargo.toml --lib` | 208 passed, 2 ignored |
| `cargo check --offline --locked --features cubecl-backend` | PASS |
| `cargo fmt --all -- --check` and `git diff --check` | PASS |

## Fixed-denominator progress

- C01: 4/4; C02: 6/6; C03: 4/4; C04: 3/4; C05: **1/4**; C06: 0/4.
- Total **18/26 = 69.2%**.
- CUDA/Windows operability, real checkpoint parity, and throughput are separate unverified requirements.

## Next

1. Implement CubeCL Firefly encoder transforms / ConvNeXt graph and compare against CPU numerics before performance work.
2. Implement native HiFiGAN/ConvTranspose decoder and connect the existing FSQ kernels into a device-resident Firefly execution path.
3. Once actual checkpoint files and reference traces are available, close C04's final end-to-end gate and validate C05's full audio-output gate. Without these artifacts neither parity gate can be claimed.
4. Optimize device lifetime, eliminate repeated weight uploads, profile kernels and validate Windows CUDA in C06.
