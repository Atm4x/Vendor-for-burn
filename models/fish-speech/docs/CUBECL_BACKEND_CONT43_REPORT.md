# Fish Speech / CubeCL CONT43 — device-resident inter-graph token chaining

Date: 2026-11-19. Development branch: `codex/fish-speech`. Remote base `6f73ebce8a4bc7974aff36a18ec2d72d80cb1d60` (CONT42). No changes to `codex/fish-speech-tensorrt`.

## Scope and implementation

- Retain validated device-side sampling results as request-bound I32 handles while reading the selected ID/EOS on CPU solely to run the synchronous controller and produce output traces. Only a small selected-ID/EOS pair crosses GPU->CPU; selected tokens are **not re-uploaded** as next-graph input.
- `CubeClRequest::pack_token_column()` copies sampler outputs to a device I32 token column. For DualAR acoustic codebook zero, the device code is derived from the sampled semantic token using the original saturating-subtraction rule. `previous_acoustic_code` in the next Fast AR segment is also forwarded from the sampled code via a typed device buffer.
- Runtime graph binding accepts device I32 only for the explicit `tokens` and `previous_acoustic_code` inputs and only when accompanied by a request-owned, verified host shadow. The existing token range/gated FishEmbedding/TokenEmbedding validators still run against the shadow. A duplicate host/device token binding, foreign request, wrong dtype/shape, or mismatched host ID is rejected. Device outputs lacking verified sampling provenance are not accepted as embedding inputs.
- Fast context forwarding and request-scoped scratch retirement remain unchanged. `reset_request` clears the last sample. On CPU runtime, full graph-to-graph execution uses the device handles for both DualAR and Naive. CPU synchronization for sampling remains a performance issue; multi-lane top-p is deferred.
- Fixed the previously broken Naive sampler sentinel: `u32::MAX` exceeds I32 and caused `CubeCL EOS exceeds i32 range` at runtime; replaced with `i32::MAX as u32`, matching Fast codebook semantics.

## Verification

| Test | Result |
| --- | --- |
| Real CPU CubeCL I32 packing / derived semantic initial code | PASS |
| Real CPU CubeCL two complete frames DualAR, Greedy, device Slow/Fast inputs | PASS |
| Real CPU CubeCL two complete frames Naive, Greedy, device Slow inputs | PASS |
| Stochastic top-p, two frames, both variants, no host logits | PASS |
| Immediate EOS termination, both variants | PASS |
| Cancellation after completed frame, both variants | PASS |
| `cargo test --offline --locked --features cubecl-cpu --lib` | 90 passed, 2 ignored |
| `cargo test --offline --locked --features cubecl-backend --lib` | 72 passed, 2 ignored |
| `cargo test --offline --locked --lib` | 58 passed, 2 ignored |
| `cargo test --offline --locked --manifest-path core/Cargo.toml --lib` | 208 passed, 2 ignored |
| `cargo check --offline --locked --features cubecl-backend` | PASS |
| `cargo fmt --all -- --check` / `git diff --check` | PASS |

## Gate status and limits

C01 4/4, C02 6/6, C03 4/4, C04 **3/4**, C05 0/4, C06 0/4. **Total 17/26 = 65.4%.**

C04's final gate remains **open**: checkpoint-level CubeCL semantic generation with CPU Firefly, compared to a reproducible reference trace. The new end-to-end tests use tiny synthetic Transformer models and zero weights; they do not demonstrate real Fish Speech audio quality or checkpoint parity. The sample kernels still use one GPU lane and host-generated exponential random values; GPU CUDA launch and Windows verification were not performed. Generated token ID/EOS readback remains necessary in the current synchronous controller, though token inputs themselves are device-to-device.

## Next

1. Construct fixed-seed checkpoint trace runner for semantic generation followed by existing CPU Firefly implementation. Supply compatible real model artifacts to validate quality/parity rather than treating synthetic graphs as equivalent.
2. Parallelize GPU top-p/repetition reduction and avoid per-codeword kernel launch overhead while preserving exact sampling semantics and device provenance.
3. Validate Windows CUDA/RTX behavior, BF16, cancellation across actual CUDA streams and performance in C06.
