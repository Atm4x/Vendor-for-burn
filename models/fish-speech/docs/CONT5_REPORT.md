# Fish Speech Rust prototype CONT5 — Git Bridge integration

Date: 2026-11-17. Exact source branch: `Atm4x/Vendor-for-burn`, `codex/fish-speech`. No real RTX SDK/GPU/weights.

## Restored repository

- GitHub Actions Git Bridge exported the exact branch with all reachable history.
- Bundle archive validated with SHA256 and `git bundle verify`; checkout passed `git fsck --connectivity-only`.
- Moved the branch's offline dependency crate from `fish-speech/` to `models/fish-speech/`.
- Added dependency-free prototype `models/fish-speech/core`, CPU fixtures/tests and reports. Did not alter legacy/obsolete code or root Burn modules.

## Real findings about trtx mock

- Offline vendored `trtx=0.9.1+rtx1.7` and `trtx-sys` compile with Rust 1.100.0 and Clang 17 (`LIBCLANG_PATH=/usr/local/swift/usr/lib`). The vendor crate's original `tensorrt-mock` smoke test passes.
- `trtx::Builder::create_network(0)` in the upstream `mock` feature yields a null underlying `INetworkDefinition`; calls to `add_input` or `nb_inputs` panic rather than building a fake network. This was verified via negative experimental tests and removed from the actual suite.
- `preflight_blueprint` is a safe structural pass with explicit rejected operations; the native `lower_blueprint` API implementation is compiled only with `tensorrt-api` and cannot be executed without the real SDK.
- The new primitive converter supports Input, unbound Weight-as-Input, Identity, Sigmoid, Elementwise Sum/Product, MatMul with transposed RHS; rejects unsupported `LastToken`, RMSNorm, RoPE, GQA, KV, embedding and other deferred AR operations. No full RTX graph.

## Honest gates

- Source/KV/CPU numerical fixtures: PASS (Core 132 tests).
- Cargo offline/mock integration: PASS (trtx vendor test, and preflight tests).
- Native `tensorrt-api` Rust/C++ compilation: PASS with no GPU or runtime inference.
- True TRT network build, engine compilation and GPU execution: NOT_TESTED_GPU.
- Original Fish Speech model weights/tokenizer parity and full TTS: BLOCKED_ON_ARTIFACT.

## Next

1. Separate runtime weight bindings from static checkpoint constants; add `safetensors` manifest and layout mapping.
2. Full typed Firefly graph, FSQ encode parity with executed original library, sample-rate converter parity.
3. Implement eligible primitives under real RTX SDK; keep `mock` as pure static preflight and never access its null network.
4. Finish missing transformer lowering (RoPE/RMSNorm/GQA/KV) with precise GPU/cached decode contracts.
