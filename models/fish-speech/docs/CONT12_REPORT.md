# Fish Speech 1.5 Rust/TensorRT-RTX — CONT12

Date: 2026-11-18. Baseline: `FISH_SPEECH_CONT11_MODELS_FISH_SPEECH_SOURCE_ONLY_2026-11-18.zip`; verified local source at `/mnt/data/fish_cont11/models/fish-speech`. Implemented locally in a new CONT12 source directory. Rust 1.100.0, vendored Cargo dependencies from previously obtained Git Bridge; no GitHub changes.

## Result: what was actually implemented

1. `core/src/ar_stream_plan.rs`: partition a complete SlowPrefill/SlowDecode/FastDecode graph into `TensorRT network → (K,V) stream copies → ...` without pretending that a mutable `KvUpdate` is a native TensorRT operator. For the synthetic two-layer Dual-AR fixture this creates **3 network segments, 2 barriers, and 4 KV copies**. Each segment has explicit typed external inputs/outputs and passes the independent `TrtBlueprint::validate` gate. The final network output bindings and aliasing to physical KV buffers remain visible in the plan.
2. Each K/V projection is explicitly cast from FP32 to contiguous BF16 **before** the planned asynchronous copy. Added `TrtPrimitive::CastToBf16` with actual `NetworkDefinition::add_cast(..., kBF16)` in `src/trtx_lowering.rs`. The byte geometry comes from the existing checked `Bf16KvLayout`/`KvWrite2d`.
3. `core/src/kv_device_owner.rs`: a backend-neutral owner for stable physical per-layer K/V allocations. Copy submission is append-only at the logical prefix, reuses the same allocations across Slow prefill/decode, resets Fast prefix per new Slow step, and poisons the session on a partial submission failure. A request-boundary reset synchronizes once and resets logical visibility; no CPU synchronization is specified for each token.
4. `core/src/ar_stream_executor.rs`: ties network segments and K/V copy barriers together through one abstract stream backend (`ArStreamDevice`). The fake backend verifies exact order: `Network, K, V, Network, K, V, Network` and no forced synchronization between steps. All intermediate buffers remain owned until `reset_request` synchronizes; failures also retain buffers rather than dropping them while work might be outstanding.
5. `tests/blueprint_mock.rs`: new integration case requiring partitioned Slow/Fast segments to pass `preflight_blueprint` with no deferred ops. Its **separate tensorrt-mock test build did not finish** within the allotted time and this new test is NOT marked PASS. The previously compiled `tensorrt-api` feature does pass.
6. All changes are **new Rust source** plus the updated Rust mock test. Original Python and obsolete scripts were not edited.

**Important qualification:** `ArStreamDevice` is a testable interface and has a fake implementation only. It is not a CUDA driver. `add_cast` passing Rust/FFI compile checks does not prove on-device engine compilation. `cudaMemcpy2DAsync`, real CUDA streams, engine creation, and GPU speech synthesis are **NOT IMPLEMENTED/NOT TESTED**.

## Acceptance evidence (CONT12)

| Check | Result |
|---|---|
| `cargo test --manifest-path core/Cargo.toml --locked --offline` | PASS: **158 unit + 28 integration = 186/186** |
| `cargo test --locked --offline` | PASS: 3/3 outer crate |
| `python -m unittest discover -s tools -p 'test_*.py'` | PASS: 5/5 exporter |
| `cargo fmt --all --check` | PASS |
| `cargo check --locked --offline --features tensorrt-api` | PASS, using Clang 17 |
| `cargo test --features tensorrt-mock` | **NOT COMPLETED**: stopped during expensive `trtx-sys` C++/FFI compilation |
| Real SDK `buildEngine` / Windows RTX inference | NOT TESTED |
| Original checkpoint/tokenizer/audio parity | NOT TESTED (no model weights) |

All checks above were performed locally. A PASS for a segmented preflight or C++/Rust binding compilation must not be interpreted as functioning TensorRT-RTX inference.

## Exact remaining work — prioritize by dependencies

### R1. Native GPU runtime (critical, SDK/RTX required)
- Implement a concrete `ArStreamDevice`/`KvStreamDevice` using verified Windows CUDA runtime APIs. It must own a single CUDA stream, stable CUDA device buffers, output bindings, and `cudaMemcpy2DAsync` K/V transfers; validate source/destination sizes and pointer/pitch constraints.
- Verify the actual TensorRT-RTX 1.7 API for engine construction, optimization profiles, binding addresses, enqueue execution and stream-sharing with CUDA. Reject unsupported operations rather than mock/fallback.
- Demonstrate `network -> K/V async copy -> network` on a single GPU stream, with no host sync per token and no active-prefix tail access. Use a single-layer small fixture first.

### R2. GPU buffer lifetime and memory performance (critical)
- CONT12 retains intermediate output allocations until request-boundary synchronization. This is intentionally conservative and can grow with long text generation. Introduce stream-ordered arena/reuse or event-based reclamation after validating kernel execution and lifetimes.
- Runtime failures must drain/synchronize before destruction of allocations. Ensure no use-after-free on partial submits, cancellation or model unload.
- Avoid recompiling or allocating a new engine per token. Validate bucket/dynamic shape behavior on actual SDK, RoPE position and KV-prefix semantics.

### R3. Weighted Slow/Fast AR execution (critical; real checkpoints/golden needed)
- Bind real SafeTensors weights, run Slow prefill, one-token Slow decode, Fast AR for every acoustic codebook, then repeat with proper Fast cache reset. Validate BF16 KV numerical parity and GQA against original Python per layer.
- Integrate GPU sampling in original order: 16-token repetition penalty → top-p → temperature → softmax → Exp(1) sampling; EOS/early stop and cancellation without copying full logits to CPU each step.
- Check reference and reference-free prompts and authentic FishTokenizer IDs. Existing input fixtures are synthetic.

### R4. Firefly and end-to-end TTS (real checkpoints/GPU needed)
- Build/execute the existing full Firefly encoder/decoder graph on TensorRT-RTX and validate unsupported op behavior, padding, weight layouts and output waveforms at 44,100 Hz.
- Connect reference audio → mono/resample/log-mel → FSQ encode → prompt → Slow/Fast AR → FSQ decode/HiFiGAN → WAV, plus batching, lifecycle and errors. Until then no end-to-end synthesis.

### R5. Windows performance and packaging (GPU/SDK required)
- On Windows RTX: numerical parity tests, end-to-end speech, benchmark warmed `torch.compile(reduce-overhead)` vs native, GPU launch counts, synchronization, VRAM, Nsight, engine cache and sizes. Then package Rust executable/library, runtime DLLs and licenses. No performance or package-size claims until measured.

### R6. Linux-only work still possible
- More fault-injection tests for stream failures and exact physical-buffers aliasing; ensure `reset_fast_prefix` never leaks stale tail.
- Offline tokenizer/checkpoint golden assets if actual files become available; fully explicit engine metadata/cache keys and error taxonomies.
- Assess memory pressure in long synthetic decode with current conservative in-flight retention; design bounded reuse without claiming measured GPU speedups.

## Progress tracking

Qualitative estimate **~47% full usable TTS** (CONT11: 46%; added validated segmentation and resource contracts, *not* CUDA). The figure is a rough work-remaining indicator, not weighted test coverage.

| Area | Approximate state | Bottleneck |
|---|---|---|
| Core/tokenizer/prompt | ~70% | Exact full golden parity |
| Checkpoint/SafeTensors | ~70% | Real 1.5 weights and parity |
| Slow/Fast AR | ~47% | Weighted GPU execution + sampling |
| Firefly | ~75% | Real engine/output |
| TensorRT backend | ~42% | Engine compilation, CUDA driver, bindings |
| Full TTS runtime | ~12% | Real pipeline and lifecycle |
| Windows GPU parity/performance | 0% | SDK/RTX |
| Windows distribution | 0% | binaries/runtime/licenses |

The perceived progress can plateau while the remaining GPU gates are closed. Do not equate 99% Firefly *structural coverage* with 99% functional synthesis.

## Environment / reproduction

- Source archive contains `models/fish-speech/`, source and lockfiles only, no targets or binaries.
- Offline Cargo config in source expects vendored dependencies at `/mnt/data/fish_cont8/repo/vendor` from the previously obtained Git Bridge; adjust the path if the workspace differs.
- Rust binary used: `/mnt/data/fish_cont8/rust/bin/cargo` (Rust 1.100.0); `LIBCLANG_PATH=/usr/local/swift/usr/lib`.
- GitHub branch has **not** been updated. Use the archive rather than treating this work as an existing commit.
