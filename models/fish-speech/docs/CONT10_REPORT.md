# Fish Speech 1.5 — CONT10: native Slow/Fast AR lowering

Date: 2026-11-17. Base: user-provided `FISH_SPEECH_CONT9_MODELS_FISH_SPEECH_SOURCE_ONLY_2026-11-17.zip`; Fish Speech 1.5 reference Python ZIP; Rust 1.100.0 offline; pinned vendored `trtx 0.9.1+rtx1.7`. Changes were made only in the local CONT10 source tree. The previously downloaded Git Bridge bundle was **not** refetched and the GitHub branch was **not** modified. Original unused Python/legacy scripts remain untouched.

## Implemented in Rust

1. New `core/src/ar_rotary.rs` constructs Fish 1.5 RoPE interleaved pair indices and FP32 coefficient arrays rounded to BF16 *before* application, as in `llama.py::precompute_freqs_cis`. Supports arbitrary static query position/frame length subject to checked dimensions; `rotate_bhtd` is a numerical CPU reference.
2. `core/src/trtx_blueprint.rs`: native blueprint primitives `EmbeddingGather`, `SplitPackedQkv`, `RotaryInterleaved`, `KvActivePrefix` replace previously deferred operations. CPU executor handles them with explicit tensor layouts, index bounds, and logical KV slicing. `FishEmbedding`, `KvUpdate`, and `GqaAttention` **remain deferred** and `require_no_deferred_ops` continues to block complete AR inference.
3. New `src/trtx_ar_ops.rs`: direct TensorRT-RTX Network Definition lowering, not ONNX:
   - `TokenEmbedding`: Gather weight rows `[V,D]` using `[B,T]` integer token ids.
   - Q/K/V: Slice packed `[B,T,(Hq+2*Hkv)*Dh]` followed by reshape and transpose to `[B,H,T,Dh]`.
   - Interleaved RoPE: static BF16-rounded cos/sin coefficients represented as FP32 constants, pair-swapped Gather on last axis, native elementwise multiply/add; **not** the generic non-interleaved half-rotation.
4. `src/trtx_lowering.rs`: `KvActivePrefix` as native BF16 TensorRT slice `[B,H,0:kv_len,D]`, avoiding attention on uninitialized capacity tail. Existing cached-buffer mutation remains a deferred operation instead of an O(capacity) concatenation fallback.
5. `graph_ir.rs::lowering_status`: statuses updated to distinguish native-lowerable operations from still-deferred FishEmbedding/KvUpdate/GQA.
6. New `tools/generate_ar_golden.py`, `core/fixtures/ar_qkv_rope_pytorch.txt`, `core/tests/ar_qkv_rope_original_pytorch.rs`: golden from independent Python/PyTorch equations mirroring supplied Fish 1.5 `llama.py`; tests check 2 batches, three time steps, GQA 4/2 head splitting, BF16 cached RoPE at position 37; tests also cover embedding bounds and poisoned KV-cache tail.

## Verified

- Rust 1.100.0 `cargo fmt --all --check`: **PASS**.
- `cargo test --locked --offline --manifest-path core/Cargo.toml`: **139/139 unit tests + 27/27 integration tests PASS**, doc tests 0 (166 total).
- `cargo test --locked --offline`: **3/3 outer crate tests PASS**.
- `cargo check --locked --offline --features tensorrt-api`: **PASS**, with `LIBCLANG_PATH=/usr/local/swift/usr/lib`, including real `trtx` / `trtx-sys` bindings. This is compile-time, not engine construction.
- `python -m unittest discover -s tools -p 'test_*.py'`: **5/5 exporter tests PASS**.
- CLI `fish-proto` default: **PASS**, accurately identifies deferred AR operations (see below).
- CLI `--firefly-trt-coverage`: **PASS**, coverage unchanged at 154/156 reference and 263/264 waveform decoder.
- `cargo test --features tensorrt-mock`: **NOT VERIFIED IN CONT10**; repeated compilation of C++/autocxx dependencies exceeded the per-command timeout. Do not interpret this as a functional model test failure or claim this feature PASSED in CONT10. CONT9 had recorded 7/7 PASS.

## Remaining AR blockers (two Slow/Fast layer synthetic fixture)

| Stage | Still deferred operations | Count |
|---|---|---:|
| Slow prefill/decode | Fish semantic-gated multi-codebook embedding | 1 |
| Slow prefill/decode | Mutable KV cache update, per-layer K + V | 4 |
| Slow prefill/decode | GQA causal attention | 2 |
| Fast decode | Mutable KV update, per-layer K + V | 4 |
| Fast decode | GQA causal attention | 2 |

Slow: **7** unresolved operators in 72 blueprint steps; Fast: **6** unresolved in 67 blueprint steps. These counts apply ONLY to the synthetic two-layer fixture and do not indicate remaining FLOPs, model size, or inference readiness.

## Critical next implementation tasks

1. **FishEmbedding**: preserve source `torch.isin` semantic mask and per-codebook offset logic. A naive 1024-way `Equal+Reduce` check every token is wasteful; consider validated on-device semantic-token LUT generated at tokenizer/model load, not a hidden host sync. Keep tokenizer ID contract and do not silently replace external `semantic_token_ids`.
2. **KV update**: use preallocated stable external KV addresses, stream-aware device updates, logical-length metadata. Do **not** materialize or copy full physical cache every decode step. Support distinct Slow and Fast cache ownership; Fast resets per Slow step.
3. **GQA/attention**: native or minimal custom CUDA attention with right-aligned causal semantics and active prefix; avoid materialized K/V repetition. Profile prefill separately from single-query decode. Numerically compare CPU/PyTorch exact head ordering, BF16 casting and masks.
4. **Dynamic positions and profiles**: current RoPE constant tables and active-prefix are **static-graph profiles**. They must become reusable bucketed/dynamic execution to avoid rebuilding an engine for each new token. Current implementation is a graph primitive, not an optimized full AR inference loop.
5. Add GPU sampling with original repetition penalty → top-p → temperature → softmax → Exp(1) distribution; avoid CPU sync per token; complete EOS and Fast loop.
6. With actual Windows RTX SDK, GPU and checkpoint: first engine build→execute parity for one transformer layer and Firefly, then end-to-end speech, TorchInductor warmed baseline and deployment.

## Global completion estimate

Full native Fish Speech TTS: **~44%**, remaining **~56%**. This is a qualitative end-to-end estimate, NOT structural operator coverage or verified GPU readiness. Domain estimates: Core/tokenizer ~70%; weights ~70%; Slow/Fast AR ~35%; Firefly ~75%; TensorRT backend ~35%; full TTS ~10%; GPU parity ~0%; Windows package ~0%.

`G5/G6/G7/G8` from the implementation specification still `NOT_TESTED_GPU` / `BLOCKED_ON_ARTIFACT`. No original `model.pth`, Firefly checkpoint, `tokenizer.tiktoken`, RTX SDK, RTX GPU, Windows runtime or TensorRT engine were used. No claim of neural speech synthesis, GPU numerical parity or competitive performance is made.
