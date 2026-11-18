# Fish Speech 1.5 → Rust + TensorRT-RTX — CONT11

Date: 2026-11-18. Local source baseline: CONT10 source-only ZIP + already-preserved interrupted CONT11 changes. Reference: supplied Fish Speech 1.5 Python (`llama.py`), **not** S2. Rust 1.100.0 and existing offline Cargo vendor. GitHub untouched; no GPU/RTX SDK, genuine checkpoint or Windows runner used.

## Result

1. **FishEmbedding** survived the interrupted session. `ar_embedding.rs` builds a checked semantic-ID membership LUT; TensorRT lowering uses word/acoustic gathers, per-codebook offsets and a semantic gate. GPU correctness remains unverified.
2. **Grouped-query attention:** `ar_attention.rs` provides right-aligned causal CPU reference and mask. `trtx_ar_ops.rs::lower_gqa_attention` lowers FP32 Q `[B,Hq,T,D]` + BF16 K/V `[B,Hkv,S,D]` to two 5-D batched matmuls with broadcasted KV group dimension, scale, causal bias, softmax over the key axis and transpose/reshape back to `[B,T,Hq*D]`. No explicit K/V `repeat_interleave`. Decode with `T=1` skips the all-zero mask tensor. Code compiles against vendored TensorRT-RTX 1.7 bindings, **but an RTX engine was neither built nor executed**.
3. **Independent PyTorch CPU golden:** `tools/generate_gqa_golden.py`, `core/fixtures/gqa_pytorch.txt`, `core/tests/gqa_original_pytorch.rs`: prefill `[B=2,Hq=4,Hkv=2,T=3,S=3,D=6]`, active-prefix continuation `[2,4,2,2,5,6]`, cached decode `[1,8,2,1,7,8]`; PyTorch SDPA with actual repeated K/V is the numerical reference. BF16 K/V are rounded before the comparison. Maximum absolute difference from Rust CPU implementation below `1e-5` in all three cases.
4. **In-place KV copy geometry:** `core/src/kv_device.rs` describes a BF16 `[B,H,capacity,D]` destination with pitched rows and contiguous `[B,H,T,D]` source. It computes checked strides, row bytes, destination offsets and lengths suitable for a stream-ordered `cudaMemcpy2DAsync` integration. Reference tests ensure only the new token range changes across batch/head rows and reject overflow, gaps and malformed buffers. **No actual CUDA call or GPU ownership is implemented in this cycle.** FP32 source activations must be converted to contiguous BF16 device buffers before submission.
5. **CPU KV numerical correctness:** `cpu_ir.rs::KvUpdate` now rounds FP32 K/V into BF16, matching the original Python BF16 cache write. Previously it silently retained FP32 precision. Regression checks BF16 round-to-even and untouched tail.
6. **Regression reconciliation:** outdated assertions expecting GQA to remain deferred were corrected in Rust Core and mock tests. GQA and FishEmbedding are now native lowering candidates, while physical `KvUpdate` is *intentionally deferred*.

## Gates executed

| Gate | Result |
|---|---|
| `cargo test --manifest-path core/Cargo.toml --locked --offline` | PASS: 149 unit + 28 integration = **177** tests |
| `cargo test --locked --offline` | PASS: 3 outer crate tests |
| `cargo test --locked --offline --features tensorrt-mock` | PASS: 3 outer + 3 blueprint + 1 backend = **7** tests (shared Cargo target) |
| `python -m unittest discover -s tools -p 'test_*.py'` | PASS: 5 tests |
| `cargo fmt --all --check` | PASS |
| `cargo check --locked --offline --features tensorrt-api` | PASS with `LIBCLANG_PATH=/usr/local/swift/usr/lib` and previously compiled Cargo target |
| CLI structural AR | PASS: SlowPrefill/SlowDecode/FastDecode each has **4 deferred physical KV-update operations** in the 2-layer synthetic fixture |
| CLI Firefly structural | PASS: reference 154/156, decoder 263/264 (remaining input placeholders and CPU LogMel) |
| Actual engine build, `cudaMemcpy2DAsync`, CUDA stream, GPU inference, WAV generation, TorchInductor parity | **NOT TESTED / NOT IMPLEMENTED** |

Compilation with the `tensorrt-api` feature does not prove that the SDK can build a valid TensorRT-RTX engine or that 5-D grouped matmul broadcasting and softmax execute on the target GPU. The 7 mock tests validate only non-GPU contracts.

## Critical remaining work

1. **External KV writes:** implement actual CUDA stream-aware, stable-address BF16 device allocation and 2-D async copies. The existing plan only computes geometry. Preserve independent Slow/Fast caches; Fast reset per Slow frame, no CPU readback per token. Validate data dependencies and frame advancement at stream boundaries.
2. **AR engine partitioning:** TensorRT-RTX lacks custom plugin layers. Split per-layer projection/attention graphs around KV writes or prove a supported native stateful KV update. Pass live Q/K/V and other intermediate bindings between network segments without accidental host synchronization. `TrtBlueprint::preflight` must continue to reject unresolved `KvUpdate` until this is real.
3. **Reusable shape/position profiles:** current GQA, RoPE and KV slicing are static-shape plans. Avoid recompiling an engine per token; use verified dynamic shapes or buckets, with logical prefix length always distinct from physical capacity.
4. **Weighted parity:** real model/checkpoint metadata, SafeTensors full mapping, tokenizer IDs, Slow prefill and decode, Fast codebooks, sampling order and EOS; reference/reference-free audio with real Firefly weights. Existing tests use synthetic fixtures.
5. **Windows GPU:** SDK-dependent engine builder + execute for one AR layer, FSQ/Firefly, full TTS; Nsight launch/sync profiling, comparison with warmed `torch.compile(reduce-overhead)`, packaging, runtime DLL/licensing.

## Completion estimate

End-to-end native TTS **~46%**, 54% remaining (qualitative, same scale as CONT10's ~44%, not node coverage). Core/tokenizer ~70%; weights ~70%; Slow/Fast AR ~43%; Firefly ~75%; TensorRT backend ~38%; full TTS ~10%; GPU parity and Windows distribution 0%. Structural support for operations is ahead of operational readiness.
