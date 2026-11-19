# Fish Speech 1.5 — CONT30: bounded source-equivalent Slow decode

## Problem corrected

CONT29 prepared a separate static `slow_decode(active_len)` TensorRT graph for each generated position, bounded to 64 new tokens. The graph encoded the RoPE position, GQA prefix length and KV write offset as constants, so engine caching alone could not reuse the shape across different positions. Extending the cap would have grown the number of prepared engines linearly with sequence length and exhausted build time / device and host memory.

## Changes

1. Added `src/dynamic_decode.rs`. Converts each static Slow decode template at the **upper bound** of a `shape_profiles::ArProfileGroup` bucket into an input-parameterized TensorRT blueprint. Before network enqueue, the Rust backend supplies actual RoPE cosine and signed-sine coefficients (from `fish_rope_coefficients`, including original BF16-frequency behavior), plus a floating attention mask with zeros for `[0, active_len)` and `-1e9` thereafter. KV write geometry is separately retargeted to the exact logical position. A nonparameterized graph is rejected if retargeting is attempted.
2. Added `TrtPrimitive::RotaryCoefficients` and `GqaAttentionMasked` with TensorRT-RTX lowering. Mask broadcasting uses `[1, 1, 1, 1, bucket_capacity]`, after scaling and before softmax. Static paths are unchanged. CPU blueprint reference implementations validate exact-prefix equivalence.
3. `PreparedDualArPlans` prepares only the logarithmic set of buckets intersecting the request, and selected positioned plans share the identical network blueprints within the bucket. For 4096 positions and bucket start 32, the core profile policy has 8 buckets rather than one graph per position.
4. `CudaDualArBackend` uploads the per-position RoPE and mask as runtime inputs and retains the original prefill, fast-codebook, logits and KV contracts. No sliding-window truncation or context loss is introduced.
5. `CudaSemanticGenerator` now has a bounded 8-entry LRU for prepared request plans. The model-level serialized `EngineRegistry` already persists across requests; the currently safe TensorRT execution contexts remain request-scoped, sharing engines throughout one generation and being destroyed after stream retirement. `max_profiled_new_tokens` remains a compatibility field but now means request token limit rather than a requirement to prepare one profile per token; its validation allows 1..=32768, additionally bounded by the model context at request preparation.
6. Updated cache shape-profile discriminator to `source-1.5-bucketed-slow-v1` to avoid confusing previous static-profile engines with the new contract.

## Verification

- Core: **227 PASS, 2 ignored** across all suites.
- Base Rust library: **53 PASS, 2 ignored**.
- CUDA-driver + TensorRT API feature library: **63 PASS, 2 ignored**.
- TensorRT mock/blueprint/resident suites: **61 PASS combined**, 2 ignored.
- `cargo fmt --all -- --check`, `git diff --check`: PASS.
- Tests include RoPE CPU-reference equality, GQA masked/full-bucket equality to exact-prefix attention at positions 1/3/8, KV write retargeting and rejection of static graphs, bucket boundary coverage (32→33, 64→65), constant engine identity within a bucket, and bounded plan-cache reuse and eviction.
- A standalone `cargo check` with the same features was attempted but exceeded the command time limit while generating bindings. A full **feature-enabled `cargo test --lib` compiled and executed successfully**, providing compile verification; do not describe the timed-out check as PASS.
- **Not tested:** genuine RTX runtime/engine enqueue, authentic weights, Windows, latency, and PyTorch CUDA numerical parity. Those remain separate operability gates.

## Progress

Functional index: **90.91% = 18 DONE + 4 PARTIAL / 22** (CONT29: 88.64%).
Detailed acceptance: **26/36 = 72.22%** (CONT29: 24/36). F12: **7/7 source-level DONE**, up from 5/7. Operability remains **7/11 = 63.64%**.

Remaining F-items in ascending order: F08 4/5 (standalone codecs), F13 3/6 (GPU sampler/transfer/EOS), F14 2/5 (Naive TensorRT), F22 4/7 (workspace accounting, warmup/unload, buffer pool). F08 was intentionally deferred under the no-new-dependencies constraint; next implementation priority is F13.

## Git workflow

Source work and tests are local. Publish source changes through GitHub Connector with a parent-SHA lease and exact local tree/blob checks. Do **not** launch a new Git Bridge export when the full local history and matching tree are already available; commit metadata can be synchronized by importing the remote commit object directly.
