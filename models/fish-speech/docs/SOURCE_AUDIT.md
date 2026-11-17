# Fish Speech 1.5 source audit (17 November 2026)

## Exact inputs

- `Fish-speech-pipeline-main(1).zip` SHA256 `11c5a556cf9e4998330bda675258dfd97d8f16032e106630d7d7136b5f061604`, 56 ZIP entries.
- `trtx-rs-main.zip` SHA256 `853606e9afc40f7850973c98fdfc192437b1c9150b6ccca524a42cf4a2984869`, 171 ZIP entries, `trtx` version `0.9.2+rtx1.7` (Cargo workspace metadata).
- Specification `FISH_SPEECH_1_5_RUST_TENSORRT_RTX_IMPLEMENTATION_SPEC_2026-11-17(1).md`.

## Active call graph

`fish_speech_lib/inference.py::FishSpeech` -> `TTSInferenceEngine.inference` -> `ReferenceLoader.load_by_hash/load_by_id` -> `VQManager.encode_reference` -> `FireflyArchitecture.encode`, and `send_Llama_request` -> worker `generate_long` -> `encode_tokens` -> `Conversation.encode_for_inference` -> `generate` -> `decode_one_token_ar` (or naive) -> `BaseTransformer.forward_generate` + Fast AR loops -> `GenerateResponse` -> `VQManager.decode_vq_tokens` -> `FireflyArchitecture.decode` -> final WAV/PCM composition.

`fish_speech_lib/model.py` is a potentially old parallel model manager. No changes were made to it.

## Source-derived contracts

- Source is Fish Speech 1.5/Atm4x fork, NOT current Fish Speech S2/S2-Pro.
- Actual checkpoints select `model_type=dual_ar` or `naive`; no checkpoint config was supplied.
- `llama.py` generates combined QKV projection with `[dim, n_local_heads*head_dim, n_local_heads*head_dim]`, scales GQA heads by `n_head / n_local_heads`, applies per-head interleaved RoPE pairs, pre-attention/pre-FFN RMSNorm and SwiGLU `w2(silu(w1(x))*w3(x))`.
- Slow token input contains `1 + num_codebooks` channels; codebook embeddings are summed only when the semantic token is in the configured semantic-token set.
- `decode_one_token_ar` samples semantic ID; calls Fast AR at position 0; derives the **first acoustic codebook from semantic ID minus semantic_begin_id, clamped below at zero**. Only remaining acoustic codes are sampled from Fast AR. Fast KV is reset every Slow step.
- CPU sampling is repetition penalty -> descending sort -> probability cumulative `> top_p` mask with top item always retained -> divide by temperature -> final softmax. Sampling is `argmax(probs / Exp(1))`, not `torch.multinomial`.
- `decode_n_tokens()` initializes a zero-filled previous-window buffer and does NOT record the prefill-generated first token into this penalty buffer; subsequent decoded tokens appear in its 16-token window.
- Full-length KV buffer is returned by Python `KVCache.update()`; Rust prototype explicitly models logical `kv_len` independent from physical capacity and only admits contiguous writes.
- Firefly configured as 44,100 Hz, n_fft=2048, hop=512, n_mels=160. Encoder ConvNeXt depths `[3,3,9,3]`, dims `[128,256,384,512]`. FSQ groups=8, quantizers=1, levels `[8,5,5,5]`, downsample `[2,2]`. HiFiGAN upsample `[8,8,2,2,2]`, kernels `[16,16,4,4,4]`.
- `vector_quantize_pytorch.GroupedResidualFSQ` is an external dependency; numeric FSQ parity remains BLOCKED until versioned implementation/golden fixtures are provided.
- The `FishTokenizer` BPE regex is custom. No `tokenizer.tiktoken` was provided; only special-ID arithmetic and conversation packing are implemented. No real Fish BPE parity claimed.
- Some source paths call `torch.compile(fullgraph=True,backend=inductor,mode=reduce-overhead)` only for token decode; prefill is not necessarily compiled. Reference and Firefly are separate performance segments.

## Source issues requiring explicit decisions later

1. `generate()` does not check EOS for the first sampled token produced by prefill, whereas subsequent decode loop checks EOS. Prototype stops on first-step EOS intentionally; compare outputs against reference.
2. `FishSpeech._initialize_engine()` creates an `engine.inference(...)` generator without consuming it, so it is not a complete warmup.
3. `FishSpeech` public reference-audio type annotation accepts bytes, while path-based loading can call `open()` on the value. Need verify public API semantics in golden tests.
4. Streaming path says streaming is disabled at user API while lower layer still emits segment/header events; it concatenates a final result. Do not claim streaming parity yet.
5. Original `decode_n_tokens` fills first 16 prior-token window with zeros, which applies penalty to token 0 even before any history. Prototype mirrors this structural convention.

## Mapping of required ops to vendored trtx API

| Operation | Rust wrapper inspection | Real GPU validated? |
|---|---|---|
| MatMul / QKV | `NetworkDefinition::add_matrix_multiply` present | UNTESTED |
| Gather / embeddings | `add_gather`, `add_gather_v2` present | UNTESTED |
| Scatter / KV write | `add_scatter` present | UNTESTED |
| GQA attention | `add_attention_v2` behind `v_1_5` (enabled by v1.7), input-mask/active-prefix semantics TBD | UNTESTED |
| RMSNorm / activations | `add_normalization`, `add_unary`, `add_elementwise` present | UNTESTED |
| Convolution | `add_convolution`, `add_deconvolution` present | UNTESTED |
| Reshape / dynamic shapes | `add_shuffle`, `add_slice`, `add_shape` present | UNTESTED |
| Top-p sampling / GPU RNG | no demonstrated integrated fused sampling path | UNTESTED |
| ReLU smoke | `trtx-adapter` source written; Cargo cache missing | BLOCKED_ON_CRATES |
| Transformer engine / full TTS | no engine built | NOT_TESTED_GPU |

The source-level presence of a wrapper method is not proof that Fish Speech network shapes or specific TRT-RTX runtime combinations are supported.

## Architectural boundaries

`config`/`tokenizer`/`conversation`/`sampling`/`session` => typed `graph` => backend adapter. `kv` models logical state; `audio` models shape arithmetic; `math` is isolated CPU fixture logic. Nothing above imports TensorRT types. `trtx` integration lives in an independent package and is not linked into core.

## Outstanding compatibility risks

- No exact checkpoint tensor names/shapes/dtypes/weight normalization removal.
- No exact BPE regex parity or model `config.json` parsing (requires serde, tokenizer ranks).
- No FSQ `get_output_from_indices` numeric implementation, FFT/mel or resampling implementation.
- Synthetic model shape graph is a **contract sketch**, not a validated executable TRT definition.
- No GPU proof of causal alignment, dynamic/active-prefix KV, attention head layout, CUDA Graph capture or launch overhead.
- No weight conversion, reference audio or synthesized waveform.

## CONT3 correction — Fast generation inputs

`DualARTransformer.forward_generate()` uses `fast_project_in` (identity or Linear+bias) on Slow hidden states. In **normal** `decode_one_token_ar` (not the `_agent` variant), the first Fast token step consumes that context at position 0 and discards its logits. Codebook 0 is derived directly from sampled semantic token `max(semantic_id - semantic_begin_id, 0)`; from position 1 onwards Fast receives `fast_embeddings(previous_acoustic_code)` and samples from the returned logits. The corrected graph and session helpers are implemented in CONT3. This was missing from CONT2's `fast_embedded_input` placeholder. GPU parity untested.
