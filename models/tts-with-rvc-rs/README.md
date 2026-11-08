# tts-with-rvc-rs — Stages 1–3: Torch/Burn numerical parity

Independent Rust model project. This stage implements the real RVC v2 `TextEncoder` input transform (`emb_phone`, `emb_pitch`, scale, LeakyReLU, transpose, padding mask) and the WaveNet fused tanh/sigmoid gate. Stage 2 adds RVC encoder relative multi-head self-attention (Q/K/V, score masking, shared-window relative keys and values, softmax, output projection) with CPU PyTorch fixtures. Stage 3 adds a complete evaluation-mode Encoder block (masked relative attention, residual LayerNorm, Conv1d-ReLU-Conv1d feedforward and residual LayerNorm). This is a transparent mathematical implementation using Burn tensor operations rather than a specialized fused GPU kernel; performance work remains. It does **not** implement the complete RVC network, HuBERT, RMVPE, FAISS index retrieval, audio inference, model loading, or a persistent worker yet.

The sources of mathematical behavior are the uploaded `tts-with-rvc` Python `infer_pack/models.py` (`TextEncoder.forward`, lines 62–71) and `infer_pack/commons.py` (`fused_add_tanh_sigmoid_multiply`, lines 108–114), and `infer_pack/attentions.py` (`MultiHeadAttention.attention` and its relative-position transforms).

## Reproducible synthetic parity

The committed fixtures were generated using CPU PyTorch by `tools/reference.py`; these are not derived from the user's voice, weights, or audio. The Python script is for **development-time reference only**, never a runtime dependency of the Rust plugin.

```sh
cargo test -p tts-with-rvc-rs --locked --offline
cargo run -p tts-with-rvc-rs --locked --offline -- verify-text-front models/tts-with-rvc-rs/tests/fixtures/text_front.json flex
cargo run -p tts-with-rvc-rs --locked --offline -- verify-attention models/tts-with-rvc-rs/tests/fixtures/relative_attention.json flex
cargo run -p tts-with-rvc-rs --locked --offline -- verify-encoder-layer models/tts-with-rvc-rs/tests/fixtures/encoder_layer.json flex
```

## Checkpoint-derived parity (local only)

A real `.pth` can be used to regenerate a private fixture containing the full original layer weights. Do not commit that fixture or the checkpoint:

```sh
python models/tts-with-rvc-rs/tools/reference.py --checkpoint /path/to/CrazyMita.pth --output /tmp/rvc-private-text-front.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-text-front /tmp/rvc-private-text-front.json flex
python models/tts-with-rvc-rs/tools/reference.py --attention --checkpoint /path/to/CrazyMita.pth --output /tmp/rvc-private-attention.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-attention /tmp/rvc-private-attention.json flex
python models/tts-with-rvc-rs/tools/reference.py --encoder-layer --checkpoint /path/to/CrazyMita.pth --output /tmp/rvc-private-encoder-layer.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-encoder-layer /tmp/rvc-private-encoder-layer.json flex
```

The checkpoint is read with `torch.load(..., weights_only=True)`. The tool expects the RVC v2 F0 fields `weight.enc_p.emb_phone.weight`, `.bias`, and `weight.enc_p.emb_pitch.weight`.

Both commands should report maximum absolute difference below `1e-4`; success must be established on an actual Rust execution, not inferred from the PyTorch fixture generation alone.

## Intended lifecycle (not implemented)

`discover -> install external runtime if needed -> launch exe -> load weights -> warmup -> ready -> infer* -> unload/shutdown`. Do not restart between requests. The orchestrator communicates over versioned IPC; the model implementation stays independent of NeuroMita Core.

## Current attention boundaries

Relative attention matches the original RVC shared-head relative-window convention (`heads_share=True`), with zero relative embeddings outside the window, -1e4 masked logits, no dropout in evaluation mode, and `out_channels == channels` as used by the RVC v2 text encoder. Other attention options (`heads_share=False`, cross-attention, proximal or block masks) are not implemented. Runtime layer weights are still supplied by a fixture; native PyTorch `.pth` loading is not implemented. Runtime tensors are FP32.

Stage 3 verifies one RVC v2 attention/FFN encoder layer, not yet the full six-layer stack or final mean/log-std projection. It uses exact source layer weights supplied via a private fixture and checks only FP32 CPU parity.

## Full RVC v2 TextEncoder (stage 4)

`src/text_encoder.rs` composes the existing projection and six relative-attention / FFN layers into the original inference-only `TextEncoder.forward` path. It returns two channels of statistics (`mean`, `log_scale`) plus sequence mask, with optional `skip_head` slicing before projection. Modules stay resident in the owning Rust process; model file decoding is not yet integrated.

The standalone reference generator `tools/reference_text_encoder.py` imports the original Python `TextEncoder` from a developer-provided source ZIP. It creates a synthetic six-layer fixture or validates the real RVC v2/F0 `.pth` checkpoint locally, without publishing private weights.

The checked-in synthetic fixture tests both normal forward and `skip_head`; the private full checkpoint fixture is **not** committed. Example build-time reference commands:

```text
python models/tts-with-rvc-rs/tools/reference_text_encoder.py --source-zip ORIGINAL_SOURCE.zip --output models/tts-with-rvc-rs/tests/fixtures/text_encoder.json
python models/tts-with-rvc-rs/tools/reference_text_encoder.py --source-zip ORIGINAL_SOURCE.zip --checkpoint MODEL.pth --output PRIVATE_TEXT_ENCODER.json
cargo test -p tts-with-rvc-rs --locked --offline
cargo run -p tts-with-rvc-rs --locked --offline -- verify-text-encoder PRIVATE_TEXT_ENCODER.json flex
```

The model's convolution and attention code runs natively with Burn; checkpoint parsing and persistent model lifecycle remain separate upcoming tasks.

### Stage 5 — direct `.pth` and RVC flow

`checkpoint::RvcCheckpoint` reads native PyTorch ZIP checkpoints through the vendored `pytorch-reader` without starting Python. It selects the `weight` dictionary, decodes F16/BF16/F32/F64 into Burn FP32 and reconstructs weight-normalized Conv1d kernels from `weight_v` and `weight_g`.

`flow::ResidualCouplingBlock` reconstructs all mean-only coupling layers from `flow.flows.{0,2,4,...}`, including WN dilated convolutions (RVC's flow uses dilation rate 1), speaker conditioning, gated tanh/sigmoid, residual/skip convolutions, channel flips and forward/reverse order. Loaded tensors remain resident until the model instance is dropped.

Check a checkpoint's native weight decoder with `tts-with-rvc-rs inspect-pth model.pth enc_p.emb_phone.weight`.

Compare a native checkpoint to an independently generated Torch reference with `tts-with-rvc-rs verify-flow model.pth flow-reference.json flex`. For generating tests, `tools/reference_flow.py` imports the original RVC classes from the supplied source ZIP. No private checkpoint, reference audio or index should be committed.

These milestones do **not** constitute complete speech synthesis: NSF-HiFiGAN, HuBERT, F0 and retrieval remain unported, and GPU parity is not yet validated.
