# tts-with-rvc-rs — Stages 1–2: Torch/Burn numerical parity

Independent Rust model project. This stage implements the real RVC v2 `TextEncoder` input transform (`emb_phone`, `emb_pitch`, scale, LeakyReLU, transpose, padding mask) and the WaveNet fused tanh/sigmoid gate. Stage 2 adds RVC encoder relative multi-head self-attention (Q/K/V, score masking, shared-window relative keys and values, softmax, output projection) with CPU PyTorch fixtures. This is a transparent mathematical implementation using Burn tensor operations rather than a specialized fused GPU kernel; performance work remains. It does **not** implement the complete RVC network, HuBERT, RMVPE, FAISS index retrieval, audio inference, model loading, or a persistent worker yet.

The sources of mathematical behavior are the uploaded `tts-with-rvc` Python `infer_pack/models.py` (`TextEncoder.forward`, lines 62–71) and `infer_pack/commons.py` (`fused_add_tanh_sigmoid_multiply`, lines 108–114), and `infer_pack/attentions.py` (`MultiHeadAttention.attention` and its relative-position transforms).

## Reproducible synthetic parity

The committed fixtures were generated using CPU PyTorch by `tools/reference.py`; these are not derived from the user's voice, weights, or audio. The Python script is for **development-time reference only**, never a runtime dependency of the Rust plugin.

```sh
cargo test -p tts-with-rvc-rs --locked --offline
cargo run -p tts-with-rvc-rs --locked --offline -- verify-text-front models/tts-with-rvc-rs/tests/fixtures/text_front.json flex
cargo run -p tts-with-rvc-rs --locked --offline -- verify-attention models/tts-with-rvc-rs/tests/fixtures/relative_attention.json flex
```

## Checkpoint-derived parity (local only)

A real `.pth` can be used to regenerate a private fixture containing the full original layer weights. Do not commit that fixture or the checkpoint:

```sh
python models/tts-with-rvc-rs/tools/reference.py --checkpoint /path/to/CrazyMita.pth --output /tmp/rvc-private-text-front.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-text-front /tmp/rvc-private-text-front.json flex
python models/tts-with-rvc-rs/tools/reference.py --attention --checkpoint /path/to/CrazyMita.pth --output /tmp/rvc-private-attention.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-attention /tmp/rvc-private-attention.json flex
```

The checkpoint is read with `torch.load(..., weights_only=True)`. The tool expects the RVC v2 F0 fields `weight.enc_p.emb_phone.weight`, `.bias`, and `weight.enc_p.emb_pitch.weight`.

Both commands should report maximum absolute difference below `1e-4`; success must be established on an actual Rust execution, not inferred from the PyTorch fixture generation alone.

## Intended lifecycle (not implemented)

`discover -> install external runtime if needed -> launch exe -> load weights -> warmup -> ready -> infer* -> unload/shutdown`. Do not restart between requests. The orchestrator communicates over versioned IPC; the model implementation stays independent of NeuroMita Core.

## Current attention boundaries

Relative attention matches the original RVC shared-head relative-window convention (`heads_share=True`), with zero relative embeddings outside the window, -1e4 masked logits, no dropout in evaluation mode, and `out_channels == channels` as used by the RVC v2 text encoder. Other attention options (`heads_share=False`, cross-attention, proximal or block masks) are not implemented. Runtime layer weights are still supplied by a fixture; native PyTorch `.pth` loading is not implemented. Runtime tensors are FP32.
