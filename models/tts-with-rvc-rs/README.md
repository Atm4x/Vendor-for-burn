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

The six-layer `TextEncoder` also supports direct checkpoint loading via `TextEncoder::load(&checkpoint, &device)` and can be checked with `verify-text-encoder-pth model.pth torch-reference.json flex`. The full model, attention and residual-flow weights are read natively from the same checkpoint without a conversion process. HuBERT/F0 input extraction and the NSF waveform decoder are not yet implemented.

### Stage 7 — NSF decoder first upsampling stage

`decoder::DecoderStage` executes the original `GeneratorNSF` stage order: `LeakyReLU(0.1)` → weight-normalized `ConvTranspose1d` → addition of the harmonic source via `noise_convs` → average of three `ResBlock1` branches. The branches each perform three paired dilated-weight-normalized Conv1d residual operations. This stage is loaded directly from `dec.ups`, `dec.noise_convs` and `dec.resblocks` tensors in `.pth` checkpoints. It does **not** yet synthesize harmonic excitation, run all four upsampling stages or produce final audio.

The original Python class is used only to generate parity vectors for a synthetic checkpoint; users do not need Python at inference time. To validate a private RVC v2/F0 checkpoint locally:

```text
python models/tts-with-rvc-rs/tools/reference_decoder_stage.py --source-zip ORIGINAL_SOURCE.zip --checkpoint PRIVATE_MODEL.pth --fixture PRIVATE_STAGE_REFERENCE.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-decoder-stage PRIVATE_MODEL.pth PRIVATE_STAGE_REFERENCE.json flex
```

This reference test is deterministic because it compares a separately provided harmonic-source tensor rather than the nondeterministic `SineGen` noise. A full synthesis test must also validate `SineGen`, all decoder stages and `conv_post`.

### Stage 8 — complete NSF waveform decoder (external harmonic source)

`decoder::NsfWaveformDecoder` now composes all configured upsampling stages with `dec.conv_pre`, optional `dec.cond` (speaker embedding), the residual branches, and `dec.conv_post` followed by tanh. All weights load directly from the original RVC `.pth` checkpoint, without Python.

This implements the waveform-decoder network **given the fully sampled harmonic source**. The `SineGen`/`SourceModuleHnNSF` pitch-to-harmonic path is **not yet included**. For deterministic parity, the Python fixture generator imports the original `GeneratorNSF` and substitutes only the stochastic harmonic source with the same supplied array as the Rust test. The convolutional generator itself is not rewritten on the Python reference side.

```text
python models/tts-with-rvc-rs/tools/reference_nsf_decoder.py --source-zip ORIGINAL_SOURCE.zip --checkpoint PRIVATE_MODEL.pth --fixture PRIVATE_NSF_REFERENCE.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-nsf-decoder PRIVATE_MODEL.pth PRIVATE_NSF_REFERENCE.json flex
```

The small public two-stage fixture uses two residual kernels per stage to keep CI lightweight; the private `CrazyMita.pth` check uses the original four stages and all three kernels. For the synthetic fixture only, the Python reference generator rewrites PyTorch's ZIP envelope through the standard `zipfile` API without changing tensor contents, because the currently vendored `pytorch-reader` rejects that particular PyTorch 2.10 ZIP layout. This is a checkpoint parser compatibility limitation requiring its own follow-up; existing `CrazyMita.pth` loads directly.

### Stage 9 — HuBERT and RMVPE checkpoints (initial mathematical slices)

Two independently testable preprocessing/inference components were added, without declaring the full models complete:

- `hubert::HubertFirstConv`: 16 kHz HuBERT Base's first convolution `(512, 1, 10), stride 5`, per-channel `GroupNorm(512, eps=1e-5)` and exact GELU. Loads directly from the `model` dictionary of a Fairseq checkpoint using `pytorch-reader`. Full seven-layer convolutional feature extraction, Transformer and output alignment remain unimplemented.
- `rmvpe::RmvpeHead`: the last bidirectional-GRU output projection `512 -> 360`, sigmoid probability and local nine-bin weighted-centroid conversion to F0. Loads the two weights from the root `rmvpe.pt` state dictionary. The Mel, U-Net and BiGRU forward paths remain unimplemented.

The source-derived reference generator `tools/reference_feature_heads.py` reads PyTorch checkpoint weights in safe mode and emits isolated numerical fixtures. For `hubert_base.pt`, the Fairseq `Dictionary` pickle global is allowlisted through a passive stub only for development-time reading; no pickle is executed by the final Rust inference code. The real model files and references must not be committed.

Standalone parity commands for these slices:

```text
cargo run -p tts-with-rvc-rs --locked --offline -- verify-hubert-conv hubert_base.pt private-hubert-conv.json flex
cargo run -p tts-with-rvc-rs --locked --offline -- verify-rmvpe-head rmvpe.pt private-rmvpe-head.json flex
```

`verify-rmvpe-head` uses GRU *output features* as input, not raw mel or WAV; `verify-hubert-conv` checks only the first conv/norm/activation, not the twelve-layer Transformer. Both models are incomplete and cannot perform audio feature extraction or pitch inference end-to-end yet.

### Stage 10 — HuBERT convolutional frontend

`hubert::HubertFrontend` extends the first conv to all seven HuBERT Base feature extractor convolutions (kernels `10,3,3,3,3,2,2`, strides `5,2,2,2,2,2,2`), followed by the 512-channel post-feature LayerNorm and `post_extract_proj` to 768 channels. The frontend accepts 16-kHz mono PCM tensors `[batch, 1, samples]` and returns `[batch, frames, 768]` before the 12 Transformer layers. It is **not** equivalent to full HuBERT features until those Transformer layers are implemented. Synthetic CI uses a reduced channel count to keep test data small; private parity uses the full original 512/768-dimensional weights.

Use `verify-hubert-frontend hubert_base.pt PRIVATE_FRONTEND_REFERENCE.json flex` to check the actual file on the CPU backend. The reference script computes the same seven conv stages, GroupNorm, LayerNorm and projection with PyTorch; it never commits the actual model.

### Stage 11 — one HuBERT Transformer block

`hubert_transformer::HubertTransformerLayer` restores one Fairseq HuBERT transformer layer directly from the `model` state dictionary: Q/K/V/out projections, 12-head scaled dot-product attention on Base, optional key padding mask, attention residual + LayerNorm, GELU/FC1/FC2 and final residual + LayerNorm (evaluation mode, dropout disabled). This is a **single encoder block**, not the full twelve-layer HuBERT model or positional convolution. It does not yet produce RVC content features from WAV end-to-end.

The source generator `tools/reference_hubert_transformer.py` creates independent PyTorch numerical references with and without key padding. Real weight fixtures stay local.

```text
python models/tts-with-rvc-rs/tools/reference_hubert_transformer.py --checkpoint hubert_base.pt --fixture PRIVATE_TRANSFORMER_REFERENCE.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-hubert-transformer hubert_base.pt PRIVATE_TRANSFORMER_REFERENCE.json flex
```

### Stage 12 — all twelve HuBERT Transformer blocks

`HubertTransformerStack` composes the verified Fairseq attention+FFN layer class with separate `.pth` weights from each of the twelve original layers. The stack retains weights in memory, uses evaluation-mode operations and supports key padding masks. The synthetic fixture exercises three layers; the private HuBERT Base fixture checks all twelve. This boundary starts **after positional convolution and outer encoder normalization**; it is not a complete HuBERT forward pass until those surrounding operations are integrated.

```text
python models/tts-with-rvc-rs/tools/reference_hubert_stack.py --checkpoint hubert_base.pt --fixture PRIVATE_HUBERT_STACK.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-hubert-stack hubert_base.pt PRIVATE_HUBERT_STACK.json flex
```

## Stage 13 — HuBERT positional convolution + encoder integration

`HubertPosition` rebuilds Fairseq's grouped positional Conv1d using kernel-axis (`dim=2`) weight normalization, removes the extra sample for even kernels, applies GELU and the outer encoder LayerNorm. When a valid-key mask is supplied, padded feature frames are zeroed **before** positional convolution. `HubertEncoder` composes this with the previously verified Transformer stack.

```sh
python models/tts-with-rvc-rs/tools/reference_hubert_position.py --checkpoint hubert_base.pt --fixture /tmp/hubert_position_private.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-hubert-position hubert_base.pt /tmp/hubert_position_private.json flex
cargo run -p tts-with-rvc-rs --locked --offline -- verify-hubert-encoder hubert_base.pt /tmp/hubert_position_private.json flex
```

The synthesis pipeline (HuBERT feature extractor input length handling, RMVPE F0, SineGen, FAISS, persistent worker) is still incomplete. Reference checkpoints, original user WAV and private fixtures must not be committed.

## Stage 14 — HuBERT waveform → content embeddings

`HubertFeatureModel` composes the native 7-layer CNN frontend, kernel-normalized positional convolution, encoder LayerNorm and configurable 9/12 Transformer layers in one inference-only Rust object. Accepts mono PCM **already sampled at 16 kHz**; no PyTorch process is needed to load the weights or execute these tensor layers. Frame validity masks must already be aligned to the CNN output length.

```sh
python models/tts-with-rvc-rs/tools/reference_hubert_model.py --checkpoint hubert_base.pt --fixture /tmp/hubert_model_private.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-hubert-model hubert_base.pt /tmp/hubert_model_private.json flex
```

This validates an unquantized FP32 inference path on Burn CPU Flex. RVC feature upsampling, audio resampling, RMVPE, SineGen, FAISS, and long-lived worker are still separate and incomplete.

## Stage 15 — first two RMVPE U-Net encoder levels

`rmvpe_unet::RmvpeEncoderStage` implements the original RMVPE evaluation-mode input BatchNorm2d, `ConvBlockRes` (Conv2d → BatchNorm2d → ReLU twice plus identity or 1×1 shortcut), and AvgPool2d(2×2). It loads FP16 weights directly from the original root `rmvpe.pt` via `RvcCheckpoint::open_root`, including running statistics. Stage 0 applies the original outer BatchNorm, and subsequent stages use only their own residual blocks. The implementation supports each of the five encoder levels, but this stage checks **only the first two** on CPU. It is not a complete RMVPE network or F0 estimator.

Reference fixtures are produced from the exact Python `Encoder`, `ResEncoderBlock`, and `ConvBlockRes` classes in the uploaded original source ZIP. Synthetic test weights are committed; private RMVPE weights and corresponding numerical references are not. The test script is development-only, not an inference dependency.

```sh
python models/tts-with-rvc-rs/tools/reference_rmvpe_encoder.py --source-zip ORIGINAL_SOURCE.zip --synthetic-checkpoint models/tts-with-rvc-rs/tests/fixtures/synthetic_rmvpe_encoder.pth --fixture models/tts-with-rvc-rs/tests/fixtures/rmvpe_encoder.json --stages 2
cargo test -p tts-with-rvc-rs --locked --offline
cargo run -p tts-with-rvc-rs --locked --offline -- verify-rmvpe-encoder models/tts-with-rvc-rs/tests/fixtures/synthetic_rmvpe_encoder.pth models/tts-with-rvc-rs/tests/fixtures/rmvpe_encoder.json flex
```

## Stage 16 — RMVPE U-Net Intermediate

The `RmvpeIntermediate` type reuses the tested `ConvBlockRes` implementation for the four original groups of 4 residual blocks each, with the first group changing channels `256 → 512` and the other groups keeping 512. It loads the original `unet.intermediate.layers.*` weights natively from `.pt` without Python, and retains all tensors in the Rust object. The production `forward` does not retain intermediate outputs, while `forward_with_levels` is provided for parity diagnostics.

The committed synthetic fixture tests a reduced two-group / two-block model. An independent private test can use the full original four-group weights with the pooled output of the complete RMVPE U-Net Encoder. This still leaves the transposed-convolution decoder, CNN 3-channel projection, BiGRU, mel extraction and audio/F0 integration unfinished.

```sh
python models/tts-with-rvc-rs/tools/reference_rmvpe_intermediate.py --source-zip ORIGINAL_SOURCE.zip --synthetic-checkpoint models/tts-with-rvc-rs/tests/fixtures/synthetic_rmvpe_intermediate.pth --fixture models/tts-with-rvc-rs/tests/fixtures/rmvpe_intermediate.json
cargo run -p tts-with-rvc-rs --locked --offline -- verify-rmvpe-intermediate models/tts-with-rvc-rs/tests/fixtures/synthetic_rmvpe_intermediate.pth models/tts-with-rvc-rs/tests/fixtures/rmvpe_intermediate.json flex
```

### RMVPE U-Net Decoder parity (Stage 17)

`RmvpeDecoder` loads the existing PyTorch U-Net decoder weights directly from a native `.pt` checkpoint and implements `ConvTranspose2d → BatchNorm2d → ReLU`, reverse-order encoder skip concatenation and residual blocks. Supported decoder strides are `[2, 2]` and `[1, 2]`, using the original PyTorch `output_padding`. The checked CPU parity path is independent of the final RMVPE mel and BiGRU stages.

`reference_rmvpe_decoder.py` obtains Torch reference tensors by executing the unmodified source classes extracted from the original RVC archive. Synthetic test weights only are committed.
