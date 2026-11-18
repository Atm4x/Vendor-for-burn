# Silero V5 RU — native Burn/CubeCL port

Native model implementation lives in `models/silero-tts`. Burn 0.22.1 uses
CubeCL; the default CPU backend is `burn/cpu` (CubeCL/LLVM), not Flex. CUDA
and WGPU are optional features. The model data and `v5_ru.pt` are not stored
in the Git repository.

## Status at cycle 1

- **Implemented:** checkpoint-to-SafeTensors converter for all three Silero
  networks (TTS / Accentor / HomoSolver), sidecar model metadata, complete
  vocabulary, exception and homograph dictionaries, PQMF coefficients.
- **Implemented:** Russian letter-mode text filtering, symbol encoder,
  basic/WordPiece tokenizer, speaker/sample-rate validation.
- **Implemented:** CubeCL tensor weight loading, ConvNeXt vocoder (8 blocks),
  2402-channel spectral head, exact-size IRFFT via CubeCL tensor matmuls,
  same-padded overlap-add/envelope normalization, 2/6-band PQMF analysis.
- **Not implemented:** accentor model inference, homograph BERT inference,
  transformer-based acoustic model and predictors, SSML, advanced prosody,
  complete `text -> PCM` orchestration. Do not claim standalone full TTS yet.
- **Performance:** IRFFT currently has O(N²) basis matmul, chosen for exact
  functional behavior and backend portability. GPU and FFT optimization later.

## Convert existing checkpoint (developer only)

```
python tools/export_weights.py /path/to/v5_ru.pt /path/to/exported
```

Requires PyTorch, safetensors and the supplied checkpoint at conversion time.
No Python interpreter or PyTorch dependency remains in native inference code.

## Offline compile

```
cargo check --offline --no-default-features --features cubecl-cpu
```

The current repository vendor bundle has checksum mismatches due to altered
line endings. Additionally, `tracel-llvm-bundler` requires the separately
bundled native LLVM distribution for CPU codegen. This is an external build
constraint, not a backend substitution. The `DOCS_RS=1` environment can skip
that download for `cargo check`; it does not validate execution/linkage.

## Acceptance gates

1. Byte-exact export and metadata verification: converter runs successfully.
2. Rust static compilation on CubeCL CPU: verified with offline `cargo check`.
3. Rust-vs-PyTorch vocoder and PCM parity: **PASS on optional Flex reference backend**,
   with real Silero V5 weights; 7 tests passed. CubeCL execution parity remains
   unverified until native LLVM runtime is available.
4. Full text-to-PCM Silero V5: **pending**.
5. CUDA execution/benchmarking: **separate, later task**.

## Parity reproduction

```
python tools/export_weights.py v5_ru.pt /tmp/silero-assets
python tools/create_parity_fixture.py v5_ru.pt /tmp/silero-assets
SILERO_PARITY_DIR=/tmp/silero-assets cargo test --offline --no-default-features --features reference-flex --lib -- --nocapture
```

With the provided checkpoint (five spectral frames, 192 mel channels):

- Spectral head max absolute errors: magnitude `4.8637e-5`, cosine `3.4176e-5`, sine `4.7296e-5`.
- Complete vocoder, iSTFT and PQMF max absolute PCM errors: 48 kHz `8.17e-7`,
  24 kHz `7.12e-7`, 8 kHz `5.63e-7`.
- These parity results are from diagnostic Burn Flex; the default execution backend
  remains CubeCL CPU. Results do not certify CPU/CUDA CubeCL numerical parity.

## Cycle 2 — complete acoustic graph and PCM parity

The complete acoustic FastPitch graph is implemented in native Burn/CubeCL
Tensor API, including multi-head attention, four-layer duration and pitch
predictors, four-layer Tacotron encoder, pitch/energy projections, length
regulation, three-stage shortened decoder, and 192-channel mel projection.
`SileroTts::load` creates a reusable session, and `infer_token_ids` executes
all neural stages up to PCM for 8/24/48 kHz.

An important TorchScript-specific behavior is that `shorten_layers.0` and
`shorten_layers.1` are each fed the same downsampled tensor; only the second
output contributes to the upsampled path. This differs from sequential blocks.
The native code reproduces the effective TorchScript graph, not an assumed
sequential FastPitch design.

The diagnostic Flex parity tests use a real `v5_ru.pt` checkpoint and verify:

- duration logits: maximum absolute error `4.77e-7`;
- pitch: `4.32e-7`;
- encoder: `5.45e-6`;
- complete token-to-mel: `6.91e-6`;
- full token-to-8kHz PCM (6800 samples): `1.69e-5`.

The source of truth for native default inference is the checkpoint's TorchScript
model. This cycle does **not** implement Accentor and HomoSolver, SSML/prosody
modifiers, or standalone process IPC. `infer_accented` expects already stressed
text. The CPU test environment does not have the native LLVM execution runtime;
`cargo check` on CubeCL CPU is not a CubeCL execution parity claim.

Reproduce acoustic fixtures and the parity tests:

```
python tools/create_acoustic_fixture.py /path/to/v5_ru.pt /tmp/silero-assets
SILERO_PARITY_DIR=/tmp/silero-assets cargo test --offline --no-default-features --features reference-flex --lib
```
