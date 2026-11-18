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
3. Runtime Rust-vs-PyTorch tensor parity: **pending**, not replaced by
   compilation success.
4. Full text-to-PCM Silero V5: **pending**.
5. CUDA execution/benchmarking: **separate, later task**.
