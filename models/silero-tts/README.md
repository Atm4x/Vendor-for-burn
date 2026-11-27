# Silero TTS — native Burn worker

The same worker reads `v5_ru.pt`, compatible multilingual V3/V4 HiFi-GAN, and plain single-graph V5 ConvNeXt TorchPackage files directly, with graph detection based on metadata and tensor schema rather than filename. V3/V4 symbol and speaker counts, Unicode-to-script symbol mapping, and warmup are checkpoint-driven. No runtime Python, PyTorch, LibTorch, or mandatory conversion is required. The prepared Russian SafeTensors directory remains supported for parity and existing installations. V3 supports plain text with the checkpoint's speaker table and 8/24/48 kHz; Russian retains its existing frontend and controls. V3 and plain V5 currently expose plain text only; random voice generation is not implemented and is not advertised. Plain V5 does not implement caret controls, neural stress, phoneme mode or multiple graph routes. See [verified model catalogue](verified_models.json) and [cycle 6 audit](docs/CYCLE6_MULTILINGUAL.md). Indic V3 requires ISO romanized input as in the upstream API.

The upstream catalogue snapshot exposes all 66 entries via `--catalogue` and `--model-info LANGUAGE MODEL`; 15 original package checkpoints are verified. Standalone TorchScript ZIPs are recognized but not executable yet. V4 supports plain text and its original optional Accentor. See [cycle 7](docs/CYCLE7_UPSTREAM_LOADER.md).

Current verification (Rust 1.100.0, Linux, Burn Flex): two English phrases and two speakers, at all three sample rates, using direct `.pt` loading. Duration frames match the original exactly; max mel error `7.63e-6`, full PCM max error `2.38e-4`, isolated vocoder max error `2.70e-5`. The PyTorch fixture tool below is a developer-only independent oracle. The worker now also has a shared `native-worker-ipc` v1 adapter (`--ipc-stdio`); see [IPC contract](IPC_STDIO_WORKER.md). A Windows Flex package recipe is available; GPU variants require separate validation. Historical cycle notes below describe previous intermediate states, not the current acceptance status.

```bash
python tools/create_english_parity_fixture.py /path/to/v3_en.pt /tmp/silero-en-oracle
SILERO_V3_EN_CHECKPOINT=/path/to/v3_en.pt \
SILERO_EN_PARITY_DIR=/tmp/silero-en-oracle \
SILERO_V5_RU_CHECKPOINT=/path/to/v5_ru.pt \
cargo test --offline --no-default-features --features reference-flex -- --include-ignored --nocapture
```

## Status at cycle 1

- **Implemented:** checkpoint-to-SafeTensors converter for all three Silero
  networks (TTS / Accentor / HomoSolver), sidecar model metadata, complete
  vocabulary, exception and homograph dictionaries, 126,523 Accentor ngrams,
  PQMF coefficients.
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

The exporter also persists `accentor-ngrams.json` (126,523 ngram IDs) from the
TorchScript embedding module. This is essential for the upcoming native
Accentor classifier and was missing from the initial export format.

## Cycle 3 — native Accentor, HomoSolver/BERT and default text inference

The complete default Russian text inference path is implemented:

`prepare_text_input -> HomoSolver (3-layer BERT) -> AccentorNgram -> symbol IDs -> FastPitch -> ConvNeXt -> IRFFT/PQMF -> PCM`.

- `src/accentor.rs`: Unicode ngram generation, CPU-side embedding-bag mean,
  4-layer Burn classifiers for stress and ё, original threshold/exception and
  punctuation rules. The classifier matmuls use the selected Burn backend.
- `src/homosolver.rs`: native 3-layer, 12-head BERT encoder with exact GELU,
  LayerNorm epsilon, marker/word pooling, binary homograph classifier, UTF-8
  sentence replacements and the original WordPiece token IDs.
- `tools/export_weights.py`: developer-only conversion of HomoSolver's INT8
  word embedding weights to dequantized FP32 in SafeTensors. This makes runtime
  independent from TorchScript/PyTorch quantization handlers.
- `SileroTts::accentuate` and `infer_text`: complete default **non-SSML**
  text-to-PCM path; sessions keep models loaded between calls.
- `SileroTts::infer_text_unambiguous`: optional shorter path that explicitly
  rejects unresolved homographs; `infer_text` does not have that limitation.

Local diagnostic reference checks:

- Accentor: 27 ngram/MLP inputs and 18 complete text outputs; exact text match.
- HomoSolver: 10 marked BERT inputs with embedding, hidden and logit checks;
  5 homograph sentence replacements exactly match PyTorch.
- Full default text-to-PCM: `Привет, мир!` (7,900 samples, max absolute error
  0.001425) and `На двери замок.` (10,100 samples, max absolute error 0.001019)
  versus the original checkpoint's `apply_tts`, both at 8 kHz.

Tests compare real checkpoint weights on the optional **diagnostic Flex** backend.
The default backend remains CubeCL CPU, which passed `cargo check --offline`.
CPU CubeCL execution requires an available LLVM runtime and remains unverified.
CUDA and Windows remain later validation, not a measure of functional coverage.

Functional gaps remain: SSML and prosody/break/word-timestamp parity, optional
stress/ё flags, independent long-lived EXE/IPC service, boundary and long-text
cases, multi-rate real-text parity, and independent CubeCL runtime validation.
The current exact-size IRFFT via Tensor matmul is functionally correct but slow.

Developer reference fixture generation:

```
python tools/export_weights.py v5_ru.pt /tmp/silero-assets
python tools/create_accentor_fixture.py v5_ru.pt /tmp/silero-assets
python tools/create_homosolver_fixture.py v5_ru.pt /tmp/silero-assets
python tools/create_text_fixture.py v5_ru.pt /tmp/silero-assets
SILERO_PARITY_DIR=/tmp/silero-assets cargo test --offline --no-default-features --features reference-flex --lib
```

Cycle 3 also implements `TextOptions` / `accentuate_with_options` /
`infer_text_with_options`, preserving Silero's five independent `put_accent`,
`put_stress_homo`, `put_yo`, `put_yo_homo`, and `stress_single_vowel`
controls and HomoSolver-specific skip lists. Their text outputs match PyTorch for
24 option-and-text combinations, after properly performing the original
TorchScript INT8 embedding dequantization in the reference model.

```
python tools/create_stress_flags_fixture.py v5_ru.pt /tmp/silero-assets
```

## Cycle 4 — source-equivalent SSML and managed duration/pitch

Added a native `xml-rs` SSML frontend and the original TorchScript inference
controls without a Python dependency:

- `src/ssml.rs`: `<speak>` root, nested `<prosody>`, `<break>`, `<p>` and
  `<s>`, inherited rate/pitch, pauses, punctuation and word-based merging;
  unsupported tags are ignored, matching the original non-recursive behavior.
- `SileroTts::infer_ssml` and `infer_ssml_with_options` handle real text input;
  `infer_text_with_symbol_durations` accepts an ordered list of substitutions.
- `tools/export_weights.py` now writes `ssml-controls.json` from the actual
  TorchScript graph: punctuation IDs, comma IDs, period ID, silence value and
  each speaker's pitch correction. Re-export an existing checkpoint once.
- Acoustic inference now reproduces duration-rate division with PyTorch's
  ties-to-even rounding, comma-to-period pause interpolation, explicit pause
  lengths, voice-specific pitch shifts, and mel-frame silencing. The earlier
  no-SSML `infer_text` path remains unchanged.
- Both symbol and expanded acoustic sequences are checked against the
  checkpoint's 5000-position limits instead of allowing a positional-encoding
  out-of-range panic.

Original-checkpoint parity from local reference fixtures:

- 13 SSML cases: parsed segments, cleaned text, complete token IDs, pause
  positions, rate vectors and pitch vectors match the source Python.
- Three full controlled acoustic cases: all duration frames match exactly;
  maximum absolute mel errors are `8.34e-6`, `1.359e-5`, `7.15e-6`.
- Full SSML-to-8 kHz PCM test with a 75 ms pause: 4500 samples and maximum
  absolute error `1.191e-5` through the diagnostic Burn Flex backend.
- For the supplied `v5_ru.pt`, the original Python package explicitly raises
  `AssertionError: This model doesn't support return_ts` on `return_ts=True`.
  Word timestamps are **not** an available original-model feature to port.

Reproduce the source fixtures with:

```
python tools/create_ssml_fixture.py v5_ru.pt /tmp/silero-assets
python tools/create_ssml_audio_fixture.py v5_ru.pt /tmp/silero-assets
SILERO_PARITY_DIR=/tmp/silero-assets cargo test --offline --no-default-features --features reference-flex --lib
```

The default backend remains Burn/CubeCL; diagnostic Flex is optional and used
only to execute numerical parity without the system-provided LLVM CPU runtime.
CubeCL CPU compilation does not imply executed CubeCL parity. CUDA, Windows and
performance remain a separate subsequent phase. The independent EXE/IPC worker
lifecycle is not part of this functional graph-parity cycle.

## Cycle 5 — ordered `symbol_durs`, all speakers and WAV output

The native wrapper now delegates `symbol_durs` expansion to a deterministic
frontend function that preserves **insertion order** and the original raw
string-replacement semantics, including overlapping keys and successive
`<break>` inserts. The package's source implementation is
`PartTTSModelMultiAcc_v3.transform_durs_to_ssml`.

Reference fixtures generated directly from the supplied `v5_ru.pt`:

- Eight `symbol_durs` scenarios (including an empty substitution key): expanded SSML, normalized segments, token
  IDs, pause indices, rate and pitch vectors exactly match the Python package.
- Complete `symbol_durs` inference for 8/24/48 kHz: each rate produces the
  same number of samples as PyTorch (4,500 / 13,500 / 27,000), with PCM
  max-absolute errors `1.191e-5`, `1.128e-5`, `1.463e-5` respectively.
- Managed-duration and pitch-corrected acoustic inference for **all five**
  actual checkpoint voices (`aidar`, `baya`, `kseniya`, `eugene`, `xenia`): exact
  duration frames and maximum mel error `1.180e-5` (diagnostic Flex).
- Mono 16-bit WAV file output compatible with the original `save_wav` numerical
  conversion, including channel count, sampling frequency and 16-bit samples.
  Public methods are `SileroTts::save_wav` and `save_ssml_wav`.

Generate the two new reference fixtures:

```
python tools/create_symbol_duration_fixture.py v5_ru.pt /tmp/silero-assets
python tools/create_multispeaker_fixture.py v5_ru.pt /tmp/silero-assets
```

The native inference remains implemented using Burn tensor primitives with
CubeCL as the primary backend. All numerical execution parity figures above
use optional Flex **only** as a diagnostic reference while the Linux CubeCL CPU
LLVM runtime is unavailable. Actual CubeCL CPU/CUDA execution, profiling, and
Windows packaging remain separate workability tasks, not missing neural-model
features. The original supplied V5 checkpoint does not support `return_ts`.

The importer does not claim support for arbitrary custom voice checkpoints:
this particular `v5_ru.pt` exposes exactly five predefined speaker IDs.

A follow-up source audit simplified consecutive empty `<p>` and `<s>` handling
to match the original Python rule. Four checkpoint-derived SSML edge cases
have explicit regression assertions. This changes no tensor backend behavior.

## Cycle 6 — standalone long-lived model worker

The `silero-tts` executable is a JSONL worker. Start it with `--stdio`, send one
JSON object followed by `\n` per request, and read one JSON reply per request.
The process keeps `SileroTts` weights and backend allocations alive across
multiple inference calls. There is no Python, PyTorch or Flex dependency in
normal CubeCL builds. `reference-flex` is a developer-only diagnostic feature.

Compile the primary CubeCL CPU configuration:

```
cargo build --offline --no-default-features --features cubecl-cpu --bin silero-tts
```

Native CPU *linking/execution* requires Tracel LLVM 23.1.0-4, which is not
included in the current Git bundle. Setting `DOCS_RS=1` skips native LLVM
setup for static `cargo check`; it is **not** a valid way to build a working
CubeCL CPU executable. Do not silently substitute Flex for CubeCL at runtime.

For development, build the independent worker with the optional diagnostic
backend and load a previously exported model:

```
cargo build --offline --no-default-features --features reference-flex --bin silero-tts
```

Example JSONL conversation (one JSON object on each line):

```
{"command":"capabilities","request_id":"1"}
{"command":"load","request_id":"2","backend":"cpu","model_dir":"/models/silero-v5-ru"}
{"command":"status","request_id":"3"}
{"command":"warmup","request_id":"4"}
{"command":"infer","request_id":"5","speaker":"aidar","sample_rate":8000,"input":{"kind":"text","text":"Привет, мир!"},"output":{"kind":"inline_f32le"}}
{"command":"infer","request_id":"6","speaker":"baya","sample_rate":24000,"input":{"kind":"ssml","ssml":"<speak>Привет<break time=\"150ms\"/>мир!</speak>"},"output":{"kind":"wav_file","path":"/output/voice.wav"}}
{"command":"unload","request_id":"7"}
{"command":"shutdown","request_id":"8"}
```

Every reply contains `request_id`, `ok`, and either `data` or `error`.
Inline audio is little-endian float32 PCM encoded with Base64; `wav_file`
produces mono PCM16 WAV. `symbol_durations` input supports an ordered list of
`[symbol, milliseconds]` pairs; `accented` accepts pre-accented text.
`TextOptions` flags may be provided under `input.options`.

The worker serializes inference and uses at most one resident model. `load`
refuses a second model until explicit `unload`; failed requests do not end the
process. `warmup` runs full text inference (including the accent and homograph
frontend); it is not a no-op. `cancel` responds with `supported: false` because
active CubeCL kernels cannot yet be interrupted safely. A supervising Core
should terminate/restart the worker to enforce an inference timeout. The stdin
request frame limit is 1 MiB; oversized frames are rejected without losing
the next request. All logging/errors must go to stderr, never protocol stdout.

## Native CubeCL CPU bootstrap and verification

The repository includes `.github/workflows/silero-cubecl-llvm-runtime.yml`, a
one-shot dependency bridge that downloads the official Tracel LLVM 23.1.0-4
Linux x64 archive on GitHub Actions, verifies it against the release checksum,
and publishes the verified archive as a short-lived workflow artifact. This
is an external **build dependency**, not a Silero model weight or alternate
inference backend.

To use the downloaded artifact locally, place its files in the bundler cache:

```
~/.cache/tracel/tracel-llvm-23.1.0-4-linux-x64.tar.xz
~/.cache/tracel/tracel-llvm-23.1.0-4-linux-x64.checksums.json
```

Then build without `DOCS_RS` and without Flex:

```
TRACEL_LLVM_BUNDLER_SKIP_CHECKSUM_DOWNLOAD=1 \
  cargo build --offline --no-default-features --features cubecl-cpu --bin silero-tts
```

The native CPU executable linked successfully with the verified LLVM package.
The `worker::tests::cubecl_cpu_executes_tensor_matmul` test executes a real
CubeCL CPU kernel and verifies the output `[7, 10, 15, 22]`; it passed on
Linux x64. Tests using optional `reference-flex` are not CubeCL parity evidence.
The real-weight CubeCL CPU inference smoke is a separate gate and must be
recorded independently.
