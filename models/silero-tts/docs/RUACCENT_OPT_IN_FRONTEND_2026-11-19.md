# Optional RUAccent frontend for Silero V5 RU

Native RUAccent is available only when compiling Silero with the Cargo feature `ruaccent`. It does **not** replace the default SileroStress frontend.

## Build

`cargo build --release --no-default-features --features reference-flex,ruaccent --bin silero-tts`

The dependency is `../ruaccent` and ONNX Runtime is loaded dynamically. Supply a compatible ONNX Runtime native library through `ORT_DYLIB_PATH` at runtime. Model weights are not bundled.

The standalone RUAccent pipeline is in `models/ruaccent`; its MIT notices are in that package. The `ruaccent/accentuator` model pack must contain the required dictionaries and four ONNX models, including `nn/nn_omograph/tiny`.

## Worker protocol

Load Silero and RUAccent together explicitly (paths are illustrative):

```json
{"command":"load","request_id":"load-1","model_dir":"/models/silero-v5-ru","backend":"flex","ruaccent_dir":"/models/ruaccent-tiny","ruaccent_omograph_model":"tiny"}
```

Then invoke the RUAccent frontend:

```json
{"command":"infer","request_id":"tts-1","speaker":"aidar","sample_rate":8000,"input":{"kind":"ruaccent_text","text":"На двери висит замок."}}
```

The existing input `{"kind":"text","text":"..."}` retains the unchanged SileroStress pipeline, even when RUAccent is loaded. The existing `accented`, `ssml`, and `symbol_durations` inputs are unchanged.

Both models are persistent for the lifetime of the loaded worker session. Long-text chunking uses the same bounded/adaptive Silero mechanism as the default frontend. Each chunk passes through RUAccent before acoustic inference.

If `ruaccent_text` is selected without loading `ruaccent_dir`, the worker returns an explicit error. Without the feature, the Rust binary does not depend on RUAccent or its ONNX models.

## Verification and remaining gates

- Linux Flex CPU `cargo test --offline --no-default-features --features reference-flex,ruaccent --lib`: **56/56 PASS**.
- RUAccent `tiny` with real ONNX weights: 39/39 functional tests passed and 20 golden outputs matched exactly. Two raw-logit checks remain outside the original `1e-4` tolerance; see `models/ruaccent/docs/REAL_TINY_ONNX_GOLDEN_AUDIT_2026-11-19.md`.
- No change to SileroStress or the default input. This is an explicit alternate frontend.
- A full `ruaccent_text` **PCM comparison with both models loaded** and Windows ORT runtime execution remain integration gates. Unit tests alone do not establish those.
