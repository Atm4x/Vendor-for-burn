# RUAccent native Rust port — integration stage

This directory carries a pinned MIT-licensed Rust implementation of the Python
RUAccent pipeline supplied as `ruaccent-main.zip`.

* Upstream Python: Denis Petrov, [RUAccent](https://github.com/Den4ikAI/ruaccent).
* Rust source: Aleksey Efimov, [ruaccent-rs](https://github.com/alksef/ruaccent-rs), pinned to `4dd4b2f49d2250ee17448484d040180e042f9cee`.
* The two original copyright notices are preserved in `LICENSE`.

## Architecture

`src/text` and `src/tokenization` provide CPU-only deterministic logic without
ONNX. The `onnx` feature additionally enables `dict`, `models` and `pipeline`:
accent placement, stress-use prediction, homograph resolution, and `е/ё`
disambiguation. The full pipeline uses ONNX Runtime **2.0.0-rc.15** with
**load-dynamic**, aligned to the Vendor-for-burn dependency; it does *not*
request prebuilt binaries from Cargo. A lazily initialized process-wide ORT
environment is shared by the four models.

The standalone `ruaccent-worker` JSONL executable is also gated by `onnx`.
The worker loads a single model pack once and reuses sessions. The `+` stress
marker is preserved exactly. It never fetches weights at runtime.

## Model files

The supplied Python ZIP contains no weights. Supply your own legally sourced
`ruaccent/accentuator` model pack including:

* `dictionary/` (`accents.json.gz` or equivalent, `omographs.json.gz`,
  `yo_words.json.gz`, `yo_homographs.json.gz`; optionally FST cache)
* `nn/nn_accent/`, `nn/nn_stress_usage_predictor/`,
  `nn/nn_yo_homograph_resolver/`, `nn/nn_omograph/tiny/`

Model files and ONNX Runtime shared libraries are not redistributed with this
source. Set `ORT_DYLIB_PATH` to the proper ONNX Runtime library when needed.

## Building

Text-only tests:

    cargo test --no-default-features

Full feature, with all dependencies resolved:

    cargo test --features onnx
    cargo build --release --features onnx --bin ruaccent-worker

The parent repository uses a checked-in `vendor/` registry mirror. Its current
mirror lacks dependencies introduced by RUAccent (notably `fst` and
`tokenizers`). Before an **offline** build, merge the pinned Cargo source archive
from the repository's *RUAccent Rust source snapshot* Actions workflow into the
parent `vendor/` folder. Only missing package directories are required; do not
commit checksum normalization performed locally to accommodate the restricted
builder. The source snapshot workflow pins the upstream Rust SHA and includes
SHA-256 checksums for its archive. Alternatively build with an isolated online
Cargo registry configuration that does not inherit the parent vendor override.

## JSONL worker

Input (one JSON object per line):

    {"command":"load","request_id":"1","model_dir":"/path/to/pack","omograph_model":"tiny"}
    {"command":"process","request_id":"2","text":"На двери замок."}
    {"command":"process","request_id":"3","text":"...","skip_regex":"https?://\\S+"}
    {"command":"status","request_id":"4"}
    {"command":"unload","request_id":"5"}
    {"command":"shutdown","request_id":"6"}

The worker also supports `capabilities`, enforces a 1-MiB line length and a
16,384-character text ceiling, and returns explicit JSON errors without exiting.

## Verification boundaries

* `cargo check --features onnx` on this pinned source: PASS (Rust 1.100.0).
* ONNX-feature unit tests and golden text splitter: PASS without model weights.
* JSONL protocol/oversized frame tests: PASS without model weights.
* Python-vs-Rust model output parity: **not yet rerun**, because neither the
  provided Python archive nor this source includes the `tiny` ONNX pack.
* Some inherited golden neural tests return early when
  `RUACCENT_TINY_WORKDIR` is unset; those green tests do **not** prove model parity.

The Silero v5 RU frontend continues using its existing Accentor/HomoSolver by
default. A RUAccent→Silero experimental adapter has been compiled locally, but
is not a default replacement: the backend's stress and homograph choices can
change the output PCM. Enable it only after acquiring actual model weights and
measuring pronunciation and output parity on representative Russian text.
