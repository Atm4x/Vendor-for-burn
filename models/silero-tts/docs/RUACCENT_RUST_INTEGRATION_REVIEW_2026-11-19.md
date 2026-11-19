# RUAccent — Rust integration evaluation

Date: 2026-11-19.

## Supplied reference

`ruaccent-main.zip` contains the Python RUAccent sources (`ruaccent/accent_model.py`, `omograph_model.py`, `stress_usage_model.py`, text processing and tokenization), not weights, ONNX model packs or dictionary data. The included `LICENSE` grants MIT-style rights; the copyright notice is `Copyright 2026 Denis Petrov` and must be retained if copying substantial portions. A package metadata classifier is not consistent with the actual license text, so legal review should use the LICENSE and notices, not the classifier alone.

## Existing native Rust implementation

`https://github.com/alksef/ruaccent-rs` provides a complete independent Rust port, version 0.3.0 (observed 2026-11-19), MIT-licensed with Aleksey Efimov and Denis Petrov copyright notices. It uses `ort`, the official `tokenizers` crate, FST/mmap dictionaries and no embedded Python. README claims golden equivalence for `tiny`, dictionary-enabled CPU, covering e/yo restoration, stress and homographs; these are upstream claims, not validations performed by the Silero project. The port expects a separate model pack including `nn/`, `dictionary/` and matching versions/manifest. Its current Cargo manifest pins `ort=2.0.0-rc.13` with `download-binaries` enabled. The parent Vendor-for-burn workspace pins `ort=2.0.0-rc.15` with `load-dynamic` and no bundled binary download.

## Integration decision

**Do not silently replace the existing SileroStress frontend.** That frontend contains its own contextual homograph solver and accentuation, and altering the stress/yo decisions can alter the synthesized PCM. RUAccent should be a separate, opt-in speech-text normalization provider (or an independent model component) whose output is adapted from `+` stress notation to Silero's supported alphabet. Integrating directly into `models/silero-tts` now would add two incompatible ONNX Runtime assumptions and would still lack the necessary weights in the supplied ZIP.

A future implementation should first obtain a licensed, checksummed model pack, reconcile `ort` runtime and ABI versions without automatic network downloads, check Windows/Linux startup and model lifecycle, and run a golden set of Russian homograph/yo examples as well as A/B acoustic quality/PCM tests. Retain both upstream MIT LICENSE notices and model-pack license provenance. A new provider must expose explicit selection and never change established SileroStress defaults without approval.

**This cycle:** investigation only, no RUAccent source imported, no new dependencies or model weights committed. The long-form text chunker works with the existing native SileroStress frontend.
