# F5-TTS Rust CONT16 → CONT17

Start from published `codex/f5-tts-rust-cont16`. Keep all changes inside `models/f5_tts`. Do not open MaryML as the architectural basis. No ONNX. Reuse the local pinned Cargo vendor; do not run redundant environment discovery or download the repo diff over GitHub web. Use the GitHub Connector only for final publication in a new branch.

Priorities in strict order:

1. **Finish all functional gaps**: a model-specific warm engine factory/cache keyed by F5 mel frame count and reference count, a single-model worker/session for repeated requests, lifecycle/unload/reload and correct reference voice hot-swap. Compose the existing `TrtF5Session`, `TrtWarmBank`, `trtx_text` and all existing graph constructors without rewriting model mathematics or introducing a universal inference framework. The current `with_warm_session` is a *scoped* engine/context owner, not a persistent general-purpose service; the current warm bank routes already-supplied shape-specific engines and does not construct new engines.
2. **Local numeric parity**: extend the Python/PyTorch fixture coverage to multi-chunk and multi-voice inputs, CFM sway+EPSS across more frame counts, and optional boundary checks of spectral head/ISTFT. Never count a mock graph as true TensorRT numeric parity. Maintain deterministic input noise for the oracle.
3. **TRT hardware checks only once (1) and (2) are satisfied**. Do not waste iterations on actual weights or Windows before then.

Current checks: Rust default 52/52 PASS; TensorRT mock 71/71 PASS + 2 ignored; cargo check `tensorrt-api` PASS, `cubecl-cpu` PASS. New session oracle: `tests/fixtures/trtx_session_pytorch_expected.json` from `tests/tools/generate_trtx_session_fixture.py`. Workability remains 43%. Functional coverage estimated 96%, and never increase it for compiler-only or hardware-only work.
