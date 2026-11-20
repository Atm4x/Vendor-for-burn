# F5-TTS native — CONT17 report

Date: 2026-11-20. Parent: published `codex/f5-tts-rust-cont16`, GitHub `7a0ba7a34fe30dd3b8fc9c02a85dbd70479faa6d`.

## Priority accounting

1. **Functional coverage first.** This contour closes the automatic per-shape TensorRT engine construction, cache eviction and single-active-model lifecycle gap. No ONNX, Python or CubeCL fallback is added to the TensorRT path.
2. **Independent local numeric parity second.** A new original-F5-TTS/NumPy oracle verifies standard multi-chunk and multiple reference voices, including duration calculations and audio cross-fades. Existing DiT/CFM/Vocos neural parity fixtures are unchanged.
3. **Actual RTX TensorRT executability third.** No real TensorRT-RTX engine is built or executed here. No Windows/CUDA or pretrained model runs are claimed.

## Implemented

- `trtx_text::TrtKernelFactory` and lazily constructed `TrtWarmBank`: `(reference_frames, total_frames)` is a typed exact-shape key; cache capacity is bounded and positive; recently used entries are retained; unsuccessful builds leave the existing warmed shape intact. Automatic reference-speed selection uses an existing matching reference shape or creates a minimal shape on demand. Legacy manual insertion and selection are maintained.
- `trtx_factory::TrtDirectFactory`: F5-specific engine-image creation for variable frame counts from caller-specified `safetensors` paths. It directly builds DiT+CFM step, Vocos and optionally Cross-Lingual SpeedTransformer, without ONNX. `validate_model` verifies all production-weight tensors before the active model is replaced.
- `trtx_factory::TrtCompiledKernel`: one dedicated long-lived thread per cached frame shape; one `with_warm_session` scope per *thread lifetime*, not per synthesis. Scope owns TensorRT runtime, engines, execution contexts and CUDA allocations while a synchronous request channel serializes CFM, Vocos and speed requests. Unload/LRU replacement sends Shutdown and joins the owner thread. This avoids self-referential lifetime transmutation, leaked `Box` objects and hidden shared global contexts.
- `trtx_worker::TrtModelWorker`: one active model, transactional reload, unload, owned reference voice registry, `[voice]` multi-speaker text, hot swap of reference audio/transcript, active voice selection and fallback, regular text and Cross-Lingual requests. Reference voices are not bound permanently to an engine; the lazy cache determines shape per request.
- No edits outside `models/f5_tts`; no modifications to Fish Speech, RVC, legacy, vendor or other backends.

## Numerical evidence

- `tests/tools/generate_trtx_worker_fixture.py` imports **the original** `chunk_text` function from `F5-TTS-main/src/f5_tts/infer/utils_infer.py` AST, applies original duration and NumPy crossfade rules, then writes `tests/fixtures/trtx_worker_python_expected.json` (metadata) and `tests/fixtures/trtx_worker_python_expected.f32le` (little-endian FP32 numeric oracle).
- Three scenarios: ordinary multi-chunk (3 synthesis calls), fixed-duration multi-chunk (3), and multi-voice with multiple references and unknown-voice fallback (6). **12 calls, 76,704 waveform samples.** Rust outputs match the independent fixture with max absolute error <= `6e-8`. Per-request duration frames and voice routing also match.
- An initial oracle error (mistaken early conversion of Python `0.3` speed multiplier to `f32`) caused a one-frame mismatch. It was fixed in the Python fixture generator after checking the original code; the production Rust multiplier was left unchanged.
- Existing full PyTorch DiT -> CFM -> Vocos -> PCM end-to-end parity from CONT11 and 22-layer DiT tests from CONT10 remain unchanged. A fresh `cubecl-cpu` full end-to-end test *did not finish compiling within the current command limit*. It is not counted as newly passed.

## Local checks

- Rust 1.100.0, pinned offline vendor, `cargo fmt --all -- --check`: PASS.
- `cargo test --offline --locked --lib -- --test-threads=1`: **52/52 PASS**.
- `cargo test --offline --locked --features tensorrt-mock --lib -- --test-threads=1`: **77/77 PASS, 2 intentionally ignored** (actual TensorRT engine tests).
- `cargo check --offline --locked --features tensorrt-api`: PASS.
- `cargo check --offline --locked --features cubecl-cpu`: PASS.
- New failure-path coverage: failed model validation retains old model, chosen voice and warmed cache; failed shape compilation does not evict usable engines. Tests also validate hot voice updates, multi-voice routing and deterministic orchestration.

## Boundaries and unfinished work

- **Actual TensorRT engine build/RTX numerical parity is unverified.** Mock confirms Rust API compilation, input contracts and service orchestration only.
- Warm thread instances remain keyed by *exact* frame counts, not dynamic-shape optimization profiles; this can compile many engines for long/unpredictable requests. LRU prevents unbounded retention, but compilation latency is unknown until hardware evaluation.
- The server/IPC transport protocol is not implemented in this library. Methods support model replacement and reference voice hot-swaps directly; an application can expose them through its own IPC boundary.
- Remaining linguistic boundary: rare Chinese pypinyin/tone-sandhi exceptions outside the CONT12 1,809-case corpus.
- Hardware-independent numerical checks can be expanded to further mixed-language multi-voice cases and sample-rate conversions before starting TensorRT hardware work.

## Progress

- Functional coverage **estimated 99%**, previously 96%. This is an engineering estimate, not proof of universal input completeness; particularly IPC integration and rare language exceptions remain out of scope/unproven.
- Numeric evidence expanded by three independent standard/multi-voice oracles.
- Workability **43% (3/7)** unchanged because no new real CUDA/TensorRT or pretrained checkpoint execution occurred.
