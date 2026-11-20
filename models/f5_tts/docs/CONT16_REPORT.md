# F5-TTS Rust CONT16 — functional TensorRT-RTX session and text orchestration

Date: 2026-11-20. Parent: published `codex/f5-tts-rust-cont15`, GitHub commit `40f1e6452ae4dc81ba340c4381e31ca5a10f412b`.

## Priorities

1. Functional coverage first: native fixed-shape TRT execution sequence and standard/Cross-Lingual/multi-voice routing.
2. Independent numerical comparison with Python/PyTorch second.
3. Actual GPU runtime/Windows/pretrained checkpoint third; none of those checks is claimed here.

## Implemented

- `src/noise.rs`: one shared deterministic Gaussian stream for both backends. CubeCL's former local sampler is removed; identical seed-to-stream logic is preserved.
- `src/backend/trtx_session.rs`: specialized fixed-shape TensorRT F5 session with explicit owned allocations, dual ping-pong CFM mel buffers, CPU-prepared CFG token/mask inputs, scalar `dt/cfg/restore`, shared timestep embedding, reference mel inpainting on final step, zero intermediate mel downloads, Vocos generated-mel subview and native ISTFT. Optional Cross-Lingual reference SpeedTransformer path. A second inference request reuses the allocated buffers.
- `src/backend/trtx_cuda.rs`: CUDA runtime opened from caller-specified DLL/SO path; explicit external TensorRT-RTX path. H2D copies use owned pinned staging and the same CUDA stream as engine enqueues; pointers are not invalidated before synchronization. D2H is only used for the final spectral head or speed logits; stream synchronized before releases, including after inference errors.
- `with_warm_session`: scoped runtime/engine/context lifetime with a caller callback handling multiple requests, avoiding self-referential Rust structures and leaked engine lifetimes.
- `src/backend/trtx_text.rs`: native standard F5 text + chunking + cross-fade, multi-voice script with `main` fallback and shared per-request RNG stream, Cross-Lingual text with one speed estimate and shared chunking/pinyin logic. A minimal fixed-shape warmed bank can route already-created engine handles by `(reference_frames, target_frames)` and unload them.
- The existing 22-block DiT, CFM step, TRT Vocos and TRT SpeedTransformer graph constructors are reused, not duplicated. No ONNX or CubeCL fallback in the TRT path. No changes to Fish Speech/RVC/Legacy.

## Numerical verification

- New independently generated PyTorch 2.10 CPU fixture (`tests/tools/generate_trtx_session_fixture.py`, `tests/fixtures/trtx_session_pytorch_expected.json`): five 256-dimensional sinusoidal time embeddings and 400 generated mel values from an Euler+CFG+inpainting trace. Session tests compare each case; tolerance **1e-4** for frequency values, **5e-6** for generated mel. PASS.
- End-to-end session structural tests use a deterministic in-memory mock transport, not a TensorRT kernel. They verify the same numerical CFM recurrence against the PyTorch fixture, final reference restoration, generated-mel Vocos view, two repeated syntheses without new allocations, one Cross-Lingual speed classification, no intermediate mel D2H copies, and synchronization/cleanup on failures.
- Original CubeCL full DiT and DiT→CFM→Vocos parity fixtures from CONT11 are retained and unchanged. No new claims about numerical equivalence of TensorRT GPU graph execution.

## Local checks

- Rust 1.100.0, pinned offline Cargo vendor.
- `cargo test --offline --locked --lib -- --test-threads=1`: **52/52 PASS**.
- `cargo test --offline --locked --features tensorrt-mock --lib -- --test-threads=1`: **71/71 PASS, 2 ignored** (real TensorRT graph creation intentionally ignored in mock mode).
- `cargo check --offline --locked --features tensorrt-api`: PASS.
- `cargo check --offline --locked --features cubecl-cpu`: PASS.
- `cargo fmt --all`: PASS.
- CUDA/WGPU hardware, Windows, real TRT engine build/execution, pretrained weights: **NOT RUN**.

## Remaining functional gaps, ranked

1. Engine factory/cache that builds/selects serialized TensorRT engines on demand for arbitrary `(reference_frames, total_frames)` instead of requiring the caller to pre-register matching fixed shapes. The warmed shape bank currently routes *already provided* sessions and does not create them. The scope-bound `with_warm_session` supports repeated requests at one shape, not a persistent multi-engine worker.
2. Single-model service worker lifecycle and engine cache eviction/hot model replacement, plus long-request chunk scheduling across newly built shapes. Existing functions make it possible to supply shape-appropriate kernels, but the production worker is not built here.
3. The TPU-free/TensorRT-only GPU ISTFT path (cuFFT) is an optimization, not a mathematical feature gap: the existing Rust ISTFT already produces the required waveform.
4. More Chinese tone-sandhi edge cases outside the CONT12 Python-parity corpus; all current data-driven language tests remain intact.

## Progress

- **Functional coverage: 96% (estimated)**, formerly 92%. This is an estimate of completed distinct source-function behavior across both backends, not a CPU/GPU or real-weight reliability claim.
- **Workability: 43% (3/7)**, unchanged; the newly added GPU transport was compiled but not run on NVIDIA hardware.
