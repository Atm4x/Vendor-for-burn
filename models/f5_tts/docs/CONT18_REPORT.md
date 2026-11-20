# F5-TTS native — CONT18: upstream speech-edit and midpoint functional parity

**Date:** 2026-11-20. **Base:** GitHub `codex/f5-tts-rust-cont17`, commit `27f5f10fbfc034cd6cad16c7935af091cc1fd39a`.

## Priority order and source of truth

1. **Feature coverage:** inspect user's original `F5-TTS-main (1).zip`, especially `src/f5_tts/infer/speech_edit.py` and `src/f5_tts/model/cfm.py`. Audit uncovered the speech-edit feature missing from all earlier Rust contours. This **invalidates any inference that previous 99% implied complete functionality**.
2. **Independent local numeric parity:** generate stable fixtures with the original speech-edit AST and PyTorch. Do not score compilation or mocks as numerical execution of the TRT network.
3. **Hardware/runtime operability:** no real TensorRT RTX engine or GPU run this contour.

This is a **F5-TTS inference-only** port, not a trainer/Gradio/ASR service port; do not count E2TTS and BigVGAN model variants as F5TTS v1+CROSS-LINGUAL inference features unless scope is explicitly broadened. MaryML is not used. No ONNX.

## Implemented

- `src/speech_edit.rs`: `EditSpan` and `SpeechEditPlan` build the exact frame-domain condition and non-prefix keep mask of the original `speech_edit.py`. Each edit interval may have an explicitly chosen replacement duration; multiple disjoint edits are supported. Uses Python-compatible `round_ties_even()` on double-precision frame positions. Enforces strictly ordered nonoverlapping in-range regions, positive replacement frames, finite input, and F5's 65,536-frame inference cap. Adds a generated frame at the end to reproduce the `CFM.sample` duration floor `max(text_length, cond_frames)+1`.
- `src/reference.rs`: `ReferenceAudio::from_channels_full()` retains the full source for speech editing, unlike the 12-second reference-prompt truncation. It retains mono downmix, RMS adjustment and 24 kHz resampling. Explicit 30-minute input cap prevents unbounded allocations, while the CFM cap further restricts final mel length.
- `src/backend/cubecl.rs`: `integrate_with_edit_mask`, `integrate_with_edit_mask_ode`, and `integrate_with_ode_method` use a shared CFM integration implementation; conditioning is zeroed in edited spans, the original mel is reinserted only at kept frames, and per-request DiT cache release occurs on both normal and error paths. Old prefix-mode inference remains unchanged.
- `src/cfm.rs`: explicit `OdeMethod::{Euler,Midpoint}`; midpoint evaluates the model at `x + v(x,t)·dt/2` and `t+dt/2`, then advances the **original** state with the midpoint velocity, matching the original fixed-step `torchdiffeq` midpoint scheme. Default `SamplingConfig` and old Euler inference are unchanged.
- `src/pipeline.rs`: `StandardF5Runtime::edit_speech`, `edit_speech_with_noise`, `edit_speech_with_noise_and_method`, `synthesize_with_ode_method` and `synthesize_with_noise_and_method`. Speech-edit decodes the **entire** restored mel, never the generated suffix alone.
- `src/backend/trtx_session.rs`: an edit-shaped session is identified by `reference_frames=0`, Vocos frames equal full CFM duration, and no Cross-Lingual reference-speed graph. Adds `edit_speech` with arbitrary frame mask; uses the existing GPU-resident CFM step graph, restores original frames in its last step and streams the *full* mel into Vocos. Standard prefix synthesis refuses to accidentally use an edit session.
- `src/backend/trtx_factory.rs`, `trtx_text.rs`, `trtx_worker.rs`: edit-shaped engine compilation, lazy LRU shape caching, persistent warmed thread protocol, and text + source-audio worker API. No new external dependency or secondary backend fallback.

## Numerical oracles

- `tests/tools/generate_speech_edit_fixture.py` **executes the frame-planning AST nodes copied programmatically from the exact user-provided original ZIP**, using real PyTorch tensor operations. 320 randomized editing cases include multiple intervals, replacement durations, mixed kept/generated frames, text-length duration extension, and time rounding near half frames. Rust masks, frame counts, and the **complete flattened FP32 mel conditioning** match: **320/320 PASS**, verified with an independent 64-bit FNV hash over every input mel bit. JSON metadata also saves f64 time bits because the default JSON floating parser can move a value by one ULP, affecting half-frame rounding; this is a fixture-transport issue rather than a Python/Rust numerical shortcut.
- `tests/tools/generate_speech_edit_ode_fixture.py` generates three PyTorch FP32 trajectories for a deterministic affine+CFG velocity network with a *non-prefix keep mask*. Cases: Euler with 3 steps, midpoint with 3 steps, midpoint with 5 steps + EPSS/sway. The CubeCL CPU test checks every output mel float against committed little-endian reference bytes; all three **PASS within max-abs < 2e-5**. PyTorch computes the Euler/midpoint equations independently; `torchdiffeq` itself was not installed or executed for these fixtures, so do not claim direct numerical parity against that package.
- TRT mock test checks restoration of noncontiguous preserved frames, full-length Vocos input, output audio finiteness, GPU Euler step count, and isolation of standard versus edit sessions. It does **not** execute TensorRT neural layers.

## Reproducible local checks

Rust 1.100.0; pinned offline vendor and `Cargo.lock` from CONT12; cargo target `/mnt/data/f5-vendor-cont09/target`:

- `cargo fmt --all -- --check`: **PASS**.
- `cargo test --offline --locked --lib -- --test-threads=1`: **56/56 PASS**.
- `cargo test --offline --locked --features tensorrt-mock --lib -- --test-threads=1`: **82/82 PASS**, **2 intentionally ignored** (real engine construction tests).
- `cargo test --offline --locked --features cubecl-cpu --lib cpu_speech_edit -- --test-threads=1`: **2/2 PASS**; 72 nonmatching tests filtered. The first cold compile was expensive, but once dependencies built the numerical tests completed in less than 2 seconds.
- `cargo check --offline --locked --features tensorrt-api`: **PASS**.
- `cargo check --offline --locked --features cubecl-cpu`: **PASS**.

For local TensorRT `trtx-sys` bindings: `LIBCLANG_PATH=/usr/local/swift/usr/lib` and `CARGO_HOME=/mnt/data/f5-vendor-cont09/cargo-home`.

## Remaining functional gaps (before claiming 100%)

- `CFM.sample` auxiliary inference branches `no_ref_audio`, `duplicate_test`, and `t_inter` are not all surfaced by the public runtime. They require an explicit feature-by-feature audit to distinguish genuine inference needs from diagnostic/training-only options.
- The **midpoint** algorithm is available in CubeCL, but the GPU-resident TensorRT CFM step/session currently uses Euler only; implementing a true midpoint update requires a two-stage velocity evaluation without losing the original state. Do not emulate it by incorrectly advancing from the intermediate state.
- Some Python inference *postprocessing* options (e.g. optional silence trimming) have not been reproduced as library entry points; ASR, UI/Gradio and network transport are separate application services.
- This contour does not run on a real NVIDIA device or with original pretrained checkpoints. Neither full TensorRT numerical parity nor actual synthesizer operability has been proven.

**Progress:** Feature completeness remains **estimated ~99%, not certified 100%**. Earlier 99% was overstated because speech-edit had been omitted from the inventory; after implementing it on both paths, remaining optional inference controls must still be audited. Numeric coverage and tests increased substantially. Hardware operability remains **43% (3/7), unchanged**.
