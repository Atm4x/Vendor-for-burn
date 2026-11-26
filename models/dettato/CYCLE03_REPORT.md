# Dettato Rust / Burn — Cycle 03 (2026-11-20)

## Baseline

Source branch: `codex/beat-this-windows-worker` at `9cfbdbf617bb040b0a5499d1b28d40f25016f873`. Development branch: `codex/dettato-burn-port-20261120`; its pre-cycle tree was verified byte-for-byte against the local `050191c2` source tree (`8bcbebcf09a6bb9a2bb949916bf5ed8631f70329`). Work remained inside `models/dettato`.

## Functionality added

- Native Grouper frontend: extended linear-power HCQT, finite geometric 200-tap PCEN, per-channel active-region normalisation, harmonic occupancy/demixing, audio profiles, note rasters.
- Burn `LaneFront`: strided stem, occupancy injection, six dilated depthwise blocks with harmonic pitch mixing, channel-normalisation, pitch attention with relative distance bias and onset/span readout.
- Burn `NoteNet`: note embeddings + 11 scalars, three 368-dimensional Transcriber tap embeddings, HDemix/Prof/Lane reader projections, seven temporal-RoPE Transformer blocks, 73-class instrument head, 32-dimensional track embeddings, keep logits and hidden states.
- Full-song 24-second window scheduling, optional overflow scheduling for crowded windows, triangular overlap weights, normalized note embeddings, and winning-window track-head states.
- `Transcriber::grouper_features`: song maps at the original 4 s sub-window stride, twelve per-note model scalars, assigned voice slot, and three matched 368-d internal feature taps. Only the requested notes' spans are accumulated, avoiding a whole-song `[T,88,368]` buffer.
- `Grouper::group_with_transcriber`: handoff to the existing `TrackHead` + original-format instrument-track decoder; independent same-instrument groups can be retained (`merge=false`).

## Local validation

- Original Python/PyTorch-generated test fixture added for PCEN and HDemix; corresponding Rust integration tests created.
- Python independent recurrence-vs-PyTorch PCEN **max absolute deviation 5.01e-6** for a deterministic `[2,24,7]` log-spectrogram.
- Python independent harmonic occupancy/demixing-vs-PyTorch **max absolute deviation 1.49e-8** for two synthetic notes.
- Git whitespace audit via `git diff --check` PASS.
- **Not verified:** Rust build, Rust test execution, safetensors load, all neural layer shapes or GPU inference. No Rust compiler or Cargo registry is available here. These new source files are not certified operational.

## Remaining functions

1. Drums: 44.1 kHz frontend, DrumNet, orchestral model, router, ensemble.
2. A440 tuning estimation/resampling and original-time restoration in the full orchestration.
3. One-call inference path across Transcriber, Evidence, Grouper and Drums, including no-notes / audio-length boundary handling.
4. PyTorch-vs-Burn complete per-layer and full-checkpoint parity, and compilation validation.

## Priority discipline

Functional coverage is tracked independently of numerical parity and runtime validation; the latter do not count as missing implementations. Performance/CUDA/TensorRT optimizations have not been started.
