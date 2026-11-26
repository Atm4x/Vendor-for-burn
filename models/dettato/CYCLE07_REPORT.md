# Dettato Burn port — cycle 07 (2026-11-21)

## Source and scope

Source GitHub branch: `codex/dettato-burn-port-20261120` at `66104aeab4a421537f259668fdc5bef6f1c731b0`, tree `2a66d12a6cf3f04a14da2c25a0d1b3e0722458a5`. The local repository was reconstructed from the exact `9cfbdbf617bb040b0a5499d1b28d40f25016f873` Git Bridge bundle and the previous six source archives; `git write-tree` matched the published tree byte for byte. Changes in this cycle are restricted to `models/dettato`. All released model weight files remain outside Git.

## Functional changes

1. **One shared Transcriber pass**. `Transcriber::shared_run()` performs the original eight-second network-window inference with a two-second stride. Every window contributes to the fine map/hidden Hann averages (Evidence). Every other window, starting at a four-second multiple, also contributes to separate coarse map/hidden/stem Hann averages (Grouper). The common pipeline uses the shared results to decode notes, prepare the 769 Evidence features, repair notes, assemble 12 + 3×368 Grouper features, and assign MIDI tracks. It avoids rerunning the learned Transcriber for these later stages.
2. **Memory budget and compatibility fallback**. The shared pass caches three float32 tap arrays, `[frames,88,368]`, in system RAM; two map ensembles require additional storage. `set_shared_cache_budget_bytes(bytes)` sets the allowed memory budget (default 256 MiB). If the song would exceed it, the original bounded-memory, multiple-pass algorithm is used instead. A larger configured budget allows one shared pass for longer songs; it is not silently allocated by default.
3. **Unified note metadata**. The low-memory and shared-cache Grouper routes now use the same `grouper_metadata` implementation, avoiding silent differences in time-window selection, voice-slot assignment and pitch handling. Repaired MIDI pitches outside [21,108] are preserved as original MIDI pitches; only the spectral lookup lane is clamped to the 88-key range, matching Python.
4. **Transcriber time upsampling**. A many-hundred-element `Tensor::cat` of per-frame slices was replaced by a single reshape/repeat/reshape sequence. A regression test verifies the ordering including an odd-length truncation.
5. **Command-line interface**. `src/bin/dettato.rs` accepts `--weights`, `--input`, optional `--output`, `--separate-instruments` and `--cache-mib`. It runs the existing four-stage API and exports MIDI without requiring a separate Rust executable project.
6. **MIDI output path parity**. `Transcription::save_midi` now creates parent directories, as the original Python export does. Added a filesystem regression test.
7. **End-to-end smoke test**. Added an ignored real-weight test that attempts to transcribe short synthetic 48 kHz audio through the public pipeline and validates MIDI Type-1 serialization; this is a test definition, **not** a claim that an end-to-end real-weight run passed.

## Checks and observed results

- Rust: `1.100.0` from the supplied offline archive. Burn: `0.22.1`. CPU backend: Flex.
- `cargo check --offline --no-default-features --features flex`: **PASS**.
- `cargo test --offline --no-default-features --features flex`: **54 library PASS**, **3 CLI PASS**, **3 other ordinary integration PASS**, **0 ordinary FAIL**; released-weight tests are ignored unless explicitly selected.
- `transcriber_forward_parity` selected real-weight test (0.2 s input): **PASS**, all six output heads. The largest absolute deviation from PyTorch is **3.26×10⁻⁵**. Burn Flex forward took **41.1 s** in the debug/test profile; no speedup is claimed from the reshape rewrite.
- Shared Hann fine/coarse accumulation tests, tap-gather tests, out-of-range piano-note lane-clamping test, MIDI parent-path test and CLI parser tests: **PASS**.
- **Eight-second real-weight Transcriber forward**: attempted on Flex CPU with the released checkpoint and an eight-second silent signal; the 360-second execution limit expired (`timeout` exit code 124). **NOT PASS**. No output maps were produced, so this is a performance/time-budget limitation, not a correctness result.
- No released-weight end-to-end MIDI transcription has been confirmed, and no CUDA numerical or performance tests have been executed.

The cargo source overlay used for offline builds lives **outside** the repository. It contains package checksum metadata adjusted to the supplied vendor content, plus metadata-only stubs for four optional Symphonia codecs disabled in this test configuration. This workaround does not validate MP3/AAC/isomp4/ALAC compilation.

## Next work

1. Complete and benchmark eight-second Transcriber inference on an optimized backend; avoid conflating Flex/debug slowness with model correctness.
2. Run the full `full_pipeline_smoke` and compare decoded MIDI events with the Python original on the same recording, including instrument boundaries, two simultaneous same-pitch notes, retuning and percussion.
3. Extend numerical checks to nonzero Grouper tap inputs and verify the float32 shared-cache approximation against the upstream float16 Hann accumulation; determine tolerance or adopt explicit FP16 accumulation where required.
4. Reduce host/device synchronization for cached taps in the CUDA path and implement an optional disk/host-spill strategy for very long songs without four learned-network passes.
