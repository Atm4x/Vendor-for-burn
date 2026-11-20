# Stage 93 — CPU-only geometric R&D

Direct Rust 1.100 commands from the repository root (no CUDA device necessary):

```sh
rustc --edition=2024 --test research/stage90-global-chunks/chunk_test_harness.rs -o /tmp/stage93_tests
/tmp/stage93_tests
rustc --edition=2024 -O research/stage93-cost-aware/measure_real.rs -o /tmp/stage93_real
/tmp/stage93_real path/to/MitaKind.wav
```

The tool uses the project's original `audio_io` and `preprocess` Rust modules; it does not use the Python resampler. Results are model *input geometry*, not measured GPU latency. Running `Run-RVC-Full-Stage93.ps1` later requires a Stage 93 Windows EXE and private models in `embed/`. To test a high attention penalty, choose `-AttentionMilli 100` (default 10). Production mode still defaults to `fixed`.
