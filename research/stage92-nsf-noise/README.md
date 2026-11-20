# Stage92: global counter-Gaussian noise field

`noise_test_harness.rs` compiles the active Stage90-92 tests using Rust 1.100.0 without CUDA, models, Python, or Cargo:

```
rustc --edition=2024 --test research/stage92-nsf-noise/noise_test_harness.rs -o stage92_noise_tests
./stage92_noise_tests
```

Full library/CLI typecheck:

```
cargo check --offline --locked -p tts-with-rvc-rs --no-default-features --features cpu,cuda,wgpu --lib --bin tts-with-rvc-rs
```

To run full-WAV Windows A/B with the original `embed/` folder, obtain the Stage92 executable from GitHub Actions and invoke `Run-RVC-Full-Stage92.ps1` with the same arguments as the Stage91 runner.

`RVC_NSF_GLOBAL_NOISE=1` requires `RVC_GLOBAL_F0=1`; the legacy random-number path remains the default. There is no claim of overall waveform parity across different chunk geometries.
