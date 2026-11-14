# Stage91 — global NSF fundamental phase

A long RVC recording is chunked for inference, but pitch comes from the same global F0 timeline. `GlobalNsfPhase` computes one oscillator seed per model input from the original absolute frame position and its reflected context. With the `RVC_NSF_GLOBAL_PHASE=1` flag set, this seed enters the NSF SineGen angle before the sine operation.

`RVC_GLOBAL_F0=1` is required and `RVC_GPU_F0=1` and `RVC_RMVPE_MEL_REUSE=1` are incompatible with it. The default flags continue to use the existing RVC source path.

For CPU mathematical tests using Rust 1.100:

```bash
rustc --edition=2024 --test research/stage91-nsf-phase/phase_test_harness.rs -o stage91_phase_tests
./stage91_phase_tests
```

For the independent Torch CPU SineGen comparator, compile `phase_probe.rs` as an executable and run:

```bash
rustc --edition=2024 -O research/stage91-nsf-phase/phase_probe.rs -o stage91_probe
python research/stage91-nsf-phase/test_torch_phase.py --probe ./stage91_probe --output torch_phase_reference.json
```

Windows A/B/BA full WAV tests use `Run-RVC-Full-Stage91.ps1` in the Stage91 EXE package. GPU performance and audible quality have not been measured.
