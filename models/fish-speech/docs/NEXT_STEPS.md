# Next agent: Fish Speech 1.5 → Rust prototype CONT5

## Current verified state

Read `CONT4_REPORT.md` and `PROTOTYPE_PROGRESS.md` first. Linux Rust 1.100.0, source-only Fish Speech 1.5, no model checkpoints, no real TensorRT-RTX. **132/132 PASS**, 54% estimate of Linux prototype. The actual original source path is `vendor/Fish-speech-pipeline-main`.

## Priority tasks without TensorRT SDK, model weights or GPU

1. Connect typed Firefly IR with true provenance of weight keys, shapes and execution order. For encoder: mono/audio→logmel→ConvNeXtEncoder→two FishConv/ConvNeXt downsample stages→GroupedResidualFSQ. Decoder: grouped FSQ index→project_out per group→two FishTransConv/ConvNeXt upsample stages→HiFiGAN. No placeholder Firefly graph that cannot enumerate inputs/edges.
2. Pin **vector-quantize-pytorch==1.14.24**. Current `fsq_cpu` implements index→code→project_out (with residual scaling), but only synthetic equation fixtures are validated. Obtain exact package source and test its real `GroupedResidualFSQ.get_output_from_indices()` with synthetic learned projection weights; add learned project_in+encode path.
3. Investigate weight parametrization format (`weight_norm` uses `original0`, `original1`); design safe `.safetensors` exporter manifest, strict names/shapes and handling of `generator.` prefix. No actual model weights supplied.
4. Implement reference-audio decoding/resampling interface (WAV and multiple sample rates) with CPU golden parity. `spectrogram_cpu` currently assumes mono 44.1kHz float input.
5. Optional: procure offline Cargo vendor closure of attached `trtx` exact lockfile (including `autocxx 0.30.0`); only then compile true mock FFI. Do not request full SDK or claim GPU inference in Linux.

## Commands

`cd core && cargo test --offline && cargo test --offline --release && cargo fmt --check`

Use the attached Rust 1.100.0 toolchain if system Rust is older. No `dotnet` builds; do not edit obsolete/legacy originals. Continue in new Rust modules; original Fish Python stays unmodified.
