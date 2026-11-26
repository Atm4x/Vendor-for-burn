# Dettato cycle 09 — offline Tracel LLVM, CUDA compilation and CPU execution

## Source and environment

- Repository: `Atm4x/Vendor-for-burn`, dedicated branch `codex/dettato-burn-port-20261120`.
- Base published HEAD: `3d010c065c832eef0954e5fe614457d9c0b2b5ef` (`d477edd10de64ea5ac8f2233c13eb634f7eb1408`).
- Scope: `models/dettato` only. No changes to `models/beat_this`, the vendored crates, or model checkpoints.
- Toolchain: Rust 1.100.0; Burn 0.22.1, CubeCL CPU and CUDA feature paths.
- LLVM: user-supplied `linux-x64.tar.xz`, Tracel LLVM `23.1.0-4`, `llvm-config --version` = `23.1.0`, NVPTX static libraries present.
- The supplied LLVM archive has SHA-256 `c394bb1b10ac785f16d3d92d688a9343ba869f720e9cf9ca3f1abf2f401f165f`. The installer accepts other archives with compatible layouts but does not establish their origin; users must verify provenance independently.

## Changes

1. Add `tools/install-tracel-llvm-linux.sh`: offline extraction into the directory expected by `tracel-llvm-bundler`, validation of LLVM version and NVPTX static library, and an installation marker to prevent a network download. This marker intentionally skips the bundler's online manifest verification; use only a trusted archive. The installer is idempotent and refuses to overwrite an incomplete installation.
2. Use `profile.test.debug = 0` for this standalone crate to avoid massive LLVM DWARF sections in CPU/CUDA test binary linking. This does not alter model computations or release settings.
3. Document offline CUDA/CPU build commands and constraints.

## Confirmed locally

- Clean installation from the supplied LLVM archive: **PASS**, including `llvm-config` version, `libLLVMNVPTXCodeGen.a` presence, and idempotent invocation.
- `cargo check --offline --no-default-features --features cuda`: **PASS**, after the local LLVM installation; measured warm build 29.8 s with two existing dead-code warnings.
- `cargo test --offline --no-default-features --features cpu --lib`: **55 PASS / 0 FAIL**, on CubeCL CPU with fusion/autotune.
- Compact CubeCL CPU test binary built and the real-weight Transcriber test started. **No completed neural forward** on that backend: it exceeded the bounded runtime. The test profile reduced the linker's debug payload.
- A separate release Flex CLI end-to-end attempt on a 0.25 s 48 kHz dual-tone recording exceeded a **260 s** limit. No resulting MIDI or end-to-end PASS.
- The combined `cuda,wgpu,flex` check was attempted but stopped on an external time limit while compiling dependencies; it is **not claimed as PASS**. Separate WGPU and CUDA compilation had succeeded in cycles 08 and 09, respectively.

## Verification limits and next practical test

This container exposes neither an NVIDIA compute device nor a WGPU graphics device. Compilation establishes API/feature compatibility, **not** GPU forward correctness, latency, or GPU parity. The entire four-model pipeline has not completed on real GPU hardware here. On a GPU host, run the release executable with `--device cuda:0` or `--device wgpu:0` using the original four checkpoints and compare the resulting MIDI to the Python implementation. Validate not just output shape, but note onset/offset, voice slots, instrument classes, track assignments, and drum-event timing. Measure warm throughput and memory separately from first-run kernel compilation.

The source and weights are retained outside Git in the working environment; `.tar.xz` and model weights are not committed. This cycle does not change the numerical implementation or claim a GPU speedup.
