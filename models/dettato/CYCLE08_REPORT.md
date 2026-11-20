# Dettato — cycle 08: CubeCL execution path and Transcriber overhead

## Source

- Repository: `Atm4x/Vendor-for-burn`, branch `codex/dettato-burn-port-20261120`.
- Parent GitHub HEAD: `730fc36e193159dc0f8307fc35d9104f72d868a3` (tree `932962a2a1a9260f912ca8127c4cb7d3d9ddf8d9`).
- Scope: `models/dettato` only. Model weights and the external Cargo vendor overlay remain outside Git. No changes to `models/beat_this` or to obsolete scripts.

## What changed

1. **Runtime selection.** CLI `--device auto|cuda[:index]|wgpu[:discrete-index]|cpu|flex` with feature-gated constructors. `--backend` is an alias. A requested backend not compiled in fails clearly.
2. **GPU optimization features.** The `cuda`, `wgpu` and `cpu` Cargo features explicitly enable `burn/fusion` and `burn/autotune`, which the crate's `default-features=false` Burn dependency would otherwise omit. Flex stays separate.
3. **Sparse pitch mixing via a single convolution.** Each `PitchMix` tensor is constructed from the checkpoint's learned taps into a grouped 2D convolution with zero padding in the pitch axis. It preserves positive/negative offsets and repeated offsets. The host no longer builds a slice/concat/zero chain for every pitch offset in every forward.
4. **Persistent attention parameters.** RoPE trigonometric tables (up to 512 frames) and relative pitch bias (up to 128 lanes) are precomputed at model initialization, stored as tensors on the target device and cropped during forward. This avoids repeated CPU preprocessing/CPU-to-GPU uploads for normal windows; longer RoPE sequences use a dynamic path.
5. **Regression tests.** Added synthetic multi-channel pitch-mixing equivalence and CLI selection/feature-availability tests. Re-ran PyTorch Transcriber checkpoint fixture.

## Verification

- Rust 1.100.0, Burn 0.22.1, local offline Cargo source overlay.
- `cargo test --no-default-features --features flex` **PASS**: 55 library, 5 CLI and 3 ordinary integration tests, 0 FAIL. The full-recording test remains ignored.
- `cargo check --no-default-features --features wgpu` **PASS**, including `fusion` and `autotune`; this proves compilation only, not WGPU adapter/real GPU inference.
- `cargo check --no-default-features --features flex` **PASS**.
- The full short Transcriber test **after both PitchMix and attention-cache changes** achieved parity with original PyTorch: worst selected-map deviation `3.248e-5`; 45.54 s in Flex/debug. Both new CPU paths are numerically validated for this fixture, but there is no demonstrated speedup on CPU or GPU.
- `cargo check --no-default-features --features cuda` **NOT PASS** in this environment: external dependency `tracel-llvm-bundler v23.1.0-4` attempts a network download that is denied. This occurs before Dettato's CUDA code is compiled.
- CubeCL CPU with LLVM and runtime GPU parity/performance **not verified**. No NVIDIA/WGPU hardware is exposed in the container. No 8 s Transcriber or end-to-end audio-to-MIDI forward PASS has been claimed.

## Remaining blockers

- Provide the Tracel LLVM archive to the actual build container (the named `linux-x64.tar.xz` attachment was not visible to the filesystem or conversation file listing in this cycle), configure the bundler's expected installation, then compile CUDA/CubeCL CPU.
- Run the full eight-second Transcriber and `full_pipeline_smoke` on a real CUDA/WGPU adapter, compare output MIDI note events and instrument assignments to Python.
- Profile and optimize remaining conv/GRU and host-readback overhead. Fusion and autotune are now enabled for CubeCL, but this is not a measured GPU speedup.
