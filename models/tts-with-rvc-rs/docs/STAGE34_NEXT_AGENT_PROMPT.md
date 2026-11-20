# Stage 34 — NSF arithmetic kernel integration

Recover the `Atm4x/Vendor-for-burn` branch containing `STAGE33_ARITHMETIC_CONV1D_RND.md` and the two research Rust sources. Study the numerical identities and the recorded actual CrazyMita activation benchmarks; do not claim that the negative K11 design is faster.

Goal: optimize actual NSF Conv1d without treating Rayon as the main solution. Prioritize K3 F(2,3) with packed weight plan constructed once at model load, and fused input/output transformations in a native Burn backend kernel; preserve correctness under finite FP32, and explicitly test f16/NaN semantics before widening dispatch. Independently study an improved blocked K11 kernel. No implicit CPU fallback in CUDA code.

Use sparse sequential compilation (`CARGO_BUILD_JOBS=2`, `RAYON_NUM_THREADS=2`, no `cargo clean`); preserve untracked old benchmark files. Verify exact sample parity, real NSF and whole WAV→WAV on CrazyMita.pth and pre-existing WAV only after microkernel acceptance gates. Publish via GitHub Connector blobs/tree/commit rather than Git importer. Never publish private pth/pt/index/WAV.

GPU: design one fused CUDA/CubeCL kernel with pretransformed device weights and output-stationary tiles, benchmark against cuDNN on identical hardware; do not claim GPU acceleration without measurements.
