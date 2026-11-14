# Stage 89 — experimental NSF spectral alternative

This directory contains **research prototypes**, not a new runtime mode or a Windows executable. It does not change production Rust inference. The original private `CrazyMita.pth` is **not included**. See `../STAGE89_SPECTRAL_NSF_GLOBAL_RND_2026-11-14.md` for measured, unmeasured and rejected hypotheses.

- `spectral_conv.py` — overlap-save spectral Conv1d, explicit real-batched GEMM interface.
- `spectral_planner.rs` — standalone Rust 1.100 capacity-aware selector with unit tests; no external crate required.
- `atlas.py` — checkpoint-wide memory-constrained optimizer and analytic FLOP model.
- `test_spectral_conv.py`, `validate_resblocks.py`, `test_atlas.py` — numerical and planner correctness.
- `bench_cpu.py` — diagnostic CPU-only operator timings.

This is **not** a drop-in substitute for the Stage 88 CUDA implementation. Future GPU implementation requires a dedicated fused packing / FFT / Tensor Core / inverse-FFT pipeline and GPU verification.
