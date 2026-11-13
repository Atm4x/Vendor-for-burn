# Stage 80 — Original PyTorch baseline recovery and chunk geometry

- Input evidence: Stage78 successful RTX4060 full WAV, Stage79 native passes but FP32/FP16 PyTorch launch fails with `AttributeError: Config.device_config`.
- Stage79 installed source has 4/5 mismatched fingerprints vs user-supplied original; do not infer PyTorch performance from this run.
- Fix: ship unchanged original Python package snapshot, use `Config` from snapshot via checked imports, compute original FP16/FP32 chunk settings, use existing CUDA Python environment only for third-party dependencies.
- Compatibility: checkpoint loading via trusted-local `torch.load(weights_only=False)` wrapper for newer torch.
- Ensure a real 3-point native sweep (300:2,600:1,900:1), with original FP16 and FP32.
- Timing caveat: Stage78 warm Direct 12.878s vs Stage79 Direct 19.712s on the same PCM and SHA. GPU state / non-interleaved order makes small gains inconclusive.
- Stage78 Direct 12.878s; ImplicitGemm 10.646s RMSE 985.608; TF32x2 10.823s RMSE 1.377; Batched 11.035s RMSE 1.384. No numerical/perf promotion.
- Next gate: real output ZIP with both successful PyTorch modes and native sweep, verify source hashes and compare latency across modes and chunk sizes.
- This snapshot does not change upstream source; no production defaults or legacy code changed.
