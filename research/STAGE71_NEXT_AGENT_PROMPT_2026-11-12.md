# Stage 71 — CpuGemm validation and promotion

Base: PR #2 Stage70 experimental CPU selection `CUBEK_EXPERIMENT_CPU_GEMM`. Use the latest Git Bridge bundle; read `research/STAGE70_REAL_HUBERT_CPU_GEMM_RND_2026-11-12.md` and user-provided `RVC_STAGE70_REAL_HUBERT_CPU_GEMM_PROFILE.zip`.

Prior real HuBERT 12-frame first layer original 69,057.6ms HOT vs CpuGemm 22.57ms HOT, with bit-identical 9216 FP32 values. Real short RVC 0..0.24sec completed with CpuGemm, all 12 HuBERT layers 694.6ms in diagnostic, whole process wall 12.07s; repeat non-diagnostic wall14.11s, both WAV hashes identical. The old CubeCL CPU auto `multi-level` path used slow SimpleUnit; CpuGemm exists in tiled kernel but not default auto on CPU.

Next: test CpuGemm correctness and speed for real HuBERT 12,100,300,900-frame shapes, TextEncoder, RMVPE, Flow, irregular strides, mixed dimensions and available dtypes. Compare automatic shape/runtime-specific selection, preserving GPU backend behavior. Profile F0/NSF in short full CPU RVC next, then full WAV if runtime permits. If successful, promote CPU-specialized selection with supported-shape dispatch and verify regression tests. No model weights in git, no unverified GPU claims, avoid obsolete scripts.
