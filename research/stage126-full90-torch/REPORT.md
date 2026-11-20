# Stage126 — full-waveform original PyTorch CUDA benchmark

Reuses original Stage79/80 full-WAV comparison contract and Stage125 direct NTC routes. Historical audio: 89.939s; original Stage79 Python environment differed from the reference source in 4/5 hashes and was not an acceptable identical source baseline; Stage80 introduced the trusted-local PyTorch source snapshot and a torch.load(weights_only=False) compatibility wrapper. Five normalized source hash comparisons are recorded and mismatches disclosed.

The new Rust change adds only requested-route and precise timing-scope metadata to infer-wav-full-benchmark JSON. It DOES NOT change Conv1d math or output. The runner requires fields to match the selected route; incorrect EXE, backend or flags fail closed. It tests full WAV, CUDA, persistent GRU, FP32 direct NSF, Stage125 NTC routes control/linear/residue/combined, 300 core frames/2 tensor batch by default, ABBA-like reverse cycle order. All Rust candidates require exact PCM16 match to control.

Original PyTorch CUDA FP32 sets cudnn.allow_tf32=false and matmul.allow_tf32=false, and original FP16 separately. Full PyTorch pipeline has different F0/chunk/preprocessing and timer scope; no PyTorch/Rust PCM parity or matched-graph speedup is claimed.

Local verification of benchmark enhancement: Rust 1.100.0 offline cargo check cpu,cuda,wgpu PASS, Python syntax and 3/3 route/PCM/archive tests PASS. Mock whole 89.94s WAV native orchestration PASS. Real RTX/PyTorch measurements pending. One Windows CI job builds the executable and bundles all scripts; old Stage125 workflow remains manual only. Legacy code untouched.
