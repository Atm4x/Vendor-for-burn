# Fish Speech Rust prototype — CONT4 progress

**Linux architectural prototype: 54%** (engineering estimate). CONT3: 45%. This percentage is not end-to-end inference completeness or GPU readiness.

| Workstream | Estimate | Current status |
|---|---:|---|
| Original Python source audit | 100% of provided source-only ZIP | Checkpoint-derived shapes/key conventions unverified |
| Rust core, AR generation, KV and sampling | 75% | Synthetic CPU contracts; true BPE/weights pending |
| Slow/Fast connected graph | 65% | Numerically executable synthetic AR, profile design; actual TRT lowering missing |
| TRT blueprint and mock | 25% | Strict deferred operation list; `trtx` offline registry blocker |
| Firefly and audio CPU prototype | 55% | Numerical ConvNeXt, full toy HiFiGAN, STFT/logmel and FSQ equations; real weight mapping and VQGAN full graph pending |
| Windows GPU execution | 0% | Explicitly excluded from current Linux phase |

**Tests: 132/132 PASS** on Rust 1.100.0 (114 core unit tests + 12 Firefly fixtures + 4 logmel fixtures + 2 FSQ formula fixtures). Both Debug and Release. `cargo fmt --check`: PASS.

No models, Python, torch, libtorch or TensorRT are required by the compiled Rust core. PyTorch / torchaudio are only used to generate independent development fixtures.
