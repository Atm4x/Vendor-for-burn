# Stage 58 — RMVPE compact salience readback: rigorous CPU proof and CUDA test package

Date: 2026-11-11. Repository: `Atm4x/Vendor-for-burn`, experimental PR #2. Base: published Stage 57 tree. Rust 1.98.1, vendored Burn/CubeCL, LLVM/Flex CPU. No local NVIDIA GPU available. This is an **opt-in R&D experiment**, not a new production default.

## Scope and implementation

The existing RMVPE inference head emits `[batch, frames, 360]` FP32 salience. Decoding originally transfers all 360 values per frame to CPU, chooses the **first** maximum and computes a nine-bin weighted centroid in FP64, with threshold-dependent unvoiced output.

Added `rmvpe::compact_salience` using Burn/CubeCL tensor operations on the input device:

1. Compute the maximum for each 360-bin row; resolve ties deterministically by selecting the **lowest matching index** (no dependence on backend `argmax` tie behavior).
2. Check that **all 360 original values are finite** via a per-frame device reduction. Encode a nonfinite frame as center `-1`; CPU decoding rejects it.
3. Gather the nine offsets `-4..4` around the selected center on-device; clamp out-of-bounds gather indices and skip out-of-range samples when decoding.
4. Read back ten FP32 values per frame: the exactly representable center index and the nine values. The existing FP64 arithmetic, 20-cent steps, threshold and 9-bin range remain unchanged.
5. Unified RMVPE uniform and per-item mixed threshold decoding via `infer_f0_batch_thresholds`. The same compact option is available for single-input `infer_f0`.

The new path is **disabled by default**. Set `RVC_RMVPE_COMPACT_READBACK=1` to use it. This prevents an unproven CUDA optimization from regressing the normal path. The library's normal dense readback still works when the flag is absent. There is no Python dependency and no change to model weights, NSF math or FAISS retrieval.

## Test matrix

- `cargo test --release --offline --locked -p tts-with-rvc-rs --lib --features flex`: **30 PASS, 0 FAIL, 2 deliberately ignored performance benchmarks**.
- `cargo check --offline --locked -p tts-with-rvc-rs --features flex,cuda --all-targets`: **PASS**.
- `cargo build --release --offline --locked -p tts-with-rvc-rs --features flex --bin tts-with-rvc-rs`: **PASS**.
- Compact vs dense salience decoder: batches `1/2/4`, lengths `1/2/7/31/33`, thresholds `0/0.03/0.5/1.0`, upper/lower edge maxima, ties including uniform zero rows, arbitrary interior peaks; **exact FP32-bit equality for every decoded F0**.
- Corruption: NaN, positive and negative Inf in off-peak bins, malformed packed length and invalid threshold; **all rejected as expected**.
- Mixed per-item thresholds at batch size 4: **exact FP32-bit equality**.

### Real CrazyMita end-to-end

Compared the **same release binary** with `RVC_RMVPE_COMPACT_READBACK` unset vs set to `1`. Private files: `CrazyMita.pth`, `hubert_base.pt`, `rmvpe.pt`, `CrazyMita.index`, and `MitaKind.wav`. `RVC_CONV_WEIGHT_PREPACK=1` and the original deterministic seed. The input is resampled from 44.1 kHz to 16 kHz, output is 40 kHz.

| Segment | Index rate | WAV dense SHA-256 | Compact comparison |
|---|---:|---|---|
| 0.35 s from 0 | 0 | `54fd4485f9bfb6a6cd998bbc92ee56ff702d0e1d27a4f048b50105fc814df9ab` | byte-identical |
| 0.35 s from 0 | 0.75 | `54c1ebede9bbc97e0f08e697b9e85bd756190419da01e47e6a8adf9b07101638` | byte-identical |
| 0.70 s from 22.4 | 0 | `1e70e1b98fc455d76da9cf2ca138a6fd9b2b66200d5007f2ce3fab6315f24b12` | byte-identical |
| 0.70 s from 22.4 | 0.75 | `3f38de7b7fa270c0a0d7442033e58d605377ba41e074bac1c9d9d029bb1e1dad` | byte-identical |

For 35 RMVPE frames, `RVC_PROFILE` reports logical output readback **50,400 bytes dense** vs **1,400 bytes compact** (36× reduction). This is the amount of output tensor data returned to the host, **not** a measurement of PCIe latency, VRAM usage or end-to-end speedup.

### Controlled LLVM/CPU performance (important negative result)

`benchmark_compact_readback_vs_dense_cpu`, 12 alternated runs per mode; includes tensor reductions and materialization/readback on the CPU and does not include the RMVPE network. This test is ignored by default and executed explicitly.

| Batch | Frames | Dense median, ms | Compact median, ms |
|---:|---:|---:|---:|
| 1 | 35 | 0.0031 | 0.0537 |
| 1 | 70 | 0.0194 | 0.1011 |
| 1 | 200 | 0.0510 | 0.4123 |
| 1 | 1000 | 0.2453 | 1.5802 |
| 4 | 35 | 0.0282 | 0.3751 |
| 4 | 70 | 0.0632 | 0.6363 |
| 4 | 200 | 0.2063 | 1.2020 |
| 4 | 1000 | 1.0112 | 5.1927 |

**Conclusion: compact is slower on Flex CPU** due to additional tensor operations, despite reducing the bytes read back. No assertion of CPU or CUDA speedup is justified. Existing dense production default remains correct. In a CUDA environment the ratio of D2H synchronization/transfer to kernel dispatch overhead can differ; actual hardware experiments are required.

## Windows CUDA package

New `Run-RVC-RMVPE-Compact-Stage58.ps1` runs two or more alternating dense/compact process pairs with the user's private real weights and WAV. It checks executable return codes, reported readback bytes and full WAV SHA-256 equality, collects RMVPE stage timing and cold-process time, and produces `RVC_STAGE58_RMVPE_COMPACT_RESULTS.zip`. It deliberately does **not** report CUDA Graph replay, because full-route hardware graph remains unimplemented/unverified. No runtime Python/PyTorch required.

New GitHub Actions `rvc-rmvpe-compact-stage58.yml`: build Windows Rust/CUDA EXE and package script. Includes a PowerShell parser check to catch variable-interpolation syntax errors before distributing a runner. CI does not have the user's RTX 4060 or private weights. Compilation and script packaging are not hardware runtime proof.

## Remaining work

1. RTX 4060 A/B using Stage 58 Windows artifact: validate GPU parity, log per-stage latency, memory and effective improvement/loss across 0.35/0.7/2 s and both FAISS modes. Do **not** enable compact by default unless hardware results establish benefit.
2. Prefer **one fused CubeCL salience reduction/gather kernel** over multiple Burn ops, to reduce per-frame GPU launch cost. Prove nonfinite/tie semantics on CPU first and validate CUDA runtime separately.
3. For `index_rate>0`, independently investigate GPU→CPU→GPU crossings around FAISS. Do not change exact IVF search semantics without full nearest-neighbor and audio parity.
4. Expand long input/chunk and speaker-option tests, then continue shared TTS-type worker integration separately from model-specific inference logic.

**Not completed:** local RTX 4060 runtime validation, production auto-enable, whole-RVC CUDA Graph, automatic native runtime distribution to clean Windows installs or NeuroMita shared-worker integration.
