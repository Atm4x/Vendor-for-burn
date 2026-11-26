# Fish Speech CubeCL CONT42 — native top-p and Naive sampling

Date: 2026-11-19. Branch: `codex/fish-speech`. Based on the source tree published in CONT41 (`be0160eff92054d2d400b0ae9b1aa4b2962dc3e7`); no TensorRT branch changes.

## Implementation

- Added a native CubeCL top-p sampler that accepts a **device F32 logits handle** and a slice (offset, vocabulary), applies unique-history repetition penalty, float-key radix thresholding with deterministic tie IDs, nucleus selection and exponential-race selection. Only the selected token/EOS flag are read back (8 bytes).
- Preserved the Core `SessionRng` state evolution: host produces one independent exponential denominator for every original vocabulary item, including top-p-excluded items; the denominator array is uploaded. This **is not GPU RNG**. The kernel is single-lane/O(32 · vocabulary) and requires parallelization and benchmark acceptance before claims of GPU throughput improvement.
- `CubeClRequest::sample_top_p` owns budget accounting, request-handle validation, scratch retention, error poisoning and the token result boundary.
- DualAR routes `SelectionMode::Sample` through device sampling for both Slow semantic and Fast acoustic outputs; the Greedy path remains unchanged. Fast context is still forwarded through a device handle. No full logits readback is performed by the new sample path.
- Naive now supports `NaiveArForward::DevicePending`; the controller delegates semantic and per-codebook selection to backend hooks while keeping existing CPU-host implementations. `CubeClNaiveBackend` keeps both semantic/acoustic output handles on the device and samples each codebook by offset, without copying logits.
- Retains CPU-only Firefly; no changes to the accelerated TensorRT implementation.

## Verification and explicit limits

- Actual CubeCL CPU kernel tests compare sampled token IDs and EOS to `logits_to_probs` + seeded `SessionRng` for multiple top-p values, penalties, temperature, ties, offset slices, and vocabularies up to 257; invalid NaN rejected. These tests confirm selected-index parity for tested fixtures, not bitwise equality of intermediate probabilities on all inputs.
- The Naive device-pending controller test verifies semantic parameter override (0.1/0.1/1.0), 16-codeword history, callback sequence and EOS column without ever invoking the host sampling callback.
- Source-level `cargo check --features cubecl-backend` passed; actual CUDA/Windows not run. A **single-lane** GPU kernel is functionally correct for tested fixtures but is **not yet performance viable** for a large vocabulary; its O(32 · vocabulary) scanning should be replaced by a multi-thread reduction/radix implementation.
- Device-generated token embedding indices still require device-side validation. Some inter-graph token IDs are returned to CPU in the synchronous generator controller. C04 device-resident token/chaining, cancellation under real async load, and full end-to-end generation parity are **open**.

## Completed tests

| Check | Result |
| --- | --- |
| CubeCL CPU library (actual kernels) | 84 passed, 2 ignored |
| CubeCL feature library | 72 passed, 2 ignored |
| Default library | 58 passed, 2 ignored |
| Core library | 208 passed, 2 ignored |
| CubeCL backend `cargo check` | PASS |
| `cargo fmt --check`, `git diff --check` | PASS |

## Progress accounting

- C01 4/4; C02 6/6; C03 4/4; C04 **2/4**; C05 0/4; C06 0/4.
- Overall **16/26 = 61.5%** source/CPU functional gates; **0** Windows CUDA acceptance gates.

## Next

- Move sampling toward block-parallel GPU mass/reduction (and GPU RNG or compact deterministic RNG state) without changing top-p and repeat-penalty semantics.
- Implement GPU-resident sampling output as validated integer graph input, so the next Slow/Fast graph does not need a host token roundtrip; maintain EOS, request poisoning, cancellation, and resource lifetime invariants.
- Obtain real checkpoint tensors and CPU Firefly for a fixed-seed end-to-end trace, then Windows CUDA acceptance.
