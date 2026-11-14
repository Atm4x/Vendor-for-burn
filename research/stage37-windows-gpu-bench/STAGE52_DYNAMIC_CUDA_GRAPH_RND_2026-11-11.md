# Stage 52 — Dynamic-input CUDA Graph, persistent CubeCL allocations

**Date:** 2026-11-11. **Repository:** `Atm4x/Vendor-for-burn`. **Experiment:** PR #2 (`experiment/conv-kernel-rnd-stage33-20261110`). **Status:** source implementation, strict Windows A/B runner and CI packaging ready; offline Rust 1.98.1 CPU/CUDA/WGPU `cargo check --offline --locked` PASS; RTX 4060 Stage 52 hardware measurements **NOT YET EXECUTED**. No production NSF or legacy code was edited.

## Origin and exact Stage 51 conclusions

The two supplied RTX 4060 archives contain two independent rounds. Hardware capture was confirmed in 24/24 NSF cases (four variants × three frame lengths × two rounds), full FP32 parity was bitwise exact. Prepacked replay was consistently the fastest of the tested kernels, while the apparent largest *ratio* of ordinary/replay latency is not the same as the smallest *absolute* latency. Median-of-round-medians for Prepacked:

| Frames | Ordinary full operation, ms | Graph replay full operation, ms | Approx. speedup |
|---:|---:|---:|---:|
| 64 | 1.0732 | 0.2075 | 5.17× |
| 256 | 1.3577 | 0.2246 | 6.02× |
| 1024 | 1.3721 | 0.2374 | 5.75× |

All eight Stage 51 stream-policy configurations gave byte-identical FP32 files for each of 64, 256 and 1024 frames. `single + sequential` was stable (median 1.1441/1.1298/1.2021 ms); `single + parallel` had a 1024-frame outlier around 2.25 ms. These measurements concern fixed-input synthetic NSF, not end-to-end RVC.

## Stage 52 problem and solution

Stage 51 Graph capture embeds GPU memory pointers. Replacing a Burn tensor (`Tensor::inplace()` or `Tensor::from_data()`) may allocate another buffer and **cannot** prove safe reuse with changing audio. Stage 52 explicitly retains the original tensor, its exact CubeCL `Handle`, and the `Client` for the lifetime of a captured graph. A `PersistentInput` validates FP32, logical tensor dimensions, contiguous strides, and byte span and updates existing storage via `Client::write(&handle, Bytes)`.

Each replay uses the same captured pointer and the same calling thread's stream. The preceding replay is synchronized with full FP32 CPU readback before any subsequent write. The condition, harmonic source and input all change between steps. A basic captured `x + 1` probe verifies two disjoint writes, followed by a complete NSF captured replay checked against a separate ordinary forward for every step. Output also must vary across steps; a stuck/stale-input graph is rejected. On CUDA, `is_hardware() == false` is an immediate test failure (not a fallback result). Any bitwise output mismatch fails the process, even if absolute error is numerically small.

The test retains a fixed shape across replay iterations; this does **not** establish support for varying frame lengths within one hardware graph. Such shapes require separate capture/reuse policies. It exercises software fallback only on non-CUDA backends and never claims GPU gains from that behavior.

## Files created

- `models/tts-with-rvc-rs/src/bin/profile_nsf_graph_stage52.rs`: self-checking dynamic-input profiler and JSON result emission.
- `models/tts-with-rvc-rs/Cargo.toml`: add Burn `extension` + `cubecl` features for direct typed primitive access (no math changes).
- `research/stage37-windows-gpu-bench/windows/Run-NSF-Dynamic-Graph.ps1`: two rounds in inverted order, `baseline`/`prepacked`, separate processes and reports, stops on false hardware capture, stale outputs or any FP32 mismatch.
- `.github/workflows/rvc-nsf-cuda-dynamic-stage52.yml`: Windows x64 executable and runner packaging, compiled with locked Rust 1.98.1 and existing vendored dependencies.
- `research/stage37-windows-gpu-bench/STAGE51_RTX4060_GRAPH_STREAM_RESULTS_2026-11-11.md`: numeric interpretation of the two supplied Stage 51 result archives.

No changes to production model operators, no new fallback routing, and no separate model workers.

## Validation matrix

| Validation | Result |
|---|---|
| Git source bundle restoration, HEAD SHA verification | PASS, `e9134c27a796117de75d30c03100a67c28735509` |
| Stage 51 supplied archives: JSON inventory, genuine hardware capture, FP32 comparison | PASS |
| Rust 1.98.1, `cargo check --offline --locked` CUDA Stage 52 executable | **PASS** |
| Rust 1.98.1, CPU Stage 52 compile | **PASS** (`cargo check --offline --locked`) |
| CPU runtime smoke (software replay) | **NOT RUN** |
| Rust 1.98.1, WGPU Stage 52 compile | **PASS** (`cargo check --offline --locked`) |
| Stage 52 Windows CI packaging | **NOT RUN**, workflow only prepared |
| RTX 4060: live persistent-pointer updates, CUDA Graph capture, changed-input parity | **NOT RUN** |
| Real `.pth` voice model / HuBERT / RMVPE / `.index` / full audio I/O | **NOT RUN** |

`cargo check` proves types and code compile, but does not prove NVRTC/runtime behavior or allocation assumptions. The profiler also asserts the underlying dtype is exactly FP32 before issuing byte writes, so an unexpected backend dtype is rejected rather than silently reinterpreted. In particular, some CUDA tensors can have pitched layouts; the current profiler detects and rejects noncontiguous layouts instead of writing incorrect raw bytes. If the RTX 4060 case rejects a pitched tensor, implement an explicit stride-aware same-allocation update rather than removing these guards. CUDA runtime redistributables and licensing remain an independent deployment task.

## Windows procedure

1. Publish the local Stage 52 commit to the existing experimental branch via the provided Git Bridge; do not merge PR #2. Push triggers the `RVC NSF Stage52 dynamic CUDA Graph proof package` workflow.
2. Download the artifact `rvc-nsf-stage52-dynamic-cuda-graph-windows-x64`, unpack it on the user's RTX 4060 PC.
3. Run: `powershell -ExecutionPolicy Bypass -File .\Run-NSF-Dynamic-Graph.ps1 -Iterations 16 -Frames "64,256,1024"`.
4. Return `NSF_CUDA_DYNAMIC_GRAPH_STAGE52_RESULTS.zip`. The runner performs two reverse-order rounds across Prepacked and Baseline, emits per-case JSON, and fails closed on unsafe/non-hardware outcomes.
5. Inspect `hardware_graph`, `basic_graph_hardware`, `bitwise_equal`, `distinct_outputs`, `refresh_ms`, `replay_ms`, `ordinary_ms`, and errors. Note that refresh timing measures enqueue, not necessarily completion; replay readback is the synchronizing boundary. Compare overall latency inclusive of refresh for production cost analysis.

## Explicit limitations and next research

1. Validate capture with changing inputs and the RTX 4060 runtime *before* claiming Stage 52 success.
2. Measure capture setup, peak VRAM, memory pool, refresh+replay overhead and break-even reuse count.
3. If capture fails, isolate memory layout, stream identity, graph allocation lifetime and unsupported capture operations; do not switch silently to software replay.
4. Establish one cached capture per relevant fixed shape/model/backend, synchronized with the inference worker lifetime. Avoid mixing a generic worker abstraction with this model-specific optimization.
5. When moving to actual RVC end-to-end inference, request a voice `.pth`, `hubert_base.pt`, `rmvpe.pt` and any associated feature index/config, and test warm multi-request inference. These weights were not needed for Stage 52 because the supplied synthetic NSF checkpoint is sufficient.

## Publication status

The existing Git Bridge export workflow was refresh-committed upstream as `e9134c27` to acquire the source bundle. Stage 52 functional changes require a separate source commit and verified publication; the GitHub branch must not be described as updated before its ref is confirmed.
