# Stage 79 — original PyTorch CUDA vs native Rust: full-WAV throughput and chunk geometry

Basis: `Atm4x/Vendor-for-burn` PR #2, Stage78 HEAD `92388c599b51ba9954f9b7285837f49d0c38ceaa`; user-provided `tts-with-rvc.zip`, `tts-with-rvc-main (2).zip`, and `RVC_STAGE77_FULL_WAV_RESULTS.zip`. **No Stage79 RTX GPU measurements exist yet**; all speed hypotheses below are unverified.

## What is actually known (real RTX 4060, Stage77)

89.939 s audio, 40-kHz PCM16 output (3,597,570 samples), 32 chunks, NSF implicit GEMM, prepack enabled, persistent GRU, identical recording and weights, CUDA Graph off:

| RMVPE Conv2d | Warm full-RVC, ms | PCM16 RMSE vs Direct | Max PCM16 delta |
|---|---:|---:|---:|
| Direct | 12444.15 | 0 | 0 |
| ImplicitGemm/TF32 | 10585.64 | 985.61 | 31490 |
| TF32×2 separate two Conv2d | 11083.80 | 1.37679 | 46 |

Ordinary TF32 is not an acceptable drop-in: near-boundary voiced decisions caused huge PCM differences. Correct compensated TF32×2 is ~12.3% faster than Direct but 4.7% slower than ordinary GEMM. Stage78 packed high+residual into one enlarged Conv2d; **GPU/JIT and full WAV results pending**. The double-batch formulation does not halve FLOPs; it may save launches or lose to occupancy/memory traffic.

## Verified source-level differences against original tts-with-rvc

The two uploaded original ZIP archives have identical meaningful Python logic; the differing source bytes are CRLF vs LF. Five normalized original module SHA-256 hashes are checked by `Run-RVC-Stage79-Compare.py` to detect a changed installed reference. Source code was not modified.

- Original reference `Config.device_config()` on a normal RTX selects FP16 with x_pad=3 s, x_query=10 s, x_center=60 s, x_max=65 s. FP32 has different windowing (x_pad=1, x_center=38, x_max=41). The full audio is highpass-filtered and RMVPE F0 is computed on its padded signal before synthesizer chunks.
- Native Rust benchmark uses FP32, default `core_frames=300`, `context_frames=40`, `overlap_frames=12`, `trailing_frames=4`, i.e. 3 s core with 0.4 s context on each side, and a fixed 10 s maximum model input. It computes chunk F0 within each chunk unless a separate opt-in is chosen. Comparing WAV samples against the original is **not** expected to be bitwise exact.
- Original Python pipeline reads and reconstructs the FAISS index when entering each full-inference `pipeline` call. Rust loads the index/model once and keeps them resident. These are real design differences, not just Conv2d kernel differences.
- Original PyTorch pipeline timer includes internal source filtering, F0 and final output scaling; Rust `warm_median_ms` covers resident inference and excludes `prepare_ms` plus `postprocess_write_ms`. The aggregated ratio in the Stage79 report is labelled **non-equivalent** rather than proof of native speed superiority.

## Unmeasured but mathematically grounded hypothesis

Under the current Rust chunk planner, 89.94 seconds of input would be processed as follows (context and overlap counted repeatedly):

| Core frames | Tensor batch | Chunk count | Sum of model-input durations | Redundancy |
|---|---:|---:|---:|---:|
| 300 | 2 | 32 | 120.54 s | +34.0% |
| 600 | 1 | 16 | 105.18 s | +16.9% |
| 900 | 1 | 11 | 100.38 s | +11.6% |

If runtime scales with redundant frames, 900-frame cores might help. But the model's max activation memory, batch scheduling, CUDA occupancy, and compiler kernel selection can overturn that prediction. The current test intentionally does **not** silently change production defaults. The longer chunks may also change pitch decisions and overlap seams; WAV comparisons must therefore use a Direct control at the *same* geometry and human listening.

## Stage79 test harness

Three scripts under `research/stage37-windows-gpu-bench/windows/`:

1. `Run-RVC-Stage79-Compare.py` — single entry point with the explicit user's Torch CUDA environment `D:\Projects\tts-with-rvc\.venv\Scripts\python.exe`, compares original PyTorch FP32+FP16 and native Rust, writes one ZIP. Uses source normalized-SHA check for the original ZIP and SHA-256 for all five private inputs.
2. `Run-RVC-OriginalTorch-Stage79.py` — loads unmodified original `tts_with_rvc` from the specified root, uses original `VC/Pipeline` API, synchronizes CUDA before and after each iteration, emits WAV and JSON; repeats 3 times for each precision.
3. `Run-RVC-Stage79-ChunkSweep.py` — tests native Direct / TF32×2 / Stage78 packed mode across selected core:batch geometries, compares same-geometry PCM16 to Direct, checks runtime policies, reports timings and ZIP. Requires `core_frames >=25`, `core+84<=1000`, `batch_size<=32`; never substitutes reference F0.

Use the Stage78 `tts-with-rvc-rs.exe` from the successful Windows Actions workflow and copy these scripts alongside it. Place `CrazyMita.pth`, `hubert_base.pt`, `rmvpe.pt`, `CrazyMita.index`, and `MitaKind.wav` in its `embed` directory. Then run in PowerShell:

```powershell
& "D:\Projects\tts-with-rvc\.venv\Scripts\python.exe" .\Run-RVC-Stage79-Compare.py --original-root "D:\Projects\tts-with-rvc" --grid "300:2,600:1,900:1" --native-modes "direct,tf32x2,batched" --torch-precision both --runs 3
```

Generated `RVC_STAGE79_ORIGINAL_COMPARISON_RESULTS.zip` contains per-case WAV, CSV/JSON, subprocess logs, original PyTorch metrics, input hashes, internal same-geometry parity. **It never includes private model weights, index, or the input WAV.** If the original source is not located at the default derived root, pass `--original-root` explicitly. If PyTorch dependencies are absent, the Python case fails and logs its error without hiding successful Rust results.

## Local verification

- Loaded exact PR #2 head through validated self-contained Git bundle: `git bundle verify` PASS, head `92388c599`.
- Original uploaded archives' normalized source equivalence PASS; all five SHA-256 original checks PASS against unmodified extraction.
- Python syntax `py_compile` PASS; Stage79 script CLI parses.
- Mocked native GPU executable with Direct / separate TF32×2 / packed TF32×2 produced three valid case reports, two parity pairs, zero harness failures. This only tests orchestration, **not CUDA or audio quality**.
- Actual Torch CUDA, real voice/checkpoint, RTX latency and Stage78 packed GPU JIT: **NOT RUN in this Linux environment**.

## Next R&D based on measurements, not guesses

1. Determine best core/batch pair separately for native Direct and compensated routes, avoid OOM, inspect WAV at seams, use repeated warm runs with identical model and source.
2. Record original FP32 versus FP16 performance using **the exact original pipeline**. Analyse stage times, not just whole-output RTF; explicitly attribute indexing, global F0, chunking and inference differences.
3. If Stage78 packed is correct and faster, design a reusable CubeCL tile-level two-component GEMM accumulating high and residual without scratch `[2B,C,H,W]` or separate output slicing. Benchmark it over several Conv2d shapes from RMVPE and HuBERT and test output against Direct.
4. If longer cores yield speedups, implement a resource-budgeted opt-in chunk planner choosing core frames from a checked GPU memory estimate. A production default change requires multi-voice F0 parity, boundary quality and reproducibility checks.

Original scripts/legacy implementations remain read-only; no diagnostics-only reference-F0 replay is used in production.
