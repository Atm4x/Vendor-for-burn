# Stage 61 — GPU-resident IVF FAISS and FP32 RMVPE postprocess study

Date: 2026-11-11. Experimental repository `Atm4x/Vendor-for-burn`, PR #2; base: published Stage 60 `f2f70f84052efb7982a754c2698fe0e70ffb75e8`. Rust 1.98.1; Burn 0.22.1 / CubeCL 0.11.3, vendored. **Research only; not the production default.**

## Why this stage

The original RVC path with `index_rate>0` issued a blocking HuBERT feature download (`Tensor::into_data()`) for CPU IVF/8NN retrieval and then re-uploaded the blended features to GPU. Stage 57 removed that transfer only when retrieval was disabled (`index_rate=0`). Stage 61 removes this particular host roundtrip while keeping the index and query feature data in CubeCL device allocations. Other host transfers remain, notably RMVPE pitch/f0 and final audio readback; this is not a claim that the *entire* pipeline is yet GPU-resident.

## Backend selection clarification

The default RVC build now enables the direct CubeCL CPU backend (`default = ["cpu"]`), not Flex. The CLI's `cpu` device chooses CubeCL CPU when both `cpu` and legacy optional `flex` features happen to be compiled. CUDA/WGPU packages explicitly pass `--no-default-features --features cuda|wgpu`, so each current prototype uses the correct compile-time CubeCL primitive backend. The long-term single-EXE, multi-backend runtime selection needs a separate backend-neutral dispatch design; the Stage 61 direct-primitive prototype does not yet implement it. There is no silent backend fallback.

## Architecture and boundaries

- `faiss_index.rs`: `packed_gpu_data()` exports the already-validated IndexIVFFlat centroids and inverted lists in original ID/slot order. There is no Python, faiss shared library or run-time index conversion tool.
- `faiss_gpu.rs`: `GpuIvfIndex` owns persistent CubeCL buffers for centroids, codes, ids and offsets, initialized **once when the inferencer loads**. Five CubeCL kernels perform (1) centroid squared distances, (2) nearest centroid, (3) candidate squared distances within the chosen list, (4) deterministic top-8 by distance/ID with inverse-square normalized weights, and (5) blended HuBERT features. The query is passed as its existing CubeCL `Handle`, no host `into_data()` for the query.
- `rvc_inference.rs`: setting `RVC_GPU_FAISS=1` explicitly opts into GPU retrieval. HuBERT feature repeat/protect remain GPU tensor operations; an unvoiced protect mask is derived from F0, which is already on CPU in the current RMVPE path. With `RVC_GPU_FAISS` unset, the original CPU index path is completely unchanged. Unsupported backend, incompatible shapes, or `nprobe != 1` fail explicitly: no hidden CPU fallback.
- This prototype only supports FAISS `IndexIVFFlat`, `nprobe=1`, `k=8`. The user-supplied CrazyMita index has 768 dimensions, 61,690 vectors and 1,581 centroids and satisfies these conditions. It is **not** a generic FAISS reimplementation, nor a promise of equivalent accuracy for arbitrary IVF settings.

### Numeric/algorithmic qualification

CubeCL and CPU may accumulate FP32 squared distances in a different order. Near equal-distance decision boundaries, top-8 identities or centroids may therefore differ. Existing CPU reference sorts candidates by (distance, original ID) and uses early-bounded L2 for centroids. The new GPU implementation preserves the requested deterministic ID tie policy *given its computed FP32 distances*, but it cannot promise bitwise equality in all near-tie cases. Do not silently claim exact full-WAV parity or change the default. A real GPU A/B runner calculates both hashes and full PCM16 max/RMSE instead of assuming perfect equality.

Host HuBERT readback measured by existing `RVC_PROFILE` changes from `B*T*C*4` bytes (CPU IVF) to **0 bytes** (GPU IVF). GPU upload of the initial index is amortized over the inferencer lifetime; rate/mask and pitch data are still small host-generated inputs. `profile retrieval` measures Rust host-side execution/submission, **not synchronized GPU-only elapsed time**; its value alone cannot establish a faster GPU query. The Windows runner additionally reports cold total elapsed time, also including model initialization. A warm/resident harness remains required to prove the performance effect.

## Executed local checks

- `cargo test --offline --locked --release -p tts-with-rvc-rs --no-default-features --features cpu,flex --lib`: **37 passed, 0 failed, 6 ignored**. The updated CLI gives direct CubeCL `Device::cpu()` priority over optional Flex. The ignored tests are opt-in private-asset and longer benchmark studies.
- Opt-in CubeCL CPU/LLVM synthetic GPU IVF test: **PASS**.
- Opt-in CubeCL CPU/LLVM test with the real CrazyMita `.index`: **PASS**, `7,680` compared FP32 output values, max absolute deviation `0` for selected real-index queries. This test does not establish whole-domain numerical identity.
- Prior `cargo check --offline --locked -p tts-with-rvc-rs --features cuda --all-targets`: **PASS** (before changing the default from Flex to CPU). CI now uses isolated `--no-default-features --features cuda`; runtime/package verification is pending.
- Prior `cargo check --offline --locked -p tts-with-rvc-rs --features wgpu --all-targets`: **PASS** (before changing the default from Flex to CPU). CI now uses isolated `--no-default-features --features wgpu`; runtime/package verification is pending.
- New CLI tests with `--no-default-features --features cpu,flex`: **2 passed, 0 failed**, including proof that `cpu` selects CubeCL CPU even when Flex is present.
- Pure `--no-default-features --features cpu --release` rerun did not finish before the build limit; stopped during LLVM/dependency compilation, so no PASS claim for that exact build.
- Git diff whitespace/syntax check: **PASS**.
- **NOT TESTED:** Full-model CUDA inference with GPU FAISS on RTX 4060; full-model WGPU runtime on NVIDIA/AMD; sustained warm inference speedup; all `nprobe` modes; Stage 61 full WAV parity on hardware.

## FP32 instead of FP64 in RMVPE

The network computes FP32 salience. FP64 is present only in the host's weighted nine-bin F0 interpolation and exponential conversion. This is a postprocess precision choice, not a fundamental tensor dtype requirement. An independent FP32 implementation was tested against the existing FP64 host reference:

- On three snippets of real `rmvpe.pt`/`MitaKind.wav`: **207 frames**, max absolute pitch error `0.000183105 Hz`, coarse-pitch bin mismatches **0**.
- On **2,048** synthetic salience frames: max absolute F0 error `0.002197266 Hz`, coarse-pitch bin mismatches **0**.
- These tests support a separate GPU FP32 F0 prototype. They do **not** prove pitch bins remain unchanged for every boundary input or that WAV output is bitwise identical after moving F0 to GPU. The current production FP64 implementation is unchanged.

## Windows A/B artifact

Workflow `.github/workflows/rvc-gpu-ivf-stage61.yml` builds separate Windows-native CUDA and WGPU EXEs on CI, verifies PowerShell parser syntax, and packages `Run-RVC-GPU-FAISS-Stage61.ps1` in each artifact. The Windows runner uses private assets local to the operator; it alternates original CPU index vs `RVC_GPU_FAISS=1` on 0.7-second real WAV segments, reports HuBERT readback bytes, full PCM16 parity, RMSE, hashes, retrieval/profile times and cold elapsed time. Artifacts must not include private weights, index or audio.

From the extracted archive, with the private input files copied to `./embed/`:

```powershell
powershell -ExecutionPolicy Bypass -File .\Run-RVC-GPU-FAISS-Stage61.ps1 -Voice ".\embed\CrazyMita.pth" -Hubert ".\embed\hubert_base.pt" -Rmvpe ".\embed\rmvpe.pt" -Index ".\embed\CrazyMita.index" -InputWav ".\embed\MitaKind.wav" -Duration 0.7 -IndexRate 0.75 -Device cuda -Rounds 2
```

Change `-Device wgpu` to test the WGPU artifact **on supported hardware**; this is not a validated AMD run merely because Windows CI compiled. Only the test user may supply the private models. The archive `RVC_STAGE61_GPU_FAISS_RESULTS.zip` contains all reported logs, summaries and output WAVs; avoid posting private input assets to GitHub.

## Next engineering steps

1. RTX 4060 A/B proof: run Stage61 with real CrazyMita weights, index and audio; verify no HuBERT feature download in GPU mode, correct readback metrics, full output quality and latency. Examine near-tie disagreement before enabling GPU index by default.
2. Run the WGPU artifact on an actual D3D12/Vulkan-capable device; test AMD GPU separately, not just RTX through WGPU.
3. Measure resident, warmed end-to-end inference and use GPU timestamps/markers to separate asynchronous dispatch from completion. Guard against huge GPU IVF allocations for long sequences, and verify buffer ownership/lifetime.
4. Stage62: investigate moving **RMVPE F0 + coarse pitch** to FP32 device operations with numeric edge-case/regression suite and no silent fallback. Remove residual F0 data roundtrip only after hardware validation.
5. Keep the integration target a single long-lived **TTS service worker** in NeuroMita Rust, not one worker per model or Python/native switch. No merge into production before full-quality GPU parity and resource-management proof.
