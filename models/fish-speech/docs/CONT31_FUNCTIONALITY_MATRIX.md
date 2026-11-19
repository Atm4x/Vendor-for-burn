# Fish Speech 1.5 — CONT31 functional acceptance matrix

## Progress policy

The acceptance basis remains the supplied original `Fish-speech-pipeline-main (1).zip` and the existing 22-component functionality matrix. GPU operability remains separate from source-level implementation. The old equal-weight status index does not change until an entire F-item moves from PARTIAL to DONE.

- **Functional item status**: 18 DONE, 4 PARTIAL, 0 MISSING / 22; historical index **90.91%**.
- **Fixed acceptance baseline**: **28/36 = 77.78%**, up from CONT30 **26/36 = 72.22%**.
- **Operability**: historical **7/11 = 63.64%**, unchanged. No authentic Windows/RTX checkpoint execution was performed.

## Remaining functional items

| ID | Acceptance gates | State | Remaining requirements |
| --- | ---: | --- | --- |
| F08 | 4/5 | PARTIAL | Standalone MP3, AAC, MP4/M4A, WMA/ASF decoders without external FFmpeg/runtime libraries |
| F11 | 6/6 | DONE | No source-level requirements; actual TensorRT execution is tracked under operability |
| F12 | 7/7 | DONE | No source-level requirements; actual TensorRT execution is tracked under operability |
| F13 | **5/6** | **PARTIAL** | Source-equivalent semantic EOS determination and bounded termination signal on GPU; Slow semantic sampling still requires a host readback once per generated column |
| F14 | 2/5 | PARTIAL | Resident Naive TensorRT execution, public GPU facade, multi-codebook sampling/EOS and cancellation |
| F22 | 4/7 | PARTIAL | TensorRT workspace/unmanaged accounting, warmup/unload, bounded device scratch pool |

## F13 — new source-level gates in CONT31

- [x] Source-order Fast codebook loop, Fast-KV reset and prime, previous-code feedback, cancellation between codebooks (existing).
- [x] **Device-side acoustic sampling**: CUDA NVRTC kernel implements once-only repetition penalties, stable descending top-p filtering before temperature, and independent exponential-race token selection. Supports codebooks of size 1–4096; the uploaded source configuration uses 1024.
- [x] **Eliminate per-codebook GPU→CPU logits readback**: a selected I32 token remains on GPU and flows directly into the next Fast TensorRT graph. The full acoustic column is transferred to CPU once at the column boundary, not after every Fast network.
- [x] Preserve Core generation/controller semantics with one `DualArBackend::sample_fast_codes` extension, including error, fast trace, cancellation and CPU compatibility.
- [x] Explicit async ownership: buffers launched into the sampler remain leased until stream retirement, including cancellation/unwind; kernel module lifetime is synchronized before destruction.
- [ ] **GPU-side semantic EOS/termination**: the Slow semantic logits and sampled token still pass through the existing host sampler. Consequently semantic EOS is correctly respected by the common generator but is not yet a GPU-resident control signal. This prevents F13 DONE.

## Verification

- `cargo check --offline --locked --features cuda-driver,tensorrt-api`: PASS.
- `cargo test --offline --locked --lib --features cuda-driver,tensorrt-api`: **65 PASS, 2 ignored**.
- Outer default library tests: **53 PASS, 2 ignored**.
- Core full tests: **228 PASS, 2 ignored** (200 unit + 28 integration).
- `cargo test --offline --locked --features tensorrt-mock`: 53 outer unit PASS, 2 ignored; 8 additional mock/blueprint/resident integration PASS.
- `tools/test_cuda_fast_sampling_host.py`: **102 PASS** C++ host-emulated cases against independently implemented source sampling math, including tied logits, repetition penalties, top-p boundaries, and 4096-code vocabulary.
- `cargo fmt`, `git diff --check`: PASS.
- CUDA NVRTC compilation, actual kernel launch, genuine TensorRT execution, measured latency, real checkpoint and PyTorch CUDA parity: **NOT TESTED**. Host emulation does not replace GPU execution.

## Risk and next step

The new Fast sampler uses CUDA NVRTC available through the existing `cudarc` dependency (no new Rust crate, but **NVRTC is a runtime requirement**). Torch RNG bit-for-bit parity is not claimed; the exponential-race categorical sampling contract is matched statistically and the deterministic CPU host harness checks the kernel's selection rule.

To finish F13, implement scalable **device-resident Slow semantic sampling and EOS**, without an O(V²) algorithm for the large semantic vocabulary, then validate the real GPU kernel on Windows RTX. Do not mark F13 DONE merely because the existing host-side EOS is already correct.
