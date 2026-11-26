# Fish Speech 1.5 — CONT32 functional acceptance matrix

## Source-level status

Functional requirements remain the original Fish Speech 1.5 Python archive and the 22-item baseline. GPU execution and numerical parity are separate operability checks, not functional gates.

- Functional status: **19 DONE, 3 PARTIAL, 0 MISSING / 22**; equal-weight status index **93.18%**.
- Fixed detailed acceptance baseline: **29/36 = 80.56%**, up from CONT31 **28/36 = 77.78%**.
- Operability: historical **7/11 = 63.64%**, unchanged. Actual NVIDIA TensorRT, CUDA NVRTC compilation, checkpoints, timings and PyTorch CUDA parity are not validated in this environment.

## Remaining items

| ID | Gates | Status | Exact remaining requirements |
| --- | ---: | --- | --- |
| F08 | 4/5 | PARTIAL | Built-in MP3/AAC/MP4/WMA decoding without FFmpeg or separate codec libraries |
| F11 | 6/6 | DONE | No source-level gaps; GPU validation tracked separately |
| F12 | 7/7 | DONE | No source-level gaps; GPU validation tracked separately |
| F13 | **6/6** | **DONE** | No source-level gaps; actual CUDA kernel launch, benchmark and parity tracked separately |
| F14 | 2/5 | PARTIAL | Resident Naive AR TensorRT path, GPU public model facade, multi-codebook sampling/EOS/cancellation |
| F22 | 4/7 | PARTIAL | TensorRT workspace/unmanaged memory accounting, warmup/unload, bounded scratch pool |

## F13 acceptance gates

- [x] Fast codebook ordering, Fast KV reset, initial semantic-derived acoustic code, cancellation and EOS ordering.
- [x] Device-side Fast acoustic top-p, repetition penalty and exponential-race sampling.
- [x] No GPU→CPU readback of logits after each Fast codebook; tokens remain on device until complete acoustic column.
- [x] Shared CPU/CUDA generator controller with validated backend contract.
- [x] CUDA asynchronous memory ownership until stream retirement, including failure and cancellation.
- [x] **Device-side semantic sampling and EOS**: full Slow semantic logits remain on device, new GPU kernel computes top-p, penalty, sample and EOS bit. CPU receives an 8-byte token/flag control result, not the full vocabulary logits; controller verifies flag agreement before Fast work. It still completes the acoustic column for EOS, as in the original Python implementation.

## Semantic sampling algorithm and caveats

One CUDA thread block executes a 32-pass FP32 score-key threshold search with shared reductions to find the exact top-p boundary without O(V²) sorting. Repeated score ties are resolved deterministically by token ID using a bounded prefix-rank search. The final token is selected by independent exponential-race keys on device. Memory usage is O(256) shared words plus input and output buffers; the host history mask upload is O(V) per step. The vocabulary is capped at 1,048,576, well beyond the supplied 1.5 semantic vocabulary.

**Performance qualification:** One-CTA, repeated full-vocabulary passes may underutilize large GPUs. This implementation is bounded and source-correct in host emulation but is not an established latency improvement until profiled with real NVRTC, TensorRT and checkpoints on RTX. The 8-byte result is synchronized at every semantic-step boundary because the existing CPU controller still needs the token and EOS decision; complete GPU-driven autoregressive loop execution is not claimed.

## Verification

- Core full test suites: **230 PASS, 2 ignored** (202 unit + 28 integration).
- Outer default crate: **53 PASS, 2 ignored**.
- `--features cuda-driver,tensorrt-api` library: **67 PASS, 2 ignored**; `cargo check` PASS.
- TensorRT mock: PASS across outer crate and integration suites.
- `tools/test_cuda_semantic_sampling_host.py`: **65 PASS** single-thread C++ emulation versus independent sorted, source-compatible Python sampling reference, including up to 155,000 tokens.
- Rustfmt and Git whitespace checks: PASS.
- Authentic GPU execution, CUDA NVRTC compilation, measured latency and full PyTorch numerical or statistical parity: **NOT TESTED**.

## Next functional target

F14: implement and connect genuine Naive AR TensorRT inference and GPU facade. F22 remains after F14. F08 remains intentionally open pending codec implementation or a revised dependency policy.
