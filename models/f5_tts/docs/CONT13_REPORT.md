# F5-TTS CONT13 — native TensorRT-RTX DiT backbone and timestep subgraphs

## Scope

Functional coverage is the primary progress metric. This cycle starts from published CONT12 (GitHub commit `c9a1c6df92768d2bd32decabede8d2b36180de64`) and modifies only `models/f5_tts`. No `maryml` files were opened. Fish Speech's pinned CONT48 bundle was downloaded through Git Bridge, verified by SHA-256 and cloned locally **only as the API reference** for `trtx 0.9.1+rtx1.7`. No Fish Speech, RVC or Legacy files were modified, and there is no ONNX code path.

## Implemented

- `backend/trtx.rs`: F5-only shape and buffer contracts, non-overlap and overflow validation, checkpoint schema for all 22 transformer blocks, output modulation/projection, and both stages of the timestep MLP. All parameter keys and `[out, in]` weight layouts match the original DiT. Supports production v1 and reduced deterministic fixture dimensions.
- `backend/trtx_native.rs`: native TensorRT-RTX network creation via `trtx` and TensorRT C++ API. Directly constructs the v1 DiT **backbone** from 22 blocks of time-dependent modulation, LayerNorm with `eps=1e-6`, Q/K/V projections, split-half RoPE, attention, FFN with tanh-GELU, residual gates, final modulation and mel projection. Separate native timestep MLP graph; its input is 256 sinusoidal channels calculated by the Rust frontend.
- Fixed-shape TensorRT bindings for `[B,T,D]` input, `[B,D]` time and `[B,T,100]` output, plus the timestep engine. CUDA pointers and explicit stream come from the caller. The resident-context closure synchronizes before destruction on successful submission and errors.
- Externally supplied TensorRT-RTX DLL/shared-library path is supported by `load_tensorrt_runtime(Some(path))`; no bundled DLL search or Python subprocess.
- Optional features `tensorrt-api` and `tensorrt-mock`, independent of `cubecl-cpu/cuda/wgpu`, using the pinned `trtx 0.9.1+rtx1.7` API. Vendor code is not copied into the product source tree.

## Validated

| Gate | Result |
|---|---|
| Native ordinary Rust tests, no accelerator dependencies | **51/51 PASS** |
| `tensorrt-api` Cargo compile check against trtx-rs and its C++ header bindings | **PASS** |
| `tensorrt-mock` Cargo compile check | **PASS** |
| TRT weights, production shape/buffer contracts and all 22 fixture blocks | **5/5 PASS** |
| Actual TensorRT engine graph build with `trtx/mock` | **NOT TESTABLE**: mock creates a null `INetworkDefinition`; 2 tests ignored explicitly |
| GPU engine compile, execution, dynamic shapes, Windows | Not attempted; requires actual TensorRT RTX library and supported NVIDIA GPU |
| CubeCL CPU check with changed Cargo.lock | Timed out during dependency compilation; neither PASS nor FAIL |

These are construction and compilation milestones, **not** proof of TensorRT numerical parity, and not proof of an operating complete backend. The independent CubeCL backend and its previously passed PyTorch parity fixtures were not changed.

## Remaining functional gaps

1. Add direct TensorRT input path: text embeddings, ConvNeXt V2, mel+reference input projection and positional Conv1d. The completed backbone currently accepts already embedded hidden states; it must **not** call CubeCL as a fallback.
2. Add TensorRT CFM state integration and device-resident Euler/noise/CFG. CFM stepping is not wired to the new TensorRT backbone yet.
3. Add native TensorRT SpeedTransformer and Vocos decoder, or an explicitly native compatible renderer; no Python and no ONNX.
4. Replace fixed-shape engine compilation with controlled dimension profiles and warmed reusable engine/context caches. Check CUDA stream lifecycle and all TensorRT bindings on supported hardware.
5. Compare TRT-native graph outputs with the same saved PyTorch/CubeCL fixtures; then real checkpoint and Windows execution separately.

## Progress accounting

Previous engineering estimate CONT12: 84% functional / 43% workability. CONT13: **86% functional** (specialized TensorRT backend partially implemented), **43% workability** (no additional native GPU-runtime checks). This percentage is an engineering estimate of functional coverage and must not be interpreted as a certified fraction of original source-code functions.
