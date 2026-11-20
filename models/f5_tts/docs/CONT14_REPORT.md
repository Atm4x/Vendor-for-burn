# F5-TTS CONT14 — fused native TensorRT-RTX input, DiT and GPU CFM step

## Scope and architecture

Source: published CONT13, `6083af65f0e7203b37631527044cb488b96f9e26`. Only `models/f5_tts` is modified. `maryml` remains unopened, Fish Speech is a read-only API reference. No ONNX graph, PyTorch runtime, CubeCL fallback, generic planner or changes to other models.

Functionality is the primary metric; actual hardware workability is separate.

## Implemented

1. `backend/trtx_input.rs` implements F5-specific safetensors schema for word embeddings, ConvNeXt text blocks (depthwise group Conv1d, affine LayerNorm, GELU-erf, GRN, pointwise projections), mel + conditioning + text projection, two group-16 positional Conv1d layers with Mish. The network accepts pre-shifted INT32 token IDs, text-valid mask and F32 mel tensors. Positional embeddings have the same `[cos, sin]` channel arrangement as the existing CubeCL/PyTorch implementation.
2. `build_serialized_full_dit` constructs a **single** direct TensorRT-RTX graph containing text frontend, mel frontend, timestep MLP and all DiT Transformer blocks. Inputs: `noisy`, `condition`, `tokens`, `text_mask`, `sincos`. Output: `output`. It does not require an external hidden-state TensorRT engine or CPU round trips for intermediate embeddings. The older independent backbone/time engine builders remain supported.
3. `TrtCfgInputs::from_original_tokens` prepares conditional/unconditional rows and retains the *original* text-valid mask for both rows. In the original Python F5, `drop_text` zeroes the token IDs after the original pad mask has been calculated; treating unconditional token IDs as padding is incorrect. This is covered by a deterministic PyTorch fixture.
4. `backend/trtx_cfm.rs` adds a fixed-shape, device-resident CFM Euler-step graph. The graph batches conditional/unconditional DiT calls, combines CFG as `v_cond + strength*(v_cond - v_uncond)`, applies `next = state + dt*velocity`, and uses the `restore` scalar to blend the original reference mel into final state only on the last step. TensorRT engine inputs permit reusing the warmed serialized graph throughout the trajectory; two state buffers are swapped by the caller. All large intermediates stay on the device.
5. Explicit CUDA-buffer size, overflow, overlap and null-stream validation for the full DiT and CFM step; caller owns input buffers and stream synchronization. No runtime DLL search, no implicit host copies.
6. Refactored the existing TRT backbone and timestep MLP builders to share the same graph-construction functions with the fused path (no duplicated F5 math).

## Verified gates

| Gate | Result |
|---|---|
| Native ordinary Rust tests | 51/51 PASS |
| TensorRT mock Rust tests (including new weights, CFG packing, buffers, CFM) | **57/57 PASS**, 2 engine-build tests deliberately ignored |
| Pytorch/Python reference CFG host packing | **PASS** for 2 original sequences of differing token validity and mel shape; independent PyTorch fixture |
| TensorRT API compile (`tensorrt-api`) | **PASS** |
| TensorRT mock compile (`tensorrt-mock`) | **PASS** |
| Rust formatting | PASS |
| CubeCL WGPU recheck | Timed out in dependency compilation; no current-cycle PASS claim |
| Actual TensorRT engine build | NOT TESTED; mock cannot instantiate `INetworkDefinition` |
| TRT numerical parity against PyTorch | NOT TESTED; needs actual TensorRT engine and NVIDIA hardware |
| CUDA/WGPU hardware and Windows | NOT TESTED |

The prior independently verified CubeCL/PyTorch DiT, CFM, Vocos and end-to-end parity remains unchanged. This cycle does not prove that TensorRT builds the engine or matches PyTorch numerically. The fused numerical semantics are designed to match those previously tested independently in CubeCL.

## Remaining functionality

- Complete TensorRT-RTX-native Vocos and Cross-Lingual SpeedTransformer graph construction, without CubeCL fallback.
- Wire the CFM engine into a concrete owner of GPU allocations, warmed execution contexts, stream lifecycle and per-request dispatch. The current graph and buffer bindings are implemented, but there is no real TensorRT execution session or GPU allocator in this environment.
- Verify network construction, actual inference and numerical parity on an NVIDIA GPU; check parameter and shape constraints, dynamic shapes/engine cache and Windows deployment separately.
- Reconcile edge cases of Chinese phoneme/tone sandhi and reference conditioning on broader real corpora. Real pretrained checkpoints are lower priority than functional equivalence.

## Progress (engineering estimate)

- Functionality: CONT13 **86%** → CONT14 **89%**; new specialized native TRT frontend and CFM mathematical graph, incomplete TRT Vocos/Speed/runtime.
- Workability: **43%** unchanged, since no real GPU or Windows execution occurred.
