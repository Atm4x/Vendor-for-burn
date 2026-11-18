# Fish Speech CONT19 — remote synchronization status

## Published source

GitHub commit `342f1b361f03f448cf2d704b66206eee531513a1` adds the TensorRT engine-builder, strict device-I/O contract and named output binding.

## Not yet synchronized

The remote branch started this cycle at `218f920a6f3da70d5f2708a3d77c8b24c75bc592` (CONT5). The full local CONT19 source-only archive includes subsequent CONT6–CONT18 work and this cycle's CUDA stream backend. Those files are not all present in this branch. **Do not infer that the remote repository contains the complete implementation or run CI for CONT19 without reconciling the full source tree.**

Locally implemented in CONT19:
- `src/trtx_engine.rs`: serialized TensorRT engine build, device bindings and enqueueV3.
- `src/cuda_stream.rs`: optional CUDA Driver API context, stream, allocations, and pitched BF16 KV memcpy2DAsync.
- `src/cuda_trtx.rs`: maps AR segment bindings to TensorRT context on the same CUDA stream.
- `cuda-driver` Cargo feature and pinned cudarc 0.19.10.

CPU/Core and FFI compilation passed locally. Real TensorRT SDK engine build, actual CUDA copying and end-to-end inference remain untested.

## Offline vendor integrity

The provided offline vendor has a cudarc-0.19.10 checksum mismatch in the disabled `src/cudnn/mod.rs` module. GPU-feature compilation was possible only with temporary local checksum metadata normalization; the original vendor checksum file was restored. Restore a trusted vendor version before claiming reproducible offline GPU builds.

## Functional coverage

F01–F22: 15 DONE / 7 PARTIAL = 84.1% equal-weight coverage, excluding CUDA/Windows verification. Production Slow/Fast/Naive GPU inference, full GPU lifecycle and some reference-audio codecs remain partial.

## Next action

Synchronize the full local CONT19 source tree to GitHub using connector commits with expected-head lease, preserving this branch's existing commits. Then complete persistent engine/context registry and real GPU gate on Windows.