# F5 TTS CONT13 -> CONT14

Work only in `models/f5_tts`, new branch based on published CONT13. Preserve CubeCL and Rust model-specific architecture, no generic graph planner and no ONNX. Fish Speech source is read-only reference, `maryml` remains unopened. Main metric: functionality / PyTorch numerical parity. Secondary metric: hardware workability.

CONT13 provided native `trtx` direct-graph construction of F5 DiT v1 transformer backbone (22 blocks) and timestep MLP, `safetensors` weight import, device buffer contracts and scoped synchronized execution. Tests passed only at schema/compilation level because trtx mock doesn't allocate `INetworkDefinition`; do not claim engine-build success. Existing source parity is on CubeCL.

Next: native TensorRT **input path** (text encoder + ConvNeXt + input conv, mel/reference projection) and complete direct F5 DiT graph; reference exact `dit_input.rs` formulas, independently compare to PyTorch on synthetic fixture once real TRT runtime available. Follow with CFM GPU state & CFG. Reuse existing trtx crate in offline vendor, no transitive package reinstall. Keep all runtime DLL paths configurable. Do not alter used CubeCL code unless a verified mathematical bug is found; never edit Legacy.
