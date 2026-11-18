# Fish Speech 1.5 — next implementation cycle after CONT7

Read `CONT7_REPORT.md` and `CONT7_NEXT_AGENT_PROMPT.md` first. The local source was built from the Git Bridge snapshot at `e89871a0` plus user-supplied CONT5 and CONT6 snapshots; the remote branch later advanced to `218f920a`. Do not equate the old Git Bridge snapshot with the remote HEAD, or push this tree over newer commits without reconciliation. No CONT7 changes were pushed.

## Verified Linux status

- Rust 1.100.0 offline: Core 143 PASS, outer 3 PASS, `tensorrt-mock` 7 PASS, exporter 5 PASS, source-PyTorch six-scenario Conv/Deconv/Linear fixture PASS, `tensorrt-api` cargo **check** PASS.
- `src/trtx_lowering.rs` contains TensorRT calls for FishConv/TransConv and ConvNeXt Linear via 1x1 Conv2D, including SafeTensors F32 weight bindings. API compiled, **no TensorRT network built or executed**.
- Firefly DAG has 420 nodes and 450 weight requirements; 171 nodes have typed Conv/Linear lowering. The other 249 nodes and full graph orchestration still require work.
- GPU execution, full TTS, real checkpoint parity, TensorRT engine, Windows packaging, CUDA graphs, and performance results remain **NOT_TESTED_GPU / BLOCKED_ON_ARTIFACT**. No PyTorch/ONNX is part of intended Rust runtime.

## Priorities

1. Add NCHW-safe TRT lowering for native activation/elementwise/residual/gamma/mask, LayerNorm channel variants, and FSQ; build full Firefly topological graph and preserve source numerical semantics.
2. Validate exact checkpoint and config/tokenizer from Fish Speech 1.5, with provenance and license. Expand real Slow/Fast AR lowering, active-prefix KV cache, GPU sampling/EOS, reference pipeline and session orchestration.
3. Windows GPU test on TensorRT-RTX 1.7 SDK with NVIDIA RTX hardware: real builder, per-layer numeric parity, end-to-end TTS, engine caching and benchmarking versus warmed matching original TorchInductor.

Use Git Bridge artifact + offline vendor for local development; GitHub writes only via connector when ready to integrate, and do not edit unused legacy scripts.
