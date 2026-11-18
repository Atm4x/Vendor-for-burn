# Fish Speech 1.5 CONT9 → next implementation pass

Begin from the CONT9 source-only ZIP and read `docs/CONT9_REPORT.md`, the original user-supplied Python Fish 1.5 code, and the pinned implementation specification. Use Rust 1.100.0 and the offline Cargo vendor from the Git Bridge artifact. Work locally, no repeated GitHub requests. Do not modify unused legacy scripts. Preserve model architecture, split Slow/Fast AR cache ownership, and no PyTorch/Python at inference time.

CONT9 verified: 161 Core PASS, 3 outer PASS, 7 TRT mock PASS, 5 Python exporter PASS, `cargo check --features tensorrt-api` PASS, CLI structural coverage PASS. Firefly reference encoder from external precomputed `LogMel` to FSQ acoustic codes structurally lowerable (154/156; other two are AudioInput/LogMel host boundaries). Decoder from external codes to waveform structurally lowerable (263/264). TensorRT-RTX engine creation/execution, pretrained numerical parity, full TTS, Windows packaging and GPU performance all remain UNTESTED.

Prioritize the **Slow/Fast AR backend**, not additional mock counters: FishEmbedding → QKV/RoPE → GQA and active-prefix KV updates with separate Fast cache reset, sampling and EOS GPU synchronization elimination. Existing TRT blueprint now lowers FP32 RMSNorm and static LastToken. Add independent PyTorch-equation fixtures and validate shape, mask and cache semantics; build one native AR layer or complete subgraph rather than claiming a full model prematurely.

When actual Windows TensorRT-RTX SDK and pretrained checkpoints are supplied, run builder→engine→execute numerical tests first for Firefly encode/decode, then Slow/Fast, then full TTS, then warmed TorchInductor performance. Do not report a mock or FFI compile as real GPU PASS.

Maintain a global remaining-work progress bar. Deliver full source-only ZIP, delta, report; no GitHub push unless explicitly requested or clearly authorized. Never run `dotnet build` for Unity.
