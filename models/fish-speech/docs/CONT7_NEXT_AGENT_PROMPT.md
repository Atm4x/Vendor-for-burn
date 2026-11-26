# Fish Speech CONT7 -> CONT8

Use `docs/CONT7_REPORT.md` as the evidence ledger and user-supplied Fish Speech 1.5 Python source as the source of truth. Start from source-only `models/fish-speech`, not a stale Git Bridge HEAD. Work locally with offline vendor and Rust 1.100.0; GitHub only through its connector, no repetitive GET loops, and do not modify legacy/obsolete code.

Priority: complete the TRT-RTX NCHW lowering for Firefly and connect it into a single valid graph. Existing `core/src/firefly_trtx_conv.rs` now describes all 171 Conv/ConvTranspose/Linear nodes, `src/trtx_lowering.rs` compile-checked FFI for these layers. Expand with activation/elementwise/reduction/mask, then normalization; **prove real weight and execution parity only where model weights and SDK actually exist**. Use original module source to get exact LayerNorm epsilon and fsq encoding. Avoid treating the 171/420 node count as GPU inference progress.

Current local evidence: 143 Core PASS, 3 outer PASS, 7 mock PASS, 5 Python exporter PASS, 6 source-PyTorch convolution/linear fixture cases PASS, `cargo check --features tensorrt-api` PASS. GPU SDK/Windows/real model weights absent. Keep all GPU gates `NOT_TESTED_GPU`. No changes were pushed from CONT7.

Once additional low-level op lowering is implemented, rerun offline tests + `cargo check --features tensorrt-api`; produce an updated source-only ZIP, a small changes-since-CONT7 ZIP, report and global remaining-work progress bar. Only run Windows CI when real integration is ready for GPU testing. Do not claim TensorRT performance or full Fish TTS before actually executing pretrained checkpoints.
