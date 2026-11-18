# CONT20 — next agent prompt

Start from the full `FISH_SPEECH_CONT19_MODELS_FISH_SPEECH_SOURCE_ONLY_2026-11-18.zip`. Rust 1.100.0; local offline code + Git Bridge/vendor only. Preserve the original Fish Speech 1.5 architecture and user instructions (no legacy/obsolete edits, no dotnet build for Unity, no network cloning/get storms). Github commits **via GitHub connector**; do not leave all new work only in ZIP. IMPORTANT: Github branch HEAD is at least `342f1b3`, but its previous history is CONT5 — do not assume CONT6–19 is synchronized. Reconcile before staging additional dependent files. Avoid overwriting changes on GitHub; use expected_sha.

Current verified code: `trtx_engine.rs` Builder/serialized engine + device-region bindings/enqueue; `cuda_stream.rs` CUDARC optional CUDA allocation/stream/actual pitched KV-copy; `cuda_trtx.rs` stream-linked TensorRT enqueue adapter. Tests: core 218 PASS, outer default 32 PASS, GPU-feature unit 34 PASS, original Python exporter 5 PASS; API compilation PASS. These are CPU/compiler checks, NOT actual GPU execution.

Caveat: offline `vendor/cudarc-0.19.10/src/cudnn/mod.rs` differs from its Cargo checksum metadata (CUDNN disabled). A temporary checksum normalization was used for only the driver-feature compilation and then restored. Clean offline build must resolve the vendor-provenance issue.

First deliverable: persistent TensorRT engine registry/caching per segment and per-model checkpoint, no per-token compilation; typed bindings and stream-safe ownership. Then native GPU sampler/EOS, Fast codebook loop, and full Slow/Naive weighted TRT inference. Only on a Windows+SDK machine measure build/infer/VRAM; never count untested GPU gates as passed.

Keep functionality progress F01–F22 separate from operability gates: currently 15 DONE, 7 PARTIAL = 84.1%. Do not boost it for mere compilation/GPU verification. State explicitly which original functions remain partial and what implemented code closes a gap. Publish full source ZIP + delta + MD and commit through GitHub connector.
