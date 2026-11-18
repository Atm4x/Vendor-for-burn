# Fish Speech 1.5 — continuation after CONT10

Start from `FISH_SPEECH_CONT10_MODELS_FISH_SPEECH_SOURCE_ONLY_2026-11-17.zip`. Read `docs/CONT10_REPORT.md` and supplied Python Fish Speech 1.5 `llama.py` (not upstream S2), plus original TensorRT-RTX design spec. Work locally using Rust 1.100.0 + vendored Cargo from Git Bridge; no repeated GitHub GET. Do not modify obsolete code or claim GPU execution without RTX SDK/weights.

CONT10: directly lowers Slow/Fast `TokenEmbedding`, QKV split/transpose, interleaved BF16-frequency RoPE, BF16 KV prefix into native TensorRT-RTX Network Definition. GPU FFI `cargo check` PASS; Core 166/166 and ordinary crate 3/3 PASS; exporter 5/5 PASS. Standalone `tensorrt-mock` tests were not re-verified in CONT10 (C++ compilation timeout); CONT9 had 7/7. Firefly coverage unchanged. Overall native TTS completion ~44%, GPU 0% verified.

Prioritize correctly implementing semantic-gated FishEmbedding, **in-place external stable-address KV update** with active-prefix attention, GQA causal SDPA, then a full weight-bound native single AR layer. Never choose an O(cache_capacity) copy on each decode step just to remove a deferred operator. Maintain explicit Slow/Fast cache ownership; reset Fast cache on every Slow AR frame; prefill Q_len>1 and cached decode Q_len=1 must have distinct masks. Reuse session-owned CUDA buffers, no CPU readback each token. Perform CPU/PyTorch golden and compile-time FFI checks first; GPU builder→execute requires real SDK/GPU. Separate model graph, weight storage, backend and session runtime.

Produce next source-only ZIP, delta, report with truthful PASS/UNTESTED gates and progressbar against full TTS, not only operator count. Push via GitHub connector ONLY if explicitly authorized; source work remains local.
