# F5-TTS Native — CONT07: functional parity first

Date: 2026-11-20. Base: published `codex/f5-tts-rust-cont06` GitHub SHA `10cbe22f49df7b23d44af85f4e36518b4c56a462` (recovered from source-only ZIP). Scope: `models/f5_tts` only. `maryml` was **not opened**. No ONNX, no edits to Fish Speech/RVC/Legacy.

## Functional changes

1. **Standard long-text inference**: native UTF-8-aware sentence chunking mirroring F5 `infer/utils_infer.py`, max chunk size derived from transcript bytes and reference length, cross-fade, proportional allocation of `fix_duration` over multiple chunks, and one request-owned RNG stream. Exposed through `StandardF5Runtime::synthesize_chunked_with_pinyin`; regular `synthesize_text_with_pinyin` now uses the same process with 0.15 s default fade. Multi-voice scripts chunk per voice segment, then concatenate segments.
2. **Reference transcript semantic correction**: original Python `preprocess_ref_audio_text()` may append `". "`; `infer_batch_process()` adds one more blank when the last character is a single-byte character. Rust now implements this extra separator consistently for both text tokens and duration estimation. Speed correction for short batches uses `f64` to match Python float semantics and avoid one-frame errors from the f32 literal 0.3.
3. **Critical Chinese text correction**: original `list_str_to_idx` treats each complete pinyin syllable (`ni3`, `hao3`, etc.) as a *single vocab entry*. Previously, Rust incorrectly tokenized pinyin into characters. Now `F5TextFrontend` builds token strings, inserts word-boundary spaces according to F5's `convert_char_to_pinyin`, and maps full syllables to vocab IDs. `encode_segments` supports explicit rjieba-equivalent segmentation; `encode()` uses a restricted script-run segmentation fallback. **A native rjieba dictionary and real context-aware tone sandhi are still missing**.
4. **Duration clamps**: the low-level planner now permits a fixed duration shorter than the reference; the existing `MelFrames` layer enforces the same minimum of `max(reference_frames, token_count)+1` as Python CFM. Validation of invalid nonfinite durations remains.

## Numerical evidence (current cycle, not imported from earlier runs)

- Rust 1.100.0 standalone `rustc --test tests/offline_frontend.rs`: **16/16 PASS**.
- **1,006/1,006** UTF-8 sentence chunking cases match *the actual function* extracted from the user-supplied original `utils_infer.py` via Python AST.
- **505/505** segment-level pinyin tokenization cases match the actual Python `convert_char_to_pinyin` function via AST and a controlled segmentation/pronunciation fixture. This establishes token-boundary and mapping parity under the fixture, **not** independent parity for actual `rjieba` or `pypinyin` linguistic results.
- **52/52** PyTorch CFM schedules match the original `get_epss_timesteps` implementation, including custom sway and several EPSS/non-EPSS step counts. Per-timestep tolerance: `2e-6`.
- **4/4** PyTorch CFM trajectories match with deterministic starting noise, CFG, Euler updates and reference-frame inpainting. Per-element tolerance: `2e-6`.
- `rustfmt --check` for changed Rust files: PASS.
- **Full DiT PyTorch ↔ CubeCL CPU parity remains NOT RUN.** `cargo test --offline --features cubecl-cpu` cannot resolve `burn` (not in cache); normal `cargo fetch` cannot resolve `index.crates.io` due unavailable network/DNS. PyTorch full DiT fixture from CONT06 remains available and unchanged. Do not report it as passing.
- No new full GPU inference, Windows/TensorRT-RTX checks, pretrained audio generation, or performance benchmarks were run.

## Reproduction

From `models/f5_tts` using Rust 1.100:

```sh
rustc --edition 2024 --test tests/offline_frontend.rs -o /tmp/f5-offline-tests
/tmp/f5-offline-tests
```

Fixtures are stored under `tests/fixtures`. To regenerate them, run `python tests/tools/generate_{standard_chunk,cfm_pytorch,cfm_trajectory,pinyin_segments}_fixture.py /path/to/F5-TTS-main.zip` (each tool separately; the middle two require PyTorch). All scripts extract the required reference functions from the original ZIP; no runtime import of F5-TTS or network download is required.

## Functional coverage estimate: **70/100** (previous 67/100)

| Area | Now / maximum | Notes |
|---|---:|---|
| Core runtime/checkpoint/CFM | 14/15 | Need further complete-path parity |
| DiT computation | 12/25 | Full Burn numerical gate still blocked |
| Text preprocessing / normalizer | 14/15 | Syllable tokenization corrected; genuine Chinese segmentation and tone sandhi incomplete |
| Audio reference, mel, Vocos | 13/15 | Real config/weight mapping incomplete |
| Cross-Lingual speed/chunking | 9/10 | Complete-path parity not established |
| TensorRT-RTX backend | 0/10 | Not started |
| Integration, IPC, voice scripts | 8/10 | Long text, multi-voice chunking added; session worker/IPC remains |
| **Total** | **70/100** | Engineering functional estimate, not a benchmark |

## Workability (secondary): **3/7 = 43%**, unchanged

Older CPU parity and GPU compile evidence are retained from CONT05/CONT04 but were not rerun on this container. New Rust-only differential checks passed. Burn-dependent numerical tests, CUDA/WGPU hardware and Windows remain open.

## Next functional priorities

1. Restore the full Cargo/Burn dependency cache as an offline artifact, then **run** full DiT numerical fixture on CubeCL CPU and fix discrepancies (not just compile).
2. Provide self-contained Chinese segmentation/rjieba equivalent and tone-sandhi-backed pinyin; compare on an actual linguistic corpus (current fake fixture is intentionally limited).
3. Validate full-path standard and cross-lingual input tokens, duration, CFM trajectory and vocoder output against independent PyTorch fixtures, then implement the separate TensorRT-RTX backend.
4. Only after functional parity: pretrained checkpoints, actual GPU, Windows and performance.
