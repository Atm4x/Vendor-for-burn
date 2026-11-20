# Dettato — native Rust / Burn port

**Status: source-level implementations of all four neural stages and an end-to-end Rust API are present. The Rust code has NOT been compiled, its checkpoint import has NOT been run, and end-to-end Burn numerical correctness is not yet confirmed. See `CYCLE04_REPORT.md` for exact validation status.**

Original Dettato: Nathan Mills / kobimusic, Apache-2.0, <https://github.com/kobimusic/dettato>. See `LICENSE` and `NOTICE`.

This is an independent crate in `models/dettato`. `models/beat_this` is unchanged. It uses Burn 0.22.1 and can be configured for Flex, CPU, CUDA and WGPU backends where the upstream Burn kernels support the operations. There is no ONNX or Python dependency at runtime.

## Covered source features

| Component | Rust code currently present | Still missing |
|---|---|---|
| Checkpoints | `config.json`, named `safetensors` tensors for all four stages, F32/F16/BF16/F64 import | Automated full-key/shape parity on released checkpoints |
| Transcriber | HCQT frontend, Stem, PitchMix, CNN backbone, pitch/time attention, RoPE, bidirectional GRU, six heads, 8 s Hann overlap, two-slot note decoder | Numerically verified layer outputs and performance optimizations |
| Evidence | 31 map features, two 368-d hidden summaries, standardization/1/32 quantization, FP16 map feature rounding; CPU note context and beat-grid features; Burn embeddings/9-layer note transformer/RoPE/SwiGLU/heads; candidate decoding; keep/drop/insert repair | Compilation/parity of Burn Evidence network, batched/precision-optimized inference |
| Grouper | HCQT extended harmonic power, PCEN, standardisation, HDemix, spectral Prof, LaneFront (depthwise conv/pitch mixing/pitch attention), full 7-layer NoteNet, 73 instrument logits, normalized 32-dimensional embeddings, keep head, weighted 24 s song windows, track-head, Hungarian overlap, optional `merge=false`, MIDI track output, streaming collection of three Transcriber taps | Compile, weight-key/shape and numeric parity; tuning/retuning |
| Drums | 44.1 kHz frontend, 280 log-power bands, DrumNet, CQ branch, three OrchestralExpert models, router/ensemble, 61 drum keys, 13 timpani notes, velocity and 40 ms dedup | Rust build, weight load and numeric parity |
| MIDI | Type-1 multitrack export, programs, dedicated drum channel, complete 4-stage orchestrator, array / file API | Rust build and end-to-end inference validation |

The Transcriber can now collect Evidence hidden vectors in a bounded-memory **second inference pass**, without storing the entire song's `[frames,88,368]` hidden tensor. It accumulates only notes' onset and mean vectors, with the same per-frame Hann-normalized weights used for the output maps. The first pass still provides the shared note maps.

The Rust EvidenceNetwork accepts 769-component evidence vectors and uses the original checkpoint key layout (`emb.*`, `bb.blocks.*`, `keep`, `don`, `dur`, `ins.*`). Its CPU-side continuous feature preparation must be numerically compared with PyTorch before the port is considered complete.

Grouper now has a source-level Burn NoteNet and `Grouper::group_with_transcriber`. Its input contract includes 12 note scalars and three 368-channel Transcriber taps, gathered using the original coarse (4 s) sub-window stride without materializing whole-song tap maps. HCQT preprocessing, demixing and spectral profile extraction execute in native Rust on CPU; learnable convolutions, attention and projections use Burn. The saved 200-tap PCEN filter is implemented as an equivalent finite geometric recurrence rather than a temporal FFT. `merge_same_instrument=false` preserves independent same-instrument tracklets.

## User-provided checkpoint layout

The supplied `dettato.zip` can be extracted as-is. Set the weights root to the extraction directory containing the `dettato` folder; `ModelPaths` recognizes `dettato/detatto-evidence-config.json` and `dettato/detatto-evidence.safetensors` as the Evidence stage without renaming. The large weight files are not tracked by Git.

The original Hugging Face directory format also remains valid:

## Weights layout

```text
weights/
  dettato/config.json
  dettato/model.safetensors
  dettato-evidence/config.json
  dettato-evidence/model.safetensors
  dettato-grouper/config.json
  dettato-grouper/model.safetensors
  dettato-drums/config.json
  dettato-drums/model.safetensors
```

## One-call API

`Dettato::from_directory(root, &device)` loads all four checkpoints. Call `transcribe(&mono_pcm, source_rate)` or `transcribe_file(path)`; the returned `PipelineResult` includes a `Transcription` with `.save_midi(path)` and a separate `PipelineInfo`. This API is implemented in source and awaits compilation and end-to-end validation. `set_merge_same_instrument(false)` preserves independent tracks of the same named instrument.

## Validation and limitations

Original-Python-generated reference fixtures cover 3 rows × 769 evidence features, PCEN/standardisation and HDemix. Python-side PCEN recurrence matched the reference with maximum absolute difference ~5.1e-6 and independent HDemix arithmetic matched within ~1.5e-8; these do **not** certify Rust parity. Rust tests cover independent algorithms (note features, insertion deduplication, candidate decoding, hierarchical clusters, Hungarian matching, instrument naming, drum peaks, MIDI output). Their Rust execution is **not confirmed**.

Recommended validation commands in a Rust 1.100+ environment with accessible Burn dependencies:

```sh
cargo fmt --manifest-path models/dettato/Cargo.toml --check
cargo test --manifest-path models/dettato/Cargo.toml --features flex
```

The authoring container does not have `cargo` or `rustc` and has no outbound DNS. Thus the new source has **not** been compiled or executed as Rust, and all compiler/API issues remain possible. Original Python modules were used to generate the reference evidence fixture. No CUDA benchmark or weight-based inference was attempted.

## Next verification priorities

1. Build with Rust 1.100+, fix all compile issues, and run unit tests.
2. Unpack the user-provided weights and run the ignored four-checkpoint initialization and model fixture tests with `DETTATO_WEIGHTS` set.
3. Compare full-network outputs PyTorch vs Burn and tighten numerical tolerances.
4. Optimize the currently un-fused GRU/attention path only after parity has been established.
