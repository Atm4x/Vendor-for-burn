# Dettato — native Rust / Burn port

**Status: partial implementation. Two of the four neural stages (Transcriber and Evidence Repairer) have source-level Burn implementations, but have not passed Rust compilation, checkpoint loading, layer parity or end-to-end numerical verification. Grouper and Drums currently provide their decoding/postprocessing algorithms, not their neural networks. The full system cannot yet perform independent end-to-end transcription.**

Original Dettato: Nathan Mills / kobimusic, Apache-2.0, <https://github.com/kobimusic/dettato>. See `LICENSE` and `NOTICE`.

This is an independent crate in `models/dettato`. `models/beat_this` is unchanged. It uses Burn 0.22.1 and can be configured for Flex, CPU, CUDA and WGPU backends where the upstream Burn kernels support the operations. There is no ONNX or Python dependency at runtime.

## Covered source features

| Component | Rust code currently present | Still missing |
|---|---|---|
| Checkpoints | `config.json`, named `safetensors` tensors for all four stages, F32/F16/BF16/F64 import | Automated full-key/shape parity on released checkpoints |
| Transcriber | HCQT frontend, Stem, PitchMix, CNN backbone, pitch/time attention, RoPE, bidirectional GRU, six heads, 8 s Hann overlap, two-slot note decoder | Numerically verified layer outputs and performance optimizations |
| Evidence | 31 map features, two 368-d hidden summaries, standardization/1/32 quantization, FP16 map feature rounding; CPU note context and beat-grid features; Burn embeddings/9-layer note transformer/RoPE/SwiGLU/heads; candidate decoding; keep/drop/insert repair | Compilation/parity of Burn Evidence network, batched/precision-optimized inference |
| Grouper | 73-class General MIDI vocabulary, cosine average-linkage clustering, Hungarian overlap matching, tracklet merge/refinement/absorption, naming using instrument-family credit and track-head layers, `merge=false`, MIDI track construction | Frontend/NoteNet tensors and neural inference, retune frontend |
| Drums | 50 fps peak selection, kit/orchestral routing, 61 drum keys, 13 timpani notes, 40 ms dedup, velocity | 44.1 kHz frontend, DrumNet, expert models, router inference |
| MIDI | Type-1 multitrack export, programs, dedicated drum channel and MIDI ports for additional melodic tracks | Full end-to-end Dettato inference orchestration |

The Transcriber can now collect Evidence hidden vectors in a bounded-memory **second inference pass**, without storing the entire song's `[frames,88,368]` hidden tensor. It accumulates only notes' onset and mean vectors, with the same per-frame Hann-normalized weights used for the output maps. The first pass still provides the shared note maps.

The Rust EvidenceNetwork accepts 769-component evidence vectors and uses the original checkpoint key layout (`emb.*`, `bb.blocks.*`, `keep`, `don`, `dur`, `ins.*`). Its CPU-side continuous feature preparation must be numerically compared with PyTorch before the port is considered complete.

The Grouper's decoder accepts predictions emitted by the *future* Burn NoteNet. It does **not** infer instruments from audio on its own yet. `merge_same_instrument=false` preserves independent same-instrument tracklets in the output.

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

## Validation and limitations

An original-Python-generated numerical fixture of 3 rows × 769 evidence features is included in `tests/fixtures/evidence_reference.json`. Rust tests cover independent algorithms (note features, insertion deduplication, candidate decoding, hierarchical clusters, Hungarian matching, instrument naming, drum peaks, MIDI output). Their Rust execution is **not confirmed**.

Recommended validation commands in a Rust 1.100+ environment with accessible Burn dependencies:

```sh
cargo fmt --manifest-path models/dettato/Cargo.toml --check
cargo test --manifest-path models/dettato/Cargo.toml --features flex
```

The authoring container does not have `cargo` or `rustc` and has no outbound DNS. Thus the new source has **not** been compiled or executed as Rust, and all compiler/API issues remain possible. Original Python modules were used to generate the reference evidence fixture. No CUDA benchmark or weight-based inference was attempted.

## Next functional priorities

1. Port the Grouper `Front`/`HDemix`/`Prof`/`LaneFront`/`NoteNet` and gather the Transcriber `stem` taps required for its input.
2. Port `drums/features.py`, DrumNet, OrchestralExpert and three-model averaging/router inference.
3. Port the retuning/resampling stage and the single-call end-to-end pipeline.
4. Verify all checkpoint names/dimensions and run local per-layer numerical parity (Transcriber, Evidence, Grouper, Drums), then address runtime and GPU performance.
