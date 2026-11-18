# Fish Speech 1.5 Rust — CONT6 (2026-11-17)

## Source provenance and checkout

- GitHub connector: `Atm4x/Vendor-for-burn`, `codex/fish-speech`.
- Git Bridge workflow `44126115177`, artifact `14347327522` (`fish-speech-git-bridge`); verified archive SHA256 against the artifact's checksum, verified `repository.bundle` with Git and checked out locally; bridge snapshot HEAD `e89871a01cbb0893528465a60af72612737c5e75`.
- Current GitHub branch observed through connector at `218f920a6f3da70d5f2708a3d77c8b24c75bc592`, a later commit integrating `models/fish-speech`. Local source tree combines the verified bridge base with the user-supplied CONT5 source snapshot and the CONT6 additions. **The bundle itself does not contain 218f920a**.
- No remote GitHub writes were performed for CONT6. This is a source-only local handoff; no edits to legacy/obsolete scripts.

## Implemented in CONT6

1. `src/weights.rs`: strict versioned SafeTensors manifest, SHA256 shard and per-tensor integrity, shard path validation, graph key/shape/dtype checks, strict unexpected-key rejection and cache identity hashing including TRT ABI/device/precision/shape profile. Tests run with synthetic weights; original checkpoints not supplied.
2. `tools/export_checkpoint.py`: offline, trusted PyTorch checkpoint to SafeTensors exporter (PyTorch not shipped for runtime). `torch.load(..., weights_only=True)`, QKV packing, normalization of semantic `model.` and Firefly `generator.` prefixes, weight normalization flattening for Conv and ConvTranspose, explicit provenance hashes and model license input. `tools/test_export_checkpoint.py`: 5 Python CPU tests.
3. `core/src/fsq_cpu.rs`: CPU reference grouped residual FSQ **encode** path using group `project_in`, two-stage FSQ `bound`, round-to-even, residual quantization, and codebook packing. `core/tests/fsq_encode_reference.rs` with four fixtures. Third-party package **v1.14.24** formulas were checked from the dependency's exact reference source; this is a formula fixture, not an end-to-end weighted model.
4. `core/src/firefly_ir.rs`: original Fish Firefly model DAG contracts for reference encode (`LogMel -> ConvNeXt -> FishConv/downsample -> FSQ`) and waveform decode (`FSQ -> FishTransConv/upsample -> HiFiGAN`). Validates temporal lengths, graph input edges, channel shapes and weights, including direct `nn.Conv1d` in ConvNeXt transitions. **156 nodes / 218 weight keys** in reference DAG and **264 nodes / 232 keys** in decoder DAG with stock config. A 1:1 audit against Python's actual instantiated architecture yields **450 source keys = 450 Rust keys; missing 0, extra 0, shape mismatches 0**.
5. `tools/audit_firefly_shapes.py`: reproducible audit against provided Python source. Uses PyTorch meta device and a *shape-only stub* for unavailable `vector_quantize_pytorch`; it establishes key/shape parity, **not numeric parity**.
6. `src/bin/fish-weights-inspect.rs`: CLI to load/check complete SafeTensors manifests and strictly validate Firefly requirement keys/types/shapes, no GPU execution. A synthetic FP32/BF16 checkpoint passed Python exporter -> Rust CLI round-trip.
7. `.gitignore` excludes nested targets/temporary Python artifacts.

## Verification ledger (Linux, Rust 1.100.0, offline)

| Check | Result | Evidence/qualification |
| --- | --- | --- |
| `cargo fmt --all --check` | PASS | CONT6 source |
| `cargo test --locked --offline` (outer crate) | PASS, 3 tests | SafeTensors & manifest CPU tests |
| `cargo test --locked --offline --manifest-path core/Cargo.toml` | PASS, 139 tests | 117 unit + 12 Firefly + 4 FSQ encode + 2 FSQ decode + 4 spectrogram |
| `cargo test --locked --offline --features tensorrt-mock` | PASS | Existing offline vendor `trtx` mock + strict preflight tests; no GPU operations |
| Python exporter tests | PASS, 5 tests | CPUs only |
| Python source weight-shape audit | PASS, 450/450 | PyTorch meta architecture + shape-compatible FSQ scaffold |
| Python checkpoint -> SafeTensors -> Rust loader | PASS, synthetic | Real checkpoint not supplied |
| `cargo check --features tensorrt-api` | **NOT COMPLETED** | C++/FFI compilation did not complete in separate timed runs; no basis for a new pass claim |
| RTX engine build / inference / numeric pretrained parity / audio | **NOT TESTED** | RTX SDK, GPU, checkpoint, tokenizer absent |

The TensorRT mock library's `INetworkDefinition` may be null; structural preflight does **not** exercise real `add_*` layer API. Do not interpret mock PASS as engine/inference parity. `tensorrt-api` and Windows GPU are separate gates.

## Reproduce

Run from `models/fish-speech` in the bundle worktree with offline `vendor/` and the uploaded Rust 1.100.0 executable on `PATH`:

```sh
cargo fmt --all --check
cargo test --locked --offline
cargo test --locked --offline --manifest-path core/Cargo.toml
LIBCLANG_PATH=/usr/local/swift/usr/lib cargo test --locked --offline --features tensorrt-mock
python -m unittest discover -s tools -p 'test_*.py' -v
cargo run --locked --offline --manifest-path core/Cargo.toml --bin fish-proto -- --firefly-weight-shapes > firefly-shapes.tsv
python tools/audit_firefly_shapes.py --source /path/to/Python/fish_speech_lib/fish_speech/models/vqgan/modules --expected firefly-shapes.tsv
```

Do not treat this `firefly-shapes.tsv` as native model weights. See spec `FISH_SPEECH_1_5_RUST_TENSORRT_RTX_IMPLEMENTATION_SPEC_2026-11-17.md` for Windows TRT-RTX acceptance gates.

## Next implementation priorities

1. Find/prepare actual Fish 1.5 `config.json`, `tokenizer.tiktoken`, semantic and Firefly checkpoints under explicit source license. Run Rust strict weight shape validation against converted real checkpoints; resolve dtype discrepancies/weight normalization with source-based tests. Keep originals and obsolete code untouched.
2. Expose Firefly typed DAG as TensorRT-RTX Network Definition lowering, starting with Conv1d/ConvTranspose and explicit padding/crop tests. Preserve inference correctness first; no mock runtime fallback.
3. Implement sampling and reference conditioning end-to-end with CPU numerical golden fixtures, and AR lifecycle + active-prefix Slow/Fast KV cache on typed graph. Keep GPU-side EOS and streaming/cancellation ownership explicit.
4. Real Windows TRT-RTX 1.7 smoke (SDK + RTX GPU required): Rust -> network -> GEMM -> engine -> execution; then layer-by-layer real-weights parity and GPU profiles.
5. Measure warm end-to-end native vs the **same** Fish 1.5 Python `torch.compile(reduce-overhead)` baseline, cold compile/cache and VRAM. Do not assert parity from structural/micro tests.

**Completion status:** Linux source design / CPU reference contracts advanced; production TRT-RTX/native TTS still incomplete.
