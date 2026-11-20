# Dettato — Rust + Burn port

Independent native Rust implementation of the [Dettato](https://github.com/kobimusic/dettato) multi-instrument Audio-to-MIDI pipeline. Original project by Nathan Mills / kobimusic, licensed under Apache-2.0. The original license and notice are retained in this directory.

**Status (cycle 08):** all four learned stages and the end-to-end API are present in Rust. Rust 1.100.0 / Burn 0.22.1 builds and the standard tests pass on Flex CPU; all four released checkpoints load. Selected neural outputs of **each of the four stages** have been compared with the original PyTorch models. **A complete recording-to-MIDI run is not yet verified**; neither eight-second Transcriber performance nor GPU parity is established. See `CYCLE06_REPORT.md`, `CYCLE07_REPORT.md`, and `CYCLE08_REPORT.md`.

This is a separate crate at `models/dettato`. No source changes to `models/beat_this` are required. Model inference uses Burn 0.22.1 tensor operators dispatched to **CubeCL** for CUDA/WGPU/CubeCL CPU, or to Flex for a CPU-only reference path. This is a Burn-on-CubeCL implementation, not a new kernel-by-kernel rewrite of all layers in raw CubeCL. No Python or ONNX runtime is required.

## Implemented stages

| Stage | Implemented components | Verification status |
| --- | --- | --- |
| Transcriber | HCQT, Stem, 15-block LaneNet, sparse pitch mixing, axial attention, RoPE, bidirectional GRU, six probability maps, 8-second overlapping windows, two-voice decoding | Real checkpoint loaded; 0.2-second PyTorch/Burn forward parity PASS for all six heads; 8-second Flex CPU forward unconfirmed |
| Evidence | 769-component features, FP16/quantized hidden representations, 9-layer Transformer with RoPE/SwiGLU, note keep/adjust/insert decoding | Real checkpoint forward PASS, selected logits match PyTorch to 4.77e-6 |
| Grouper | Extended HCQT, PCEN, HDemix, Prof, LaneFront, 7-layer NoteNet, 73 instruments, 32-D embeddings, TrackHead, clustering, Hungarian matching and separate MIDI tracks | Real checkpoint forward PASS; probabilities/embeddings/keep/hidden PyTorch comparison PASS (max 1.765e-3) |
| Drums | 44.1 kHz frontend, drum-kit network, router, three independent orchestral experts, ensemble, 61 GM drum notes, 13 timpani pitches and velocity | Real checkpoint forward PASS; selected PyTorch outputs match within 3.10e-4 |

The Rust code also covers retuning, multi-format audio decoding, time conversion, MIDI Type-1 export and `merge_same_instrument=false` for keeping distinct instances of the same instrument in separate tracks. The complete flow is **implemented, not certified** by a real full-song test. Cycle 07 adds a shared Transcriber pass with fine/coarse Hann averages and a configurable RAM budget: the three tap maps are cached for one-pass Evidence/Grouper inference when the budget allows; otherwise the prior low-memory multi-pass path remains. This removes redundant learned-network passes without changing model weights.

## Weights

Unpack the released safetensors separately; they are not part of Git. Both the official per-model directories and the user's flat ZIP layout are supported. For the flat layout:

```text
weights/
  dettato/
    dettato-config.json
    dettato.safetensors
    detatto-evidence-config.json
    detatto-evidence.safetensors
    dettato-grouper-config.json
    dettato-grouper.safetensors
    dettato-drums-config.json
    dettato-drums.safetensors
```

The `detatto-evidence` spelling is intentional: the loader accepts the original misspelling.

## API

`Dettato::from_directory(root, &device)` initializes the four networks. `Dettato::transcribe(&mono_pcm, sample_rate)` or `transcribe_file(path)` returns a `PipelineResult` containing a `Transcription`. `Transcription::save_midi(path)` exports separate MIDI tracks and creates parent directories. Call `set_merge_same_instrument(false)` to retain distinct same-instrument groups. Call `pipeline.transcriber.set_shared_cache_budget_bytes(bytes)` to control RAM allocated for the shared Transcriber pass (default 256 MiB; zero always uses the low-memory fallback). Pitched stages work on retuned mono 16 kHz; the 44.1 kHz drums stage runs only when the input recording has a source sample rate of at least 40 kHz.

## Command-line executable

The crate also provides a `dettato` CLI. Example, with a flat extracted weights directory:

```sh
cargo run --release --no-default-features --features wgpu -- --weights /path/to/weights/dettato --input song.wav --output song.mid --device wgpu:0 --separate-instruments --cache-mib 4096
```

`--output` defaults to the input name with `.mid` extension. `--separate-instruments` disables merging tracks with the same instrument label. `--cache-mib` changes the RAM cache budget (default 256 MiB); the larger budget lets longer songs share one Transcriber pass at the expense of RAM. `--device` selects the **actual backend**, not just a label: `auto`, `cuda`, `cuda:N`, `wgpu`, `wgpu:N` (discrete GPU index), `cpu` (CubeCL/LLVM), or `flex`. An explicit backend that was not compiled into the binary produces a clear error rather than silently falling back. `auto` selects Burn's highest-priority compiled backend (CUDA before WGPU/CPU/Flex). `--backend` is accepted as an alias. The CLI does not ship CUDA driver DLLs, the Tracel LLVM bundle, or model weights.

## CubeCL GPU builds

CUDA and WGPU both run the **same learned networks**, without ONNX conversion or network simplifications. The `cuda`, `wgpu`, and `cpu` Cargo features turn on Burn's CubeCL runtime, kernel fusion, and autotuning; `flex` remains a pure-Rust CPU reference backend.

```sh
cd models/dettato
cargo build --release --no-default-features --features cuda
./target/release/dettato --weights /path/to/weights/dettato --input song.wav --device cuda:0

cargo build --release --no-default-features --features wgpu
./target/release/dettato --weights /path/to/weights/dettato --input song.wav --device wgpu:0

cargo build --release --no-default-features --features 'cuda,wgpu,flex'
./target/release/dettato --weights /path/to/weights/dettato --input song.wav --device cuda:0
```

`wgpu` without an index uses adapter auto-selection and can select a software adapter on some systems; use `wgpu:0` to require the first **discrete** adapter. CUDA needs a working NVIDIA driver. CubeCL CPU needs the Tracel LLVM runtime. The offline build in cycle 08 verified WGPU compilation, **not** GPU execution. Cycle 08 CUDA compilation stopped in `tracel-llvm-bundler` before this archive became available. Cycle 09 verified `cargo check --no-default-features --features cuda` with LLVM 23.1.0-4 installed locally. This is compilation only; no NVIDIA GPU is exposed in the test environment.

## Offline Tracel LLVM for CubeCL (Linux)

CUDA and CubeCL CPU builds with Burn 0.22.1 require the `tracel-llvm-bundler` release **23.1.0-4**. If the build machine has no network access, provide the matching `linux-x64.tar.xz` obtained from a trusted source and install it into the location expected by the bundler:

```sh
bash models/dettato/tools/install-tracel-llvm-linux.sh /path/to/linux-x64.tar.xz
export LLVM_SYS_231_PREFIX="${XDG_DATA_HOME:-$HOME/.local/share}/tracel/tracel-llvm-23.1.0-4"
cd models/dettato
cargo check --no-default-features --features cuda
cargo test --no-default-features --features cpu --lib
```

The script validates the archive directory and `llvm-config` version and checks that NVPTX support is present. It marks the existing locally supplied bundle as installed so `tracel-llvm-bundler` will not fetch it. It does not download LLVM or ship LLVM binaries. The marker is a **trusted, local offline bypass** of the bundler's online SHA-256 manifest verification; only use it with a trusted LLVM archive. A separate, complete Cargo vendor snapshot is still required for `--offline` builds (the earlier snapshot needed an external resolver overlay). To avoid huge DWARF-heavy integration-test links against static LLVM, this crate sets `profile.test.debug = 0`.

## Transcriber device-operation reductions (cycle 08)

`PitchMix` now encodes sparse pitch offsets as a **single depthwise Conv2d** (groups = channels) along the pitch axis instead of repeated slice/concat/zero/multiply/add operations; `AxialAttention` caches relative pitch bias and RoPE sine/cosine tables on its initialized device. For unexpected sequence lengths over 512, RoPE regenerates tables on the current device. These operations execute through CubeCL on CUDA/WGPU, with no repeated CPU tensor upload for standard 8-second windows. Original model weights and output shapes are preserved. Numeric short-forward parity is tested separately; these changes have **not** been benchmarked on an actual GPU.

## Build and tests

```sh
cd models/dettato
cargo test --no-default-features --features flex
export DETTATO_WEIGHTS=/path/to/weights/dettato
cargo test --no-default-features --features flex --test transcriber_forward_parity -- --ignored --nocapture
cargo test --no-default-features --features flex --test grouper_forward_parity -- --ignored --nocapture
cargo test --no-default-features --features flex --test evidence_forward_parity -- --ignored --nocapture
```

The full default feature set also enables MP3/AAC/ISO-MP4/ALAC decoders through Symphonia. Those optional dependencies were unavailable in the local offline vendor snapshot, so all current local measurements use `--no-default-features --features flex` and an external Cargo source overlay. They do not establish that every optional codec builds offline.

The `transcriber_forward_reference.json` and `grouper_forward_reference.json` contain output values generated by the **original PyTorch networks and released weights**, not inferred fixture values. All real-weight tests are marked ignored by default, so the ordinary test suite can run without downloading checkpoints.

## Remaining work

Cycle 08 additionally supports explicit CubeCL device selection and optimized Transcriber kernels/table reuse. It includes an ignored `full_pipeline_smoke` that exercises the four-stage entrypoint with real weights; it has not been accepted as a completed full-song inference test. Exact numerical parity of the new shared tap buffers remains unverified against the PyTorch full-song accumulation (the stand-alone fine/coarse Hann tests do pass).

The eight-second Transcriber Flex CPU smoke still times out after 360 s; optimized-backend completion and full-song MIDI verification, broader numerical comparisons, optimized release/GPU benchmarks, DSP resampling parity with upstream `soxr HQ`, and optional audio decoder/packaging validation. The unoptimized Flex CPU test profile takes approximately 41 seconds to execute a 0.2-second Transcriber input; it must **not** be interpreted as the expected release or CUDA throughput.
