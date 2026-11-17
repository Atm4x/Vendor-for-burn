# Beat This! — native Rust inference

This independent Rust crate implements inference for [Beat This!](https://github.com/CPJKU/beat_this), including direct **PyTorch Lightning `.ckpt`** loading, stereo-to-mono decoding, resampling, log-mel feature extraction, CNN and PartialFT frontend, RoFormer transformer, logits aggregation, and beat/downbeat extraction. **Python, PyTorch, ONNX, and LibTorch are not needed at runtime.**

The model uses the Burn fork in this repository. The checkpoint is read directly with the vendored `pytorch-reader` implementation, rather than requiring a separate conversion step.

## Build and run

From `models/beat_this/` in this repository, using Rust 1.100.0:

```sh
cargo test --offline
cargo build --release --offline
./target/release/beat-this-rs --checkpoint /path/to/small1.ckpt \
  --input /path/to/music.flac --output /path/to/music.beats
```

Use `final1.ckpt` for the full-size model. The command line can decode FLAC and uncompressed WAV; the vendored Symphonia features also enable OGG/Vorbis. The output is a standard tab-separated `.beats` file: `<seconds>\t<beat number>`, with 1 denoting a downbeat.

To compile a CUDA-capable binary (requires the GPU/CubeCL LLVM toolchain and a suitable CUDA setup):

```sh
cargo build --release --offline --no-default-features --features cuda
./target/release/beat-this-rs --device cuda --checkpoint final1.ckpt --input music.flac
```

Other optional backend: `--no-default-features --features cpu` and `--device cpu`. The default build uses Burn's `flex` backend (`--device flex`) and works on the CPU. Features and device selection have to agree.

**High-fidelity resampling:** the reference preprocessing uses SoXR. On Linux, install `libsoxr.so.0`; on Windows, provide the SoXR DLL in the runtime DLL search path. The resampler is loaded dynamically, so the binary itself does not link to Python or PyTorch. By default, the engine uses SoXR if available, otherwise prints a warning and uses the pure-Rust Rubato fallback. The latter has **different numerical behavior** and may change beat predictions. Set `BEAT_THIS_RESAMPLER=soxr` to **require** SoXR or `BEAT_THIS_RESAMPLER=rubato` to explicitly choose the fallback.

To inspect or validate the network, add `--logits-json logits.json`, and optionally `--save-spect-json mel.json`. Supplying `--spect-json mel.json` skips audio decoding and uses the exact provided feature matrix. `--limit-seconds 6` tests a short excerpt without changing model weights. `--help` lists available options.

## Model topology

- Mono PCM → 22050 Hz → Hann STFT (`n_fft=1024`, hop 441, centered with reflect-padding) → 128 triangular Slaney Mel bins (30–11000 Hz) → `log1p(1000 × mel)`.
- CNN stem and three frequency/time PartialFT blocks, implemented with Burn Conv2D and attention/FFN operations; inference-time BatchNorm is folded into scale and bias.
- Six RoFormer blocks using pairwise rotary embeddings, 32-dimensional heads, RMS-normalized attention, sigmoid gates, and feed-forward layers. Channel width depends on the checkpoint: `small1` = 128; `final1` = 512.
- Overlapping 1500-frame windows with a 6-frame guard border; identical `keep_first` merge policy. Minimal postprocessor: local-max beats/downbeats with downbeat snapping, and TSV export with estimated beat numbers.

This crate does **not** implement training, the optional madmom/DBN postprocessor, or other proprietary inference providers. It does not claim those features.

The RVC experiments in this repository include specialized NCT↔NTC transport for **1D transposed convolutions**. Beat This!'s frontend uses **Conv2D**, so its convolution path deliberately uses the vendored Burn Conv2D dispatch instead of forcing an unrelated ConvTranspose1D layout/kernel.

## Validation and limitations

See [VALIDATION.md](VALIDATION.md) for PyTorch-versus-Rust logit comparisons with the supplied `small1.ckpt`, `final1.ckpt`, and music sample. The original PyTorch implementation and native Rust match across the **entire 209-second provided FLAC** (10,458 frames) on both `small1.ckpt` and `final1.ckpt`. Mean logit errors are approximately 1e-5 on long-form input. Both models also reproduce the original beat/downbeat counts on that file. The CUDA feature compiled and linked successfully offline after installing the locally provided Tracel LLVM 23.1.0-4 archive; runtime execution and profiling on an NVIDIA GPU remain **unverified**. The CPU backend can be slower than optimized multi-threaded PyTorch, and real-time performance and GPU speedups are **not established**.

## Licensing

Beat This!'s original implementation is MIT-licensed. The corresponding upstream license is copied into [LICENSE](LICENSE). The vendored libraries are subject to their respective licenses; downloaded checkpoint licensing must be checked independently for redistribution.
