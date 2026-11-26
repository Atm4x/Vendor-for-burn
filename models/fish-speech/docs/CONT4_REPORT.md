# Fish Speech 1.5 native Rust prototype — CONT4

Date: 2026-11-17. Scope: **CPU numerical implementation only**. No real TensorRT-RTX, GPU, model checkpoints or packaged inference runtime. Original Fish Python sources and legacy scripts were not modified.

## Progress: Linux architectural prototype 54% (engineering estimate)

CONT3: 45%, 97/97 PASS. CONT4: **132/132 PASS** (114 core unit tests, 12 Firefly/PyTorch numerical fixtures, 4 log-mel/PyTorch fixtures, 2 FSQ formula fixtures). `cargo test --offline --release`, `cargo fmt --check`, and CLI `cargo run --offline --release` pass with Rust 1.100.0.

The 54% figure is a planning estimate of the Linux prototype only, **not** accuracy, end-to-end TTS progress, TRT readiness, performance or percentage of model weights loaded.

## What is newly executable

1. `core/src/firefly_cpu.rs`: validated NCT tensors, FishConvNet Conv1d (constant left/right padding, groups, dilation), FishTransConvNet ConvTranspose1d (correct time trimming), channels-first LayerNorm, ConvNeXt block (depthwise + LN + Linear/GELU/Linear + gamma + residual), HiFiGAN ResBlock1, ParallelBlock and a composable small HiFiGAN decoder.
2. `core/src/spectrogram_cpu.rs`: no-dependency CPU radix-2 FFT, original reflection padding, periodic Hann window, STFT magnitude `sqrt(real²+imag²+1e-6)`, Slaney mel filterbank and log mel clamp; **44.1 kHz mono input only**, no resampler/WAV decoder.
3. `core/src/fsq_cpu.rs`: mixed-radix `[8,5,5,5]` FSQ index decoder (1000 codes), grouped residual codebook decoding, q-index scales and per-group learned project-out matrix/bias. Layout `[B, groups * quantizers, frames]` → `[B, group_width * groups, frames]`.
4. `core/src/audio.rs`: distinction between allocated quantized frames (`ceil(mel_frames/4)`) and valid Python `feature_lengths` (`floor(mel_frames/4)`) plus codebook decode masks and invalid reflect-pad lengths.

## Source verification

Original user ZIP:
- `fish_speech_lib/fish_speech/models/vqgan/modules/firefly.py`: FishConvNet, FishTransConvNet, ConvNeXtBlock, HiFiGANGenerator and ResBlock1.
- `fish_speech_lib/fish_speech/utils/spectrogram.py`: LinearSpectrogram and LogMelSpectrogram.
- `fish_speech_lib/fish_speech/models/vqgan/modules/fsq.py`: DownsampleFiniteScalarQuantize and GroupedResidualFSQ usage.
- `fish_speech_lib/config.py`: configured 160 mel bins, 44,100 Hz, FFT=2048, hop=512, FSQ levels `[8,5,5,5]`, eight groups, quantizer count one.

Third-party reference formulas were inspected at EXACT tag 1.14.24, not repository main:
- https://github.com/lucidrains/vector-quantize-pytorch/blob/1.14.24/vector_quantize_pytorch/finite_scalar_quantization.py
- https://github.com/lucidrains/vector-quantize-pytorch/blob/1.14.24/vector_quantize_pytorch/residual_fsq.py

## Evidence and limitations

### Tested against actually executed original Python modules

- `core/fixtures/generate_firefly_golden.py` imports the unmodified `firefly.py`; 12 tests: regular/grouped/dilated/short Conv1d; standard/upsampling/dilated ConvTranspose1d; ConvNeXt including very short sequences; ResBlock1; **complete tiny HiFiGAN decoder** with two upsample stages and three parallel ResBlock1 branches per stage. All 12 PASS at FP32 tolerance typically `atol=2–5e-6`, `rtol=1–2e-5`.
- `core/fixtures/generate_spectrogram_golden.py` imports the unmodified `spectrogram.py`; compares 769-, 1024-, 3073- and 4096-sample reference waveforms. All four PASS at absolute max tolerance < 0.001 log-mel units across all 160 bins and frames.

### FSQ tests are weaker evidence

- The exact 1.14.24 source equations were inspected. `core/fixtures/generate_fsq_golden.py` generates independent Torch reference values **from those equations** with synthetic project-out matrices (not from an imported, executed `GroupedResidualFSQ` class). Two fixtures PASS, plus tests of 1000/1000 index-digit round-trips and invalid indices. **Original library execution parity is still untested**. No real FSQ learned `project_in` or Firefly checkpoint has been loaded.

### Important correctness finding

In the unmodified Python Firefly `encode()`, `feature_lengths = (audio_lengths // 512) // 4` uses floor division, while two stride-2 FishConv layers allocate `ceil(mel_frames/4)` physical frames. `FireflyFrameLengths` distinguishes the two to avoid incorrectly exposing padded tail as valid reference content; `FireflyDecodeLengths` gives both physical and masked sample lengths. For raw input length <=768 samples, original reflect-pad frontend fails, and Rust returns an explicit error.

## What did NOT happen

- No full Fish 1.5 inference; AR graphs continue to run only with synthetic weights.
- No exact Fish tokenizer or checkpoint config, no real weights, no numerical model logits, no speech synthesised from an actual model.
- No actual FSQ upstream class parity or real VQGAN full encode/decode with checkpoint weights; the numerical building blocks do not constitute a runnable model.
- No `trtx` FFI compilation: offline registry still lacks `autocxx 0.30.0` and transitive crates. TRT mock not tested.
- No Windows GPU/TensorRT engine, CUDA graphs, VRAM/latency/RTF, no distribution size figures.

## Next work

1. Add a **connected typed Firefly graph** with explicit metadata and checkpoint weight keys for every backbone/FSQ/HiFiGAN stage. It must build on numerical kernels and surface missing weights explicitly.
2. Integrate learned grouped FSQ `project_in` and compare `encode`/`decode` against the true v1.14.24 package, not just independently reconstructed formulas; use small PyTorch fixtures.
3. Add actual checkpoint JSON/tokenizer fixtures when available; avoid inventing model dimensions and token IDs.
4. Separate reference-audio decoding/resampling contract and test parity with torchaudio for realistic sample rates.
5. The trtx mock step needs an offline vendor closure for `autocxx` and transitive crates. No real SDK/GPU required yet. Do not bypass this by reporting structural plans as TRT execution.

Use `cargo test --offline` in `core/`. Full handoff: `NEXT_STEPS.md`.
