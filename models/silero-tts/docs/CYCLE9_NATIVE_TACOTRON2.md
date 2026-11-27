# Cycle 9 — native Tacotron2 (V2)

Base: `d2a37d0db8e297d61a7f0539e4a14ddfe1986ac4`. Rust 1.100.0, release, Burn Flex, offline, four Rayon threads.

The V2 graph is typed Rust, using Burn operations, with shared HiFi-GAN residual and upsample stages. No executable Python/TorchScript interpreter, PyTorch, LibTorch, exported weights or new dependencies in the runtime. `V2Checkpoint` validates the package metadata and shapes; dispatch recognizes package structure and the V2 alphabet convention, then validates the Tacotron topology. Mono V2 exposes the speaker `default` since the serialized package does not name its single speaker. Multispeaker V2 uses the checkpoint's speaker IDs.

The encoder uses three Conv1d/BatchNorm/ReLU stages and two LSTM directions with preprojected input. Decoder uses prenet with inference-time Bernoulli dropout, two LSTM cells and location-sensitive cumulative attention, gate termination, and five-layer postnet. The V2 HiFi-GAN uses the existing residual blocks and a strided pre-convolution for low-rate output. The output is cropped to the original mel-derived sample length. Package ZIP parsing explicitly uses offset 0 (the official package starts at byte 0); this avoids the erroneous inferred archive offset on original `mykyta_v2.pt`. Non-package reader behavior is unchanged.

## Acceptance

- Offline locked release build: PASS.
- Unit tests: 60 passed, 2 ignored (external fixtures). Other integration tests: 3 passed, 3 ignored (external fixtures).
- `original_v2_fixed_mask_parity`: PASS, 5 cases, 11 full PCM and 11 isolated vocoder comparisons. Maximum absolute errors: encoder 9.46e-7, mel 1.375198e-3, PCM 2.215996e-3, isolated vocoder 2.6152e-5. Worst PCM SNR 58.985 dB. Fixed-mask full PCM tolerance 3e-3 was set before running the test. Frame counts and decoder gate termination matched. This is not a claim of bit-exact independent RNG sequences.
- Full native IPC smoke matrix: 31/31 original packages at 8 kHz, including all 28 earlier models. Extra V2 IPC tests at lj/multi 16 kHz and mykyta 24/48 kHz: 4/4. Real V2 inference remains stochastic; the smoke test checks both WAVs independently rather than assuming equal hashes.
- `lj_v2`, `multi_v2`, `mykyta_v2` added to verified catalogue, preserving original checkpoint URL, length, SHA-256. All other V2 models remain unverified. `aidar_v2` and `ru_v3` require parsing QInt8Storage / packed dynamic quantized LSTM parameters and porting their recurrent accentor; not claimed supported. Remaining V5 JIT-labelled package and standalone JIT models are not implemented. Windows acceptance requires its own CI result.

Reproduce oracle (development only):

```
python models/silero-tts/tools/create_v2_parity_fixture.py CHECKPOINT_DIR FIXTURE_DIR
SILERO_V2_CHECKPOINTS=CHECKPOINT_DIR SILERO_V2_FIXTURES=FIXTURE_DIR RAYON_NUM_THREADS=4 cargo test --offline --release --locked --manifest-path models/silero-tts/Cargo.toml --no-default-features --features reference-flex --lib original_v2_fixed_mask_parity -- --ignored --nocapture
```

The oracle captures the exact dropout mask by restoring the RNG state around original scripted prenet calls, then uses the original encoder, scripted decoder step, postnet and vocoder. Runtime never reads this fixture.
