# Cycle 11 — standalone checkpoints

Base: a7bf8caa8dede56d3f10b47dece1dd80e4b08c84. Rust 1.100, offline Flex release.

- Standalone reader decodes `data.pkl` and tensor storages; it never interprets serialized code. The official V1 tokenset and rate are selected by the original artifact digest, not by filename. Unknown standalone files are explicitly rejected because vocabulary and stride metadata are not stored in the checkpoint tensor state.
- Twenty V1 variants share the V2 Tacotron2 implementation with a fullband HiFi-GAN topology and an 8/16 kHz-specific input stride. Warmup now selects a supported rate.
- V5 CIS JIT shares the V5 acoustic layers and iSTFT vocoder, but uses the original standalone duration semantics and fullband synthesis. The official companion package proves the same three symbol embeddings and an exact unique 60-row speaker permutation across all three speaker embeddings. `create_v5_jit_frontend.py` reproduces this binding. Only metadata is embedded; weights are still read from the original standalone archive. The standalone graph ignores its sample-rate argument and produces 48 kHz audio, so only 48 kHz is advertised.
- Original catalogue metadata is preserved, even where it differs from the actual supported rates. The verified manifest supplies the artifact digest and smoke configuration. No external runtime or downloaded dependency is added; serde and sha2 were already vendored.

## Validation

Offline release build passed. Unit tests: 61 passed, 3 ignored. Complete IPC smoke on original files: 66/66 (load, warmup, two nonempty WAVs, checksum, unload, shutdown). Stochastic V1/V2 do not require identical repeated WAVs.

V1 short fixed-mask oracle: 20/20 passed (encoder max 9.09e-7, mel max 6.31e-4, PCM max 5.16e-4). Longer catalogue examples expose recurrent numerical amplification: only two aidar variants passed strict combined thresholds, while the remaining variants exceeded at least one threshold. Decoder gate/frame count agreed on the original long fixtures. Do not conflate basic IPC coverage with full numerical parity. The fixed-mask test still uses the original thresholds (encoder 2e-4, mel/PCM 3e-3, isolated vocoder 1e-3), now reporting independent stage metrics before failing. Recurrent accentor logits from cycle 10 remain an open numerical defect.

V5 CIS JIT oracle: Russian, Armenian and Azerbaijani 3/3, maximum absolute FP32 PCM errors 4.405e-5, 3.670e-5, 3.082e-4 respectively. Test loads a renamed file and independently obtains token IDs from the official companion frontend. Original graph produces identical output for 8/24 kHz arguments. Runtime supports ordinary text, not all optional original features (SSML, explicit emphasis, random voice, etc.).

Reproduction: use `tools/create_v2_parity_fixture.py --standalone` (and `--short` for short inputs), set `SILERO_V1_MODELS`, `SILERO_V2_CHECKPOINTS`, `SILERO_V2_FIXTURES`, run ignored `original_v2_fixed_mask_parity`; `tools/verify_v5_jit_parity.py BINARY COMPANION_PACKAGE JIT`; `tools/smoke_native_worker.py` for IPC. Python/PyTorch are development oracles only.

Complete long rerun stage maxima: memory=1.408e-06 (20 cases), mel=0.416591644 (20 cases), vocoder=2.2396e-05 (20 cases), pcm=0.25694114 (20 cases).
