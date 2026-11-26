# Fish Speech 1.5 — functional parity ledger, CONT15

**Baseline:** CONT14 source-only ZIP + attached modified Python Fish Speech 1.5. No newer Fish Speech (S2) is used. Work performed locally with Rust 1.100.0 and pre-existing offline Cargo vendor. CUDA/Windows checks do **not** affect the functional percentage.

**Definition of progress:** 22 equally weighted source behaviors, `DONE=1`, `PARTIAL=0.5`, `MISSING=0`. `DONE` means the behavior is represented by working Rust code with synthetic CPU tests; it does **not** prove full-model numerical parity, fast execution, real checkpoint compatibility or successful GPU execution. `PARTIAL` means a source behavior or runtime implementation is still missing. This is an accounting heuristic, not a percentage of engineering time.

| ID | Original Python function/contract | Status | Exact remaining implementation |
|---|---|---|---|
| F01 | Semantic/Firefly checkpoint configuration and `model_type` | PARTIAL | `config.json` Semantic DualAR/Naive is parsed, but full Firefly config, model-specific aliases and initialization from real checkpoint files are not jointly resolved |
| F02 | SafeTensors conversion/loading | DONE | Manifest SHA-256 validation, F32/F16/BF16 → CPU F32 bridge implemented; authentic model test is a **verification gate**, not functional incompleteness |
| F03 | Fish tokenizer/BPE | DONE | Fish regex, byte-pair ranks and 1036 special tokens implemented; authentic tokenizer parity remains in verification |
| F04 | `clean_text()` | DONE | Rust port with fixtures |
| F05 | `split_text()` | DONE | Rust port with original Python golden cases |
| F06 | `Conversation` serialization / VQ | DONE | Rust implementation |
| F07 | `encode_tokens()` reference/no-reference | DONE | Rust prompt encoding |
| F08 | `ReferenceLoader.load_audio()` | PARTIAL | WAV, FLAC, OGG, bytes/path, mono and caching work; MP3, AAC/M4A and WMA decoding are absent from the offline vendor |
| F09 | Resampler + LogMel preprocessing | DONE | 44.1 kHz torchaudio-like sinc-Hann resampling + CPU LogMel boundary |
| F10 | Weighted Firefly `ConvNeXt + FSQ encode` | PARTIAL | Structural TRT graph and individual CPU operations exist, but no unified **weighted** executable reference encoder bound into TTS |
| F11 | Weighted Slow AR `prefill` | PARTIAL | **CONT15:** real CPU Graph IR execution with bound weight tensors implemented; native TensorRT engine path, complete checkpoint-level mapping and production backend lifecycle still not implemented |
| F12 | Weighted Slow AR `cached decode` | PARTIAL | **CONT15:** stateful CPU decode reuses BF16 KV-cache, active-prefix attention and Python-compatible RoPE coefficients; device execution and checkpoint-level parity remain unfinished |
| F13 | Dual-AR Fast acoustic codebook loop | PARTIAL | **CONT15:** CPU loop now primes Fast at step 0, derives acoustic code 0 from semantic ID, samples subsequent codebooks, and resets Fast KV at every Slow step; GPU stream execution/sampling still require implementation |
| F14 | `NaiveTransformer` | PARTIAL | **CONT15:** new CPU Naive weighted head and independent codebook sampling path; production TensorRT lowering/execution, complete e2e integration and authentic checkpoint parity remain |
| F15 | Repetition penalty → top-p → temperature → Exp(1); EOS | DONE | **CONT15:** CPU sampling connected to both weighted generators and EOS handling; GPU-side sampling is a distinct missing optimized backend implementation |
| F16 | `generate_long()` | DONE | **CONT15:** existing chunk planner now has callable weighted SemanticGenerator adapters for both model kinds; model parity and acoustic backend unfinished |
| F17 | Weighted Firefly `FSQ decode → HiFiGAN` | PARTIAL | Structural 264-node decoder IR and FSQ CPU math, but no complete weighted waveform executor yielding genuine speech |
| F18 | PCM concatenation, 44.1 kHz | DONE | Rust `synthesize` |
| F19 | Complete TTS synthesis | PARTIAL | **CONT15:** weighted *semantic* generator wired to `synthesize()`; acoustic decoder still not the full weighted Firefly; cannot yet produce genuine model speech |
| F20 | `header/segment/final/error` streaming events | DONE | Callback stream implemented; actual low-latency model streaming needs GPU lifecycle work |
| F21 | Reference cache by ID/audio SHA-256 | DONE | Acoustic token reuse with per-request captions; cache on/off/clear |
| F22 | Cancellation, model lifecycle, queues | PARTIAL | Segment cancellation exists; per-token cancellation, warm/unload/model switch, GPU allocation lifetime and request queue remain |

**CONT14:** 10 DONE, 11 PARTIAL, 1 MISSING → (10 + 11×0.5)/22 = **70.45%**.

**CONT15:** 12 DONE, 10 PARTIAL, 0 MISSING → (12 + 10×0.5)/22 = **77.27%**.

`F15` and `F16` advanced to DONE and `F14` from MISSING to PARTIAL; `F11–F13` remain PARTIAL because the complete production execution and authentic model integration are still incomplete, despite substantial runnable CPU functionality. No CUDA/Windows **test** was included in this calculation.

## Separate operability/acceptance ledger

| Gate | Result at CONT15 |
|---|---|
| V01 — Core CPU unit + integration tests | **PASS**: 181 unit + 27 integration = **208 tests** |
| V02 — outer Rust crate unit tests | **PASS: 18 tests** |
| V03 — offline Python checkpoint exporter | **PASS: 5 tests** |
| V04 — text Python golden parity | Prior 144-case PASS, not separately rerun |
| V05 — Rust→`trtx` 1.7 API compile | **PASS**: `cargo check --locked --offline --features tensorrt-api` |
| V06 — `trtx` mock integration | **INCOMPLETE**: C++/`trtx-sys` build exceeded test timeout; not PASS and not FAIL |
| V07 — authentic checkpoint and tokenizer parity | BLOCKED: original checkpoint/tokenizer assets absent |
| V08 — TensorRT-RTX engine build/inference on RTX | NOT TESTED, no GPU/SDK |
| V09 — genuine speech output/quality | NOT TESTED, weighted Firefly incomplete |
| V10 — Windows EXE/package/profiling | NOT TESTED |
| V11 — reference audio CPU contracts | Prior WAV/FLAC/OGG/resampling PASS, component tests pass in regression |

**CUDA/Windows-only acceptance: 0/3** (real engine, end-to-end voice, Windows distribution). Independent of the 77.27% functional score.

## Next unclosed original behaviors, implementation sequence

1. **F10/F17/F19:** implement a unified weighted Firefly encoder and decoder and bind it to `AcousticDecoder` + reference `ReferenceCache`, including exact `ConvNeXt/HiFiGAN` padding, output length and waveform crop.
2. **F11–F14:** replace the CPU reference executor with real weight-bound TensorRT-RTX execution: network partitioning, external KV buffers, `cudaMemcpy2DAsync`, stream ownership and tensor lifetimes; finish Naive head lowering.
3. **F01:** resolve full Firefly checkpoint/config/weight aliases and support loading both model components from safe manifests.
4. **F08:** offline native MP3/AAC/MP4 codec sources and associated licensing/dependencies.
5. **F22:** model selection, warmup/unload, cancellation on token boundaries, bounded reusable GPU scratch buffers and queues.
6. **V07–V10 separately:** validate on actual checkpoint/RTX and measure speed against original warmed TorchInductor. These are **operability checks**, not functional-percent deductions.

Do not describe synthetic CPU inference as genuine voice, TensorRT mock as GPU, or `cargo check` as an RTX engine smoke.
