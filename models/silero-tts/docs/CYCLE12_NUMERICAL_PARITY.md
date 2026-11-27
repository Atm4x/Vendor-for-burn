# Cycle 12 — recurrent accentor parity and Tacotron2 error attribution

Base: `8d8b54b5dbd56aacafe0f17e2b89268281eb8173`. Rust 1.100, offline Flex release, AMD EPYC 9V74, four Rayon/PyTorch threads for parity; original development oracle PyTorch 2.10.0+cpu, x86 quantization backend. No new runtime dependency, no graph interpreter, one worker binary.

## Recurrent accentor: numerical defect fixed

The packed dynamic-linear reference rounds the fused affine transform `x * inverse_scale + zero_point`, not the product independently. A quantized identity projection reproduces the difference at symmetric extrema `[-0.9640276, 0, 0.9640276]`: the centered integer result is `[-63, 0, 64]`. This single-quantum discrepancy propagated through the four bidirectional LSTM layers. Use `f32::mul_add` before ties-to-even rounding. No model-specific approximation or fallback.

The prior maximum logit error was 1.598976; after the fix it is 5.3e-5 across 68 cases (34 sentences each for aidar_v2 and ru_v3). All 68 accented strings match the independent oracle. Added a unit test for the affine rounding boundary and a hard 1e-4 logit tolerance to the ignored real-weight test. The expanded sentences cover different word lengths, punctuation, compound words, manual stresses and yo recovery.

## V1: numerical amplification remains open

The production recurrence has not been changed. Extracted the existing decoder rollout so an independent test can replace only its previous output or encoder memory. Captured pre-postnet mel in the original fixture generator. With identical original dropout masks, all 20 long fixtures preserve frame count and gate stop in each diagnostic mode. Max errors over all models:

| Mode | Raw mel max abs | Postnet mel max abs | Cases within mel 3e-3 |
|---|---:|---:|---:|
| Native closed loop | 0.427200317 | 0.416591644 | 2/20 |
| Exact original encoder memory, native feedback | 0.191333771 | 0.184782028 | 2/20 |
| Native encoder, original previous mel each step | 0.000012398 | 0.000017643 | 20/20 |
| Exact encoder memory and original feedback | 0.000012398 | 0.000016689 | 20/20 |

Teacher forcing is diagnostic only, never used in production. Its raw and postnet mel errors now have a separate hard 2e-4 bound. The original closed-loop test retains its 3e-3 mel/PCM and 1e-3 isolated vocoder limits. FP64 Flex was evaluated on four models: the maximum mel error improved for lj and gilles, worsened for baya and aidar, and did not close any of the three failing cases. That experiment was discarded rather than adding a runtime precision branch.

The same original PyTorch model is sensitive to arithmetic changes. With oneDNN disabled and identical dropout masks, 6/10 distinct V1 acoustic graphs fail the same strict mel/PCM thresholds; max mel 0.0192604, max PCM 0.0369545. In four other paired runs the only change was advancing each element of the first feedback frame by one FP32 ULP towards +infinity. Encoder memory and dropout masks were exactly identical; all four exceed the original thresholds, and natasha reaches mel 0.1134195 / PCM 0.0947568. These experiments establish amplification and backend dependence; they do not prove an arbitrary Rust trajectory equivalent or excuse the remaining large closed-loop deviations. Keep the strict gate open and do not claim bitwise or full long-sequence parity.

## Regression

- Release build passed; 62 unit tests passed, four ignored (real weight/oracle tests).
- Expanded real recurrent accentor test: 68/68 with logit tolerance 1e-4.
- All 16 V2 official checkpoints: 18 scenarios, 37/37 PCM comparisons using original dropout masks, max PCM 0.002215996.
- V1 diagnostic test: 20/20 teacher-forced cases for each of two memory sources (40/40), all gate/frame counts agree in four modes.
- Full native IPC smoke: 66/66 official checkpoints (all original weights, not exported fixtures).

## Reproduction

From `models/silero-tts`:

```sh
python tools/create_recurrent_accent_fixture.py CHECKPOINTS ACCENT_FIXTURES
SILERO_MODEL_MATRIX=CHECKPOINTS SILERO_RECURRENT_FIXTURE=ACCENT_FIXTURES cargo test --offline --release --no-default-features --features reference-flex --lib official_recurrent_accentor_parity -- --ignored --nocapture
python tools/create_v2_parity_fixture.py JIT_CHECKPOINTS V1_FIXTURES --standalone
SILERO_V1_MODELS=aidar_16khz,baya_16khz,lj_16khz,natasha_16khz SILERO_V2_CHECKPOINTS=JIT_CHECKPOINTS SILERO_V2_FIXTURES=V1_FIXTURES cargo test --offline --release --no-default-features --features reference-flex --lib original_v1_error_attribution -- --ignored --nocapture
python tools/create_v2_parity_fixture.py JIT_CHECKPOINTS ALT_FIXTURES --standalone --disable-mkldnn --models aidar_16khz baya_16khz gilles_16khz irina_16khz kseniya_16khz lj_16khz natasha_16khz ruslan_16khz thorsten_16khz tux_16khz
python tools/create_v2_parity_fixture.py JIT_CHECKPOINTS ULP_FIXTURES --standalone --feedback-ulp --models baya_16khz gilles_16khz lj_16khz natasha_16khz
python tools/compare_tacotron_oracles.py V1_FIXTURES ALT_FIXTURES
python tools/compare_tacotron_oracles.py V1_FIXTURES ULP_FIXTURES
```

All fixtures use Python only as a development oracle. No original weights, generated audio or binaries are included in the commit.
