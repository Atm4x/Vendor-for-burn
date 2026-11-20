# Stage 76 — Diagnose RMVPE voiced-threshold flips and validate causal F0 restoration

Date: 2026-11-12. PR #2 parent Stage75: \`e31df3f33e09c21637a2ae9c1d0f203d2c52cdaf\`.

## Evidence — user-provided real RTX4060 Stage75

Analyzed **all 405 files** in \`RVC_STAGE75_EXACT_F0_RESULTS.zip\`: per-mode 32 per-chunk float32 F0 and coarse pitch arrays, WAVs, stderr, stdout, benchmark JSON, CSV and summary. Four full 89.93925s WAV modes, 32 chunks, 3,597,570 output samples, CPU F0, NSF all-GEMM, persistent BiGRU, CUDA Graph off, exact repeat per mode.

| RMVPE UNet conv | Warm full RVC s | Pitch bin changes | Voicing changes | F0 RMSE Hz | Max F0 Hz | PCM16 RMSE vs Direct |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 12.2958 | 0 | 0 | 0 | 0 | 0 |
| Conv2d ImplicitGemm only | **10.3378** | 15 | **3** | 3.53135 | 240.34586 | **985.608** |
| ConvTranspose2d Gemm only | 12.1132 | **0** | **0** | **0.000404** | **0.004944** | **2.597** |
| Both | 10.9696 | 15 | 3 | 3.53142 | 240.34586 | **991.129** |

All F0 comparisons cover 11,990 aligned frames, including the 148-frame final chunk. Only three F0 frames have absolute differences above 1 Hz in Conv2d mode. **Every large difference is a voiced/unvoiced toggle at an isolated 10ms RMVPE frame**, surrounded by closely matching F0:

| Chunk | Index within F0 row | Approx absolute time | Direct F0 Hz | GEMM F0 Hz |
|---|---:|---:|---:|---:|
| 005 | 289 | 16.89s | 0 | 186.726 |
| 014 | 123 | 41.15s | 0 | 238.511 |
| 027 | 72 | 78.08s | 240.346 | 0 |

Other 12 changed coarse pitch bins are rounding boundaries with small F0 differences (typical 0.002-0.025 Hz). \`salience_to_f0\` in \`rmvpe.rs\` returns 0 for per-frame max salience **<=0.03**, otherwise a weighted nine-bin centroid. Therefore these three toggles imply the peak confidence crosses that decision threshold under the GPU arithmetic change. This does **not** yet prove which Conv2d layer or specific rounding operation changed the score.

### Strong localization to NSF phase-sensitive regions

Independently computed exact PCM16 squared-error energy on each chunk's output region using the *actual user WAVs*, not synthetic audio. The **union of chunks 005,014,027 (10.01% of output duration) contains 99.9043% of total Conv2d-vs-Direct squared PCM error**. Outside those chunks, PCM16 RMSE is ~32.15, not 985.61. Peak 1s audio RMSE at 80-81s is ~7,359 PCM16 units.

The production \`SineGen\` constructs 400 samples per F0 frame, retains a cumulative wrapped phase across frames, and changes voiced masking/noise amplitude at F0=0. A single 10-ms voiced decision change can therefore affect the remaining synthesis frames within a chunk.

Performed a **model-limited offline phase-integrator control** using all actual Stage75 F0 frames, same 40kHz/400 samples per frame wrapped phase and sine formula, without the downstream neural decoder/noise:
- Raw GEMM F0 vs Direct sine-source RMSE = **0.0268703**.
- Restoring *only the three mismatched voiced decisions* to Direct F0, leaving all other GEMM F0 untouched, yielded RMSE **0.0004319**.
- Source sine **squared-error reduction ~99.9742%** (ratio 0.00025836).

This strongly supports a voicing-threshold → accumulated-phase mechanism. **It is NOT a full end-to-end RVC A/B yet**, and cannot be cited as an RTX 4060 measured final waveform speed/quality result.

## Experimental Stage76 code

Add an **explicitly opt-in diagnostic F0 reference** to the actual \`RvcV2Inferencer\`. At the existing \`f0_batch\` CPU decode boundary, **before FAISS protection, coarse pitch, TextEncoder or NSF**, it can read Direct F0 captured by the existing Stage75 \`RVC_F0_AUDIT_DIR\` writer from the exact input and chunk geometry:
- \`RVC_F0_REFERENCE_DIR\` — directory containing reference \`chunk_NNN.f0.f32le\` plus meta JSON.
- \`RVC_F0_REFERENCE_MODE=voicing\` — restore reference F0 **only for frames where zero/nonzero decision differs**.
- \`RVC_F0_REFERENCE_MODE=full\` — restore all reference F0 frames.

Strictly enforce same chunk index, frame count, input samples, nonnegative finite F0, CPU F0 path, zero pitch shift, and no shared global F0. No reference argument means **production algorithm unchanged**. This debug ablation is not a production reference-dependency solution.

The new \`Run-RVC-F0-Causal-Stage76.ps1\` A/B uses four real full-WAV cases in order:
1. Direct reference plus F0 capture.
2. Conv2d GEMM uncorrected.
3. Conv2d GEMM + restore only 3 voicing-boundary frames.
4. Conv2d GEMM + fully replace all F0 from Direct.

Each case runs 2 full passes, exports WAV/FP32 F0/PCM16 SNR/RMSE, stderr replay counts and file hashes to \`RVC_STAGE76_F0_CAUSAL_RESULTS.zip\`. CI must confirm Windows release build and reference unit test; **no physical GPU test has run for Stage76 yet**.

### Acceptance

The voicing-only case should show 0 voicing mismatches and 3 fixed frames per full pass, and a large drop in WAV RMSE relative to raw Conv2d; exact remaining error is **unknown** until tested. Full-override should produce bit-identical F0 and plausibly bit-identical WAV; investigate any difference. Preserve baseline NSF all-GEMM/persistent GRU; do not enable Conv2d GEMM or change the RMVPE decision threshold for ordinary users until full-RVC listening and multiple audio inputs are verified.

CUDA Graph replay is **not integrated**: all Stage75 real WAV benchmarks report \`graph_replay=false\`. Do not mix graph changes into this numerical experiment.
