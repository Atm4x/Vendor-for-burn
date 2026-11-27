# Cycle 6 — catalogue-wide architecture audit, 2026-11-27

Baseline `7f71da6abe4f869381efb3d992982b5f0fcccda5`. Scope: official Silero **TTS**, not VAD, STT or other projects. Runtime is Rust-only; upstream Python/PyTorch is used exclusively as a developer test oracle. The loader dispatches by package structure and tensor schema, never by model filename. No additional runtime dependencies.

## Implemented graph families

- `v3_hifigan`: common 363-tensor V3 graph, dynamic Unicode/script alphabet mapping and speaker embeddings. Six real packages verified.
- `v5_ru`: existing 311-tensor neural-stress graph, Accentor and HomoSolver. Newer, structurally different graphs must not be silently accepted.
- `v5_convnext_plain`: shared single-graph plain-text V5 with explicit token IDs, dynamic speaker routes within the one package, and per-language transliteration from checkpoint metadata. PQMF buffers are extracted from the script module. Existing acoustic and ConvNeXt synthesis kernels are reused.

`model_variant` previously reported `v3_en`; consumers should now treat `v3_hifigan` as a graph identity, separate from the catalogue ID. `random` is a synthetic-voice pseudo-speaker, not a concrete trained voice, and is excluded from advertised choices. No support is claimed for random generation, V3/plain V5 SSML or explicit durations, V5 caret controls, phoneme inputs or multi-graph routing. Indic V3 expects upstream-style ISO romanization. Long-form chunking is the existing native policy, not a claim of exact upstream behavior on arbitrary long input.

## Official catalogue snapshot

Source: https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml (retrieved 2026-11-27). Enumerated 46 entries with a `latest.package` URL; this is a defined denominator, not all releases or every historical artifact. Some entries refer to standalone TorchScript. 16 distinct packages acquired and inspected in this or preceding cycles, 10 verified for native plain text. Windows staging uses `verified_models.json` (exact URL, size, SHA-256 and smoke input); catalogue entries not verified are deliberately omitted from installable models.

| Group | Verified | Not yet verified |
| --- | --- | --- |
| V3 | v3_en, v3_en_indic, v3_de, v3_es, v3_fr, v3_indic | ru_v3, v3_1_ru, v3_xal, v3_tt, v3_ua, v3_uz |
| V4 | — | v4_ru, v4_ua, v4_uz, v4_indic, v4_cyrillic |
| V5 | v5_ru, v5_cis_base, v5_cis_base_nostress, v5_turkic | v5_1_ru, v5_2_ru, v5_3_ru, v5_4_ru, v5_5_ru, v5_6_ru, v5_caucasian, v5_cis_ext, v5_cis_base_nostress_jit |
| V2 | — | aidar_v2, baya_v2, irina_v2, kseniya_v2, natasha_v2, ruslan_v2, lj_v2, thorsten_v2, tux_v2, gilles_v2, aigul_v2, erdni_v2, dilyara_v2, dilnavoz_v2, mykyta_v2, multi_v2 |

The acquired V4 packages have different graph structures (382 tensors, or 399 in V4 RU). V5_6 RU adds control embeddings and has 321 TTS tensors. These are not solved by removing shape checks. Priority: native V4 common graph, older V3 variants, V5 multi-graph and control variants, standalone JIT/V2. Preserve unsupported-architecture errors rather than pretending to execute arbitrary TorchScript.

## Acceptance

Rust 1.100.0, offline release, Burn Flex on Linux. `tools/verify_multilingual_matrix.py` independently compares original upstream FP32 PCM against the worker from an arbitrary checkpoint filename, tests 8/24/48 kHz and the exact concrete speaker set. 33/33 cases passed, exact sample counts; max absolute error 4.6846271e-4 and min SNR 76.16 dB. V5 plain subset: 15/15 including Armenian and Azerbaijani native script/Latin inputs mapped from metadata, max error 1.1786819e-4. Manifest IPC smoke: 10/10 model files (plus corrupt-file rejection); warmup, deterministic repeated WAV, unload, checksum and sample count. Existing worker regression: 58 ordinary unit tests plus 1 real fixture acoustic/vocoder test, 6 IPC/worker integration tests including real RU/EN. All five acquired V4 packages and v5_6_ru rejected with explicit graph compatibility errors (6/6); importer standalone tests 3/3. Windows packaging script now tests the entire verified manifest; actual Windows run of this revision is a separate acceptance gate.

Reproduce with `RAYON_NUM_THREADS=4 python tools/verify_multilingual_matrix.py /path/to/silero-tts /path/to/checkpoints`, with `<catalogue-id>.pt` filenames in the checkpoint directory. Add `--reject-unimplemented` when the six incompatible packages are also present. Python dependencies belong only to the test oracle. `tools/smoke_native_worker.py /path/to/silero-tts --matrix-dir /path/to/checkpoints` uses stdlib only, expecting `<catalogue-id>-renamed.pt` as in the staging script.
