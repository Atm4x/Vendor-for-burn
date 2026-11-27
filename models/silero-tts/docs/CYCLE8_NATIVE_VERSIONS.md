# Cycle 8 — native version coverage

A single Rust worker dispatches typed graphs by checkpoint schema, without executing serialized Python or TorchScript code. No new runtime dependencies.

- V3: v3_1_ru, v3_xal, v3_tt, v3_uz, v3_ua. Shared ASCII accentor for V3.1 and V4; warmup chooses a valid Unicode symbol from the checkpoint.
- V5: v5_1_ru through v5_6_ru, v5_caucasian, v5_cis_ext. Embedded PQMF variants, sparse speaker indices, and separate pitch-only and sentence-conditioned prosody policies. Explicit focus/emphasis and SSML on conditioned V5 are not implemented.
- All 13 newly supported checkpoints were compared against original upstream PyTorch at 8/24/48 kHz, with 39 matching-length PCM comparisons. Five additional V5.6 intonation cases at all three rates also passed. Maximum absolute FP32 errors: V3 4.54e-4; V5 4.24e-5. Original weights remain external.
- The manifest now lists 28 verified checkpoints. The catalogue has 66 entries, including 20 standalone JIT entries. Coverage counts entries, not every optional feature.
- Remaining: V2 Tacotron2 autoregressive families (aidar_v2, lj_v2, mykyta_v2, multi_v2 inspected), ru_v3 packed quantized bidirectional recurrent accentor, remaining V5 catalogue entry and standalone JIT graph families. Unsupported graphs are rejected, not substituted.
