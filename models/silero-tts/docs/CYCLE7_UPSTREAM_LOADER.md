# Cycle 7: upstream loader and shared V4 graph

Source: user-supplied `silero-v5-ru-offline.zip`, original `silero/silero.py`, `silero/tts_utils.py`, `latest_silero_models.yml`.

The original API resolves a language and model ID in the catalogue, downloads/caches its URL, then loads `tts_models/model` with `torch.package.PackageImporter` for packaged models or `torch.jit.load` for older models. Executable Python wrappers and TorchScript graphs in the package supply architecture-specific behavior. A catalogue loader alone cannot provide arbitrary graph execution in native Rust.

The native implementation embeds a lossless-for-loading JSON projection of the source catalogue (source SHA-256 stored alongside). `--catalogue` and `--model-info LANGUAGE MODEL` expose all 66 entries (46 package, 20 JIT) and explicit verification status. Download/cache remains supervisor-owned; the worker accepts a resolved local checkpoint path. ZIP member structure, not extension or model ID, identifies package versus standalone TorchScript. Standalone TorchScript is detected and rejected explicitly pending native lowering.

V4 shares the existing acoustic and HiFi-GAN implementation with V3, adding dynamic symbol/speaker tables, sparse speaker indices, reduced vocoder widths, SnakeBeta, and the original 24-to-8 kHz PQMF route. The optional V4 Accentor has native int8 dequantization and original embedded ngrams, exceptions and stress/yo controls. There is no runtime Python dependency. Catalogue support does not imply graph support; the verified manifest lists only tested checkpoints with exact URL/hash.

Verified original V4 checkpoints: ru, ua, uz, indic, cyrillic. 27 FP32 comparisons with upstream (8/24/48 kHz), maximum absolute error 6.60e-5. Five Russian phrases include explicit stress, yo and hyphenated words. Total verified checkpoints: 15/46 packages, 15/66 catalogue entries. `random`, SSML, phoneme input, remaining V2/V3/V5 graphs and standalone JIT execution are not claimed. The Python oracle is developer-only.

Reproduce with `python tools/verify_multilingual_matrix.py /path/to/silero-tts /path/to/checkpoints --reject-unimplemented` and `python tools/smoke_native_worker.py /path/to/silero-tts --matrix-dir /path/to/checkpoints`.
