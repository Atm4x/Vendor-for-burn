# Silero V5 RU + native RUAccent — Linux and Windows release-gate audit

Date: 2026-11-20. Scope: native Rust 1.100.0 `reference-flex,ruaccent`, RUAccent `tiny`, ONNX Runtime 1.22.0 CPU.

## Immutable model inputs and provenance

- Official Silero V5 RU checkpoint (`https://models.silero.ai/models/tts/ru/v5_ru.pt`), verified SHA-256 `7ba04d42340fe0398042eed2e0d12d62e23096d626b1b9feff4dcb1309197ab4`, converted to SafeTensors for testing.
- RUAccent `ruaccent/accentuator` MIT tiny model pinned to Hugging Face revision `b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855`.
- `ort 2.0.0-rc.15` dynamic Rust bindings, Microsoft ONNX Runtime native library 1.22.0. Runtime inference needs neither Python nor Torch; the Windows CI asset-preparation job uses Torch only once to export the checkpoint.
- The optional `ruaccent_text` provider is opt-in. The default `text` continues to use SileroStress.

## Linux acceptance — PASS

- Real-model release worker rebuilt after restoring **all five experimental ONNX arena files** to their published source versions. No experimental CPU arena flag is present in this result.
- Real-model `verify_ruaccent_real_pcm.py`: **12/12 PASS**, clean worker exit code 0. Both models load in one process; RUAccent PCM is bit-identical to explicitly accented text *on the same platform*; repeated inference is bit-identical; finite PCM at 8/24/48 kHz; long-form in one request; ordinary `text` PCM unchanged before/after RUAccent load; explicit missing-model errors and recovery.
- Reverified long input: 730 text characters; 503,800 PCM samples at 8 kHz = **62.975 s audio**, **2.206 s inference** (RTF 0.03503, 28.55x real-time). Earlier separate Linux run measured 2.045 s for the same long audio; times vary with the shared VM.
- Independently, **57/57** Rust release unit tests PASS with `reference-flex,ruaccent`. Corpus run: 18/18 valid golden inputs synthesize; two intentionally invalid inputs are rejected.

## Windows x64 acceptance — PASS

- GitHub Actions [#44620728387](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44620728387) completed **successfully** on Windows 2025 runner.
- Checkpoint SHA and model manifest verified by the Linux prep job. Windows job installs the official Microsoft ONNX Runtime DLL from NuGet, compiles Silero+RUAccent release with Rust 1.100.0, then executes the same Python-stdlib-only test against actual model assets.
- Windows Rust release unit tests **57/57 PASS**, real-model PCM checks **12/12 PASS**, error list empty, worker exit code 0. The real-model JSONL test takes 10.328 s overall, including startup/loads and requests.
- Long input: **62.975 s audio** in **2.482 s inference**, RTF **0.03941**, 25.37x real-time (Windows GitHub runner; not a production workstation benchmark).
- PCM is **bitwise equal to the explicit-stress reference within each operating system**, but Linux and Windows emitted **different PCM SHA-256 digests** for the same phrases. Cross-platform bitwise equality is **not proven**, and no samplewise cross-OS error metric was collected. Do not overclaim strict cross-OS numerical parity.
- Earlier Windows failures were strictly infrastructure/test-harness problems (missing NumPy, missing SciPy while exporting TorchPackage, Python console encoding); they were fixed without changing Rust TTS numerical algorithms.

## Persistent worker / memory measurements on Linux

- 90 sequential inferences using three voices/rate/text combinations in one persistent worker; **all 90 outputs passed** and PCM SHA was identical on repetition within each input class; 10.562 s total.
- Combined Silero + RUAccent: after load ~902 MiB RSS, after 90 requests ~913 MiB, peak `VmHWM` ~976 MiB. Silero-only after load ~238 MiB. Combined deployment should budget **at least 1 GiB** of process memory plus operational headroom.
- A temporary ONNX CPU memory-arena experiment (`with_arena_allocator(false)` for all four ONNX sessions) reduced RSS after 90 requests from ~915.6 to ~885.7 MiB (saves ~29.9 MiB, ~3.3%), while 90-request durations were 10.794 vs 10.572 s, within benchmarking noise. This marginal saving does not justify another configuration mode; **experiment rejected and all five sources restored**, no runtime changes published.
- The 90-request check is not a proof of leak freedom over unlimited uptime.

## Strict raw-logit numerical parity — OPEN

- Actual model output decisions and 20/20 final RUAccent golden strings match; however two *strict* raw-logit tests fail at tolerance `1e-4` with observed differences roughly `0.001421` (accent) and `0.002067` (omograph).
- Separate Python ORT 1.22.0 diagnostic showed **up to 0.05896** error for omograph against old Python golden raw logits, even when the final class remains correct. Golden source weight/runtime provenance remains underdetermined. Forcing Rust graph optimization `ORT_ENABLE_ALL` did not resolve the two failures and was reverted. The strict test thresholds remain unchanged.
- Functional parity **is verified**, low-level raw-logit parity **is not**.

## Packaging and pending release decisions

- `models/ruaccent` and opt-in Silero `ruaccent_text` are published in `codex/silero-tts-cubecl-port-20261118`; CI and the standalone end-to-end regression script live in that branch.
- Complete offline Cargo vendor additions are validated by Actions [#44611749111](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44611749111) and isolated review [PR #5](https://github.com/Atm4x/Vendor-for-burn/pull/5). They are **not merged** into the main Silero branch, deliberately; the PR contains thousands of vendored files. No Burn/CubeCL source changes were required.
- Windows and Linux real-model PCM are now functional release gates **closed**. Strict logits parity, product packaging of the external model pack + ONNX DLL, cancellation/streaming policies, and broad cross-hardware numerical tolerance are **open** if required for general production release.

## Evidence files

The companion evidence ZIP contains the portable test source, actual Linux and Windows JSONL test reports, 57/57 Windows Rust test log, stress/memory A/B data, and release build logs. It includes **no** neural weights, SafeTensors, Torch checkpoint or ONNX Runtime binaries.
