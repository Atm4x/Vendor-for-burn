# RUAccent: real tiny ONNX model audit (2026-11-19)

## Exact inputs

- Rust native source: `models/ruaccent`, derived from `alksef/ruaccent-rs` at `4dd4b2f49d2250ee17448484d040180e042f9cee`.
- Official model: `ruaccent/accentuator`, immutable Hugging Face revision `b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855`.
- Only the `tiny` ONNX subset, dictionaries and tokenizers were downloaded. No weights are committed or stored in the workflow artifact.
- `ort = 2.0.0-rc.15` Rust binding with `load-dynamic`, CPU inference using the Microsoft.ML.OnnxRuntime **1.22.0** Linux native library from Microsoft's official NuGet package.
- GitHub Actions workflow `.github/workflows/ruaccent-tiny-model-parity.yml`; successful functional runner [44596020035](https://github.com/Atm4x/Vendor-for-burn/actions/runs/44596020035).

## Results with real ONNX models

| Check | Result |
|---|---|
| Required model assets | 13/13 found |
| Entire Rust library suite including strict raw-logit comparison | **39 PASS / 2 FAIL** |
| Same suite, excluding exactly those two raw-logit tolerance tests | **39/39 PASS** |
| Worker protocol tests | **2/2 PASS** |
| Golden text splitter | **PASS** |
| Golden `process_all` comparison | **20/20 exact output strings** (passes within the 39-test set) |
| Real JSONL worker load, 2 process, status, shutdown | **PASS**, exit code 0 |

Worker inference verified these outputs:

- `На двери висит замок.` → `На дв+ери вис+ит зам+ок.`
- `Он все понял и ушел.` → `+Он всё п+онял и уш+ёл.`

The two remaining strict numerical failures:

1. `models::accent::tests::logits_match_golden_within_1e4`: one observed logit `7.4491653` vs Python reference `7.4505863`; absolute difference ~`0.001421`.
2. `models::omograph::tests::logits_match_golden_within_1e4`: one observed logit `-1.541515` vs `-1.5394485`; absolute difference ~`0.002067`.

These exceed the existing `1e-4` raw-logit tolerance, so strict logit-level parity is **not established**. We **did not** change the comparison threshold or suppress these failures. Functional word/stress decisions and the final 20 golden strings do match. The difference may relate to CPU numerical kernels/runtime, but its root cause has not been conclusively identified.

The CI workflow audits the complete strict suite with `continue-on-error`, then makes successful functional parity and the real worker run hard gates. Therefore a green workflow indicates **functional verification**, not a passing strict logit suite. Exact failures remain visible in `cargo-model-tests.log`.

## Outstanding work / release boundary

- The native `models/ruaccent` pipeline and standalone worker now run with real `tiny` weights; a fresh Rust rewrite is unnecessary.
- The optional Silero `ruaccent_text` adapter was previously type-checked locally but **has not been merged into the standard Silero worker**. The existing SileroStress behavior and PCM remain unchanged.
- Any additional direct Silero integration must be opt-in, have explicit model-pack requirements, and have a real paired model integration test. Do not silently replace the existing accentor.
- Resolve or explicitly document the two `1e-4` raw-logit discrepancies if strict low-level parity is a release requirement. Test at least one Windows ORT bundle before claiming a Windows release.
- Real model weights and an appropriately licensed ONNX Runtime library must be provided at deployment time. The pinned model pack is not in the source repository.

## Evidence

Workflow artifact `ruaccent-tiny-model-parity-evidence`, run 44596020035, includes `model-manifest.json`, `cargo-model-tests.log`, `cargo-functional-tests.log`, and `worker.json`, without neural weights.
