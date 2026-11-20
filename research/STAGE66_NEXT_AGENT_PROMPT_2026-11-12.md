# Stage 66 — verify Stage65 before further GPU optimization

Use the latest Git Bridge `rvc-experiment.bundle` from the Stage65 commit and check source-head, then inspect the RTX4060 `RVC_STAGE65_FULL_WAV_RESULTS.zip`. Keep existing PR #2 branch; do not touch unused Flex legacy scripts or private assets.

Confirm full `RvcV2Inferencer` invokes `ResidualExecution::PRODUCTION` and all default NSF calls are sequential; parallel remains explicit experiment only. Test numerical parity and precise output length on real CrazyMita models. Check Stage65 CI build and unit test.

Compare warmed whole-WAV latency against Stage63 under equal GPU load, chunk shape, index and F0 settings. Stage64 synced diagnostic time and Stage44/50 synthetic NSF microbenchmarks are not end-to-end baselines.

If end-to-end RTF is still poor, instrument actual NSF Conv1d / ConvTranspose1d and RMVPE GRU, remove costly launches structurally and validate on the full route. Do not implement synthetic-only speedups as production claims. Preserve single CPU/CUDA/WGPU EXE and no implicit backend fallback.

Prepare source-only handoff, measured report, and following prompt.
