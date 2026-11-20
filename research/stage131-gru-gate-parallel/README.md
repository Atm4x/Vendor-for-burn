# RVC Stage131: persistent GRU gate-parallel A/B

Requires RTX4060 Windows build artifact, Nsight Systems/Compute installed at paths used by prior Stage125 scripts, 85–95 second `embed/MitaKind.wav`, and RVC model assets in `embed/`. No assets are packaged by CI.

Run from the extracted Windows artifact folder:

```powershell
.\Run-Stage131-GRU-AB.ps1 -Runs 3 -Cycles 2 -Profiler all
```

Quick preflight:

```powershell
.\Run-Stage131-GRU-AB.ps1 -CheckOnly
```

Compare modes `control`, `coalesced`, `gate_parallel`, `split2`, `split4` from the same binary with identical warm timing scope. `gate_parallel` and `coalesced` must be PCM-bit-exact to baseline; `split2/4` are diagnostic and require SNR >=70dB by default. Nsight Systems records synchronized warm GPU kernel durations; Nsight Compute reports hardware counters on isolated GRU launches and is not comparable to warm latency. All raw diagnostics and postprocessed per-mode summaries are written to `stage131_gru_results/<timestamp>.zip`, excluding WAV, checkpoints and executable.
