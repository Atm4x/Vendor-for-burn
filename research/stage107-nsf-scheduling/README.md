# Stage 107 — NSF register budget and phase scheduling

Targets the two effects measured in Stage 106 on RTX 4060:

- `noise40` Direct Conv1d was L1/TEX-limited; temporal pair achieved 8.37× end-to-end speedup. `interior` retains exactly the same vectorized output-channel tiling and pair accumulation while hoisting interior/boundary decisions outside the tap loop.
- `upsample8` and `upsample10` polyphase improved end-to-end latency by eliminating conversions, but the polyphase compute kernel was not faster than Direct. `single` trades two-output per-lane reuse for twice as many independent lanes and a single accumulator. No claim of speedup until CUDA A/B.

## Tests

`rustc --edition=2024 --test research/stage107-nsf-scheduling/polyphase_single_reference.rs -o /tmp/stage107-poly-test && /tmp/stage107-poly-test`

`rustc --edition=2024 --test research/stage107-nsf-scheduling/noise_interior_reference.rs -o /tmp/stage107-noise-test && /tmp/stage107-noise-test`

`python research/stage107-nsf-scheduling/windows/test_stage107.py`

Windows release CUDA/CPU probes are built in the sole active workflow and compare complete output tensors against the existing operator. CPU mathematical tests are not a substitute for CUDA output parity.

## RTX 4060 run

Extract the GitHub Actions artifact `rvc-stage107-nsf-scheduling-windows-x64` and run PowerShell **as Administrator** for performance counters:

`./Run-RVC-Stage107-NSF-Scheduling.ps1 -Backend cuda -Profiler ncu -Runs 12 -NcuLaunchCount 3 -PythonExe 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe'`

Without hardware counters, pass `-Profiler none`. Report `RVC_STAGE107_NSF_SCHEDULING_RESULTS.zip` contains JSON, logs and `.ncu-rep` only; no WAV or models. `ncu` process-scope NVTX requires `nvToolsExt64_1.dll` from Torch and `RVC_NVTX=1`; the runner configures and validates both before executing anything.

## Interpretation

Compare non-NCU medians for end-to-end operator performance, and `.ncu-rep` main compute-kernel times for arithmetic dataflow efficiency. Do not infer kernel speedup from operator time alone; Stage 106 polyphase demonstrates the difference. The two strategies are research-only; default production inference is unchanged until GPU A/B and PCM gate both pass.
