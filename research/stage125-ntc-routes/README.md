# Stage125 — direct NTC Conv1d route matrix (no implicit GEMM)

One Windows CI artifact contains tts-with-rvc-rs.exe, Run-Stage125-NTC-Routes.ps1, Export-Stage125-NCU-Metrics.py, README, technical report and R&D workflow contract. Place your existing embed/ folder next to the EXE. No model files or WAV are distributed.

Run from the extracted artifact:

    .\Run-Stage125-NTC-Routes.ps1 -Cycles 2 -Runs 5 -Profiler all

Modes are control, linear, residue, combined, and optional legacy. Use -Modes 'control,linear,residue,combined,legacy' to add legacy. In every mode the binary runs full RVC with persistent RMVPE GRU, ResBlock transport, shared NTC and temporal tile4. Control enables Stage123 activation placement. Linear adds Stage124 fast-interior and linear-index Conv1d. Residue additionally processes the FIRST C128/K11 D3 and D5 convolution in residue-ordered time tiles with no new pack buffer. Combined enables both. Legacy has no activation-placement optimization.

The run alternates route order across cycles. For each candidate, it compares complete PCM16 WAV output with control before NCU/NSYS. On any PCM mismatch it stops profiling, writes logs, a manifest and an archive. It reports cold and warm times independently. NCU uses --set full and name matching, default --launch-skip 32 --launch-count 4 to try to capture K11 C128 D3 first/second and D5 first/second; VERIFY exact kernel geometry from NCU logs. NSYS exports full CUDA timeline as SQLite, not aggregated nsys stats. The ZIP excludes WAV, weights and audio.

Available options include -Profiler measure|ncu|nsys|all, -Modes, -Runs, -Cycles, -CheckOnly, -KeepWav and profiler paths. CUDA benchmarks and PCM parity must be run on your GPU; Windows CI only builds and checks CPU mathematical references. The user's only Python interpreter is D:\Projects\tts-with-rvc\.venv\Scripts\python.exe. The bundled NCU exporter uses this interpreter, not Conda Python.

These routes are experimental: Irodori's measured packed residue D1 kernel is not reproduced. The residue-order route tests a cheaper scheduling variation only, and may be slower; neither it nor fast-interior is accepted until Nsight and bit-exact PCM justify promotion. No generalized implicit GEMM, separate im2col or additional CUDA allocations.
