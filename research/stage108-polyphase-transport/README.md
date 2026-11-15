# Stage 108 — KIC transport for Polyphase Single (RTX 4060)

Changes only memory access geometry. No Winograd approximation or Tensor Core arithmetic.

- `single`: baseline weight order `[Cin,K,Cout/group]`, input view `[N,T,Cin]` with original NCT strides.
- `kic`: weights are prepacked once to contiguous `[K,Cin,Cout/group]`; identical input view and exact multiply-add ordering.
- `kic_ntc`: same weights with a real NCT→NTC contiguous copy per call. Its transfer cost must be counted.

## Execute

The Windows CI packages `stage106_polyphase_probe.exe` and `Run-RVC-Stage108-Transport.ps1`.

Run PowerShell as Administrator if NCU GPU performance counter permissions require it. Use:

`./Run-RVC-Stage108-Transport.ps1 -Backend cuda -Profiler ncu -Runs 12 -NcuLaunchCount 2 -PythonExe 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe'`

The runner validates full output parity against standard ConvTranspose1d and captures only the compute kernel, not the optional NTC-copy kernel. Compare end-to-end benchmark medians to avoid false claims when NTC copying is used. A single run creates a WAV-free ZIP with six NCU reports and JSON logs.

For CPU mathematics: `rustc --edition=2024 --test research/stage108-polyphase-transport/reference.rs -o stage108-tests && ./stage108-tests`.

## Known limits

No RTX 4060 measurements of the new packed layouts exist yet. The default NSF inference is not changed. Only after GPU parity and A/B benchmarking can a production model choose a candidate.
