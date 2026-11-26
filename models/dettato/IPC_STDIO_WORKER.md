# Dettato worker

Start `dettato-worker.exe --ipc-stdio` with `--backend cpu|cuda|wgpu`, absolute `--input-dir` and `--output-dir`, and absolute `--<stage>-config` and `--<stage>-weights` paths for each of `transcriber`, `evidence`, `grouper`, and `drums`. The optional `--gpu-id` accepts the shared PCI or Windows LUID selector for CUDA/WGPU. The optional `--cache-mib` sets the Transcriber cache budget.

Use JSON Lines IPC v1. Run `load` and `warmup` before `infer`. The Dettato v1 inference payload is:

```json
{"dettato_version":1,"input":{"file_name":"song.wav","size_bytes":123456,"sha256":"<64 lowercase hex characters>"},"output":"tracks","resample_low_rate_drums":true}
```

`output` is `combined` for one multitrack MIDI or `tracks` for a MIDI per detected track. The result includes the source sample rate, duration, note and drum counts, and an `outputs` array. Each entry has a verified file descriptor and track metadata, or a null track for combined MIDI. The worker keeps the loaded models for subsequent songs. `unload` releases them.
