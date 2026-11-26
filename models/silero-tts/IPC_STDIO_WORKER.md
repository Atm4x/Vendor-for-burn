# Silero native IPC v1

Launch: `silero-tts --ipc-stdio --backend flex --checkpoint ABSOLUTE_FILE --input-dir ABSOLUTE_DIR --output-dir ABSOLUTE_DIR`. The same binary accepts either official `v3_en.pt` or `v5_ru.pt`; checkpoint metadata determines the variant. No conversion, Python, or PyTorch is required at runtime. The `--backend` is fixed for the lifetime of the process. `flex` denotes Burn Flex CPU; builds with corresponding Cargo features can also expose `cpu` (CubeCL CPU), `cuda`, or `wgpu`. GPU builds accept `--gpu-id pci:0000:01:00.0` or a Windows `luid:` selector via the shared resolver. Without a selector CUDA uses index 0 and WGPU uses its default adapter. GPU builds and their device selection require separate hardware acceptance. The initial Windows package declares only Flex.

The transport uses the shared native-worker-ipc v1 envelope. `capabilities`, `status`, `load`, `warmup`, `unload`, `shutdown` take `{}`. Load initializes the checkpoint and synchronizes the device; warmup requires `loaded` and moves to `ready`; infer requires `ready`. Unload clears the session. Failed load does not publish a session. The process holds one model and one backend. Cancellation requires host process termination.

Example inference request:

```json
{"version":1,"id":"00000000-0000-4000-8000-000000000001","command":"infer","payload":{"silero_version":1,"input":{"kind":"text","text":"Hello, world!"},"speaker":"en_0","sample_rate":8000}}
```

Russian inputs additionally support `ssml`, `accented`, and `symbol_durations`; external RUAccent is not available through this IPC adapter. English currently supports plain text. Sample rates are 8000, 24000, 48000. Status after load exposes actual speaker names. Successful inference returns `silero_version`, `sample_rate_hz`, `sample_count`, `compute_ms`, and `audio` (`file_name`, `size_bytes`, `sha256`). Audio is mono signed PCM16 WAV, written under a unique name in the fixed output directory. The host verifies the descriptor before use. Output bytes are never included in the JSON envelope; no user-defined output path is accepted. Legacy `--stdio` is retained for existing tests and callers.
