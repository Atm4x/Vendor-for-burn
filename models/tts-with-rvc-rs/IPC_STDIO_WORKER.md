# RVC v2 native voice conversion worker — NeuroMita supervisor IPC v1

This worker uses the **same supervisor-facing protocol** as
`models/gigaam/IPC_STDIO_WORKER.md` on branch
`codex/gigaam-beat-worker-contract-20261118`, and `models/beat_this/src/worker.rs`
on branch `codex/beat-this-windows-worker`. It does not implement GigaAM's legacy
`{"method":"transcribe"}` API.

The worker is inference-only, with one loaded model per process; it never runs
Python, ONNX Runtime, TTS, ASR, VAD, or network downloads. The host selects
and owns a **fixed model ID** and explicit Burn backend at process launch.

## Package and invocation

Stage the model files yourself, outside GitHub / CI artifacts:

- `model/voice.pth`: RVC v2 768-channel checkpoint
- `model/hubert.pt`: compatible HuBERT checkpoint
- `model/rmvpe.pt`: RMVPE checkpoint
- `model/index.index`: optional 768-dimensional FAISS IVF index

```powershell
.\rvc-worker.exe --ipc-stdio --backend cuda --model-id rvc_v2_crazymita --model-dir 'D:\AIData\Models\RVC\CrazyMita' --input-dir 'D:\AIData\WorkerInput' --output-dir 'D:\AIData\WorkerOutput' --batch-size 8
```

`--backend` is mandatory: `cpu`, `cuda`, `wgpu`; it never silently falls back.
All directories must be existing absolute, real directories (no symlink at their
root). A package must fix the model ID and its assets; the `load` request cannot
switch the model or backend. The host owns filesystem permissions of the input
and output directories and the process lifecycle. To cancel an in-flight CUDA
operation, **terminate the worker process**. An `unload` drops the model; CUDA
allocator reclamation and true GPU output parity require hardware validation.

## Framing

Each frame is a UTF-8 JSON object followed by `\n`, at most 1 MiB, read from
stdin. The worker sends **one** JSON response on stdout per valid framed request;
all logs/errors go to stderr. Invalid framing, malformed envelope, unknown
fields in the envelope, invalid UUID or unsupported protocol version are fatal.
Unknown command or invalid command payload gets a nonfatal error reply.

```json
{"version":1,"id":"d693ac39-dd75-48f8-b5fb-5db199110159","command":"capabilities","payload":{}}
{"version":1,"id":"b2e2ecaa-9d58-4d9f-a345-6a3d2d84f31b","command":"load","payload":{"model_id":"rvc_v2_crazymita"}}
{"version":1,"id":"937a1d41-0e69-4c37-872f-d6c5e84a8cab","command":"warmup","payload":{}}
{"version":1,"id":"58e21c92-d329-43cd-bcff-0f604c15683f","command":"status","payload":{}}
{"version":1,"id":"e4164eae-165d-4935-bcfb-31e5d8dabd61","command":"infer","payload":{"rvc_version":1,"kind":"voice_conversion","input":{"file_name":"speech.wav","size_bytes":123456,"sha256":"<64 lowercase hex characters>"},"index_rate":0.75,"pitch_shift_semitones":0,"rms_mix_rate":0.5}}
{"version":1,"id":"225d20a8-758e-41ce-83de-ea2387b45fa2","command":"unload","payload":{}}
{"version":1,"id":"bb644729-6cfd-43b8-a755-b7d899b4e67e","command":"shutdown","payload":{}}
```

Response: `{"version":1,"id":"<UUID>","ok":true,"result":{},"error":null}`.
Errors: `{"version":1,"id":"<UUID>","ok":false,"result":null,"error":"..."}`.
Commands are `capabilities/load/warmup/status/infer/unload/shutdown`. The phases
are `started`, `loaded` and `ready`. `infer` requires `ready`.

### Input snapshot and output

The supervisor copies a WAV to `--input-dir` and supplies a single-leaf ASCII
filename, exact `size_bytes` (1–67108864) and lowercase SHA-256. Worker rejects
symlinks, path traversal, modifications, invalid RIFF/WAVE headers and digest
mismatch. It reads and verifies the file **once**, and runs the same internal
preprocessing/inference as the CLI on the verified bytes.

Output is `--output-dir/<request-uuid>.wav`. The worker creates the file with
`create_new`, never overwrites existing files, and returns `{file_name,
size_bytes,sha256}` plus input/output sample-rate/count metadata. Host verifies
and consumes this output snapshot, then deletes it. The output is 40 kHz mono
PCM16; input may be WAV mono or stereo with a supported PCM/float format.

The optional `index_rate` defaults to 0.75 when `index.index` is present and 0
otherwise; positive index rate is rejected when no index is installed.
`rms_mix_rate` defaults to 0.5, pitch shift to 0 semitones.

## Deployment and acceptance limits

The worker shares the **same Rust RVC checkpoint reader and CubeCL inference
implementation** as the CLI, including fixed chunking and tensor batching. A
CLI/worker unit check cannot establish that the CUDA backend is compiled for
Windows, successfully JITs every kernel on RTX 4060, or has identical audio to
PyTorch. Those require a Windows build and a private-model end-to-end run.

The protocol does not provide partial streaming, in-process cancellation, hot
model swapping, multiple concurrent models, or automatic backend fallback.
EOF shuts down the process. This executable is *voice conversion*, not a TTS
model that generates speech from text. NeuroMita orchestration stays above the
worker, including TTS->RVC chaining when requested.
