# GigaAM persistent inference worker

The `gigaam-rs worker` command is a local, single-model subprocess API for Rust hosts. It uses the same Burn/CubeCL implementation as the library and the existing `serve` command. It does **not** run Python, ONNX Runtime, or its own VAD model.

## Launch and ownership

```powershell
.\gigaam-rs.exe worker --device cuda --model-cache D:\AIData\Models\GigaAM --batch-size 16
```

A worker starts **unloaded**. Its host owns the process and chooses the backend at launch. One loaded model is allowed per process; a second `load` returns an error until `unload` is called. Model changes are explicit, not implicit. `shutdown` replies and exits cleanly. End-of-file on stdin exits the worker. Asynchronous cancellation of an active GPU operation is **not implemented**; the supervisor must terminate the process if it needs to abort a blocked native call.

The optional `--download-model` allows registered models to be fetched through the existing HTTPS cache implementation; by default there is no network access. `--python-audio` selects FFmpeg-compatible PCM16 input conversion (and requires an installed `ffmpeg` executable). Otherwise the built-in WAV/FLAC/Vorbis decoders are used.

## Framing

All input is UTF-8 JSON Lines on stdin. Every nonblank line is one request. Every request yields exactly one UTF-8 JSON response on stdout, including failures. No logs or progress messages are written to stdout. Responses preserve the request `id`:

```json
{"id":1,"method":"status"}
{"id":2,"method":"list_models"}
{"id":3,"method":"load","model":"D:\\AIData\\Models\\GigaAM\\v3_ctc.ckpt","batch_size":16}
{"id":4,"method":"transcribe","audio_path":"D:\\Audio\\speech.wav","word_timestamps":false}
{"id":5,"method":"transcribe_longform","audio_path":"D:\\Audio\\long.wav","segments":[{"start":0.25,"end":4.75}]}
{"id":6,"method":"unload"}
{"id":7,"method":"shutdown"}
```

Success: `{"id":4,"ok":true,"result":{...}}`. Failure: `{"id":4,"ok":false,"error":"..."}`; an individual bad request does not terminate the process. Invalid JSON yields `id:null`.

`load.model` may be a checkpoint filename, model directory, or a registered model alias in the configured cache. `load.batch_size` is optional (default from CLI, initially 16; allowed range 1–256). A failed load leaves the worker unloaded. `unload` is idempotent. `status` returns `protocol_version`, `state` (`unloaded` or `ready`), `model`, `model_kind`, `sample_rate` (16000), `batch_size`, and `external_vad` (true).

After `load`, the worker also supports existing `serve` methods: `transcribe`, `transcribe_longform`, `transcribe_batch`, `embed` and `emotion`. `transcribe_longform` requires externally supplied VAD intervals in seconds; `group_vad` defaults to true and follows the upstream grouping rules, but **no detection** is performed. `embed` writes little-endian float32 embeddings to `output_path` and returns shape metadata. Requests are processed sequentially in input order; batching refers to multiple recordings within a single inference request, not parallel request scheduling.

## Division of responsibility

- **Supervisor:** process start/stop, model selection, external VAD, cancellation/timeout policy, input file ownership, and error recovery.
- **Worker:** one loaded model, checkpoint/cache resolution, preprocessing, inference, transcriptions/embeddings/emotions, request/response framing.
- **Burn runtime:** CPU/GPU execution and tensor memory. Releasing the service on `unload` drops the Rust model; immediate CUDA allocator reclamation is backend-dependent and needs device verification.

`serve <model_path>` remains supported for callers that want the previous eager-loaded, inference-only JSONL process. `worker` is an additive command with explicit lifecycle management, not a replacement of the public `GigaAmService` Rust API.
