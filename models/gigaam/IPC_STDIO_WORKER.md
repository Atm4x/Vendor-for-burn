# GigaAM packaged worker — Beat This! protocol v1

This is the **same supervisor-facing stdio envelope and lifecycle** as `models/beat_this/src/worker.rs` on `codex/beat-this-windows-worker`. It is separate from the older `gigaam-rs worker` / `gigaam-rs serve` compatibility APIs. The worker does not run its own VAD, does not require ONNX and does not make network requests.

## Launch

```text
gigaam-rs.exe --ipc-stdio --backend cuda --model-id gigaam_v3_ctc --model-dir D:\AIData\Models\GigaAM\v3_ctc --input-dir D:\AIData\WorkerInput
```

As with Beat This!, `--model-dir` and `--input-dir` must be absolute, existing, non-symlink directories. `--runtime-dir` is accepted for host launch compatibility but is optional (GigaAM's native WAV/FLAC decoder needs no SoXR). `--output-dir` is required only for SSL embedding results, also an existing, absolute, non-symlink directory. `--backend` is mandatory (`cuda` when compiled with CUDA, `cpu` when built with Flex or CubeCL CPU). A CPU/Flex build advertises `burn_flex_cpu_fp32`, CUDA advertises `burn_cubecl_cuda_fp32`. An unsupported backend fails at launch without writing to stdout.

Each worker package identifies **one registered model** via `--model-id` (`gigaam_v3_ctc`, `gigaam_v3_e2e_ctc`, `gigaam_v3_e2e_rnnt`, `gigaam_v3_ssl`, `gigaam_emo`, and other registered GigaAM model families). `load` cannot switch to a different model ID. The expected checkpoint is `<model-name>.ckpt` inside `--model-dir`; the synthetic fixture names `model.ckpt` and `model.safetensors` are also supported with their explicit `config.json`. Matching tokenizer files must be in the model directory for E2E and other tokenizer-based checkpoints.

## Framing and lifecycle

Send UTF-8 JSON Lines to stdin, one object per line. The frame must terminate with `\n` and must be at most 1 MiB. The Beat This! envelope is preserved exactly:

```json
{"version":1,"id":"f6976f6d-6db4-4cec-9721-709c89304caf","command":"capabilities","payload":{}}
{"version":1,"id":"fd21a536-ac28-498b-b5b2-f92eaa8290eb","command":"load","payload":{"model_id":"gigaam_v3_ctc"}}
{"version":1,"id":"a66e16be-c4d4-4fa2-9c6b-995267d43dd5","command":"warmup","payload":{}}
{"version":1,"id":"7c118aee-6fd3-4f72-b3d6-4432e820c520","command":"status","payload":{}}
{"version":1,"id":"f4d1d230-5244-40d4-bcb1-ea10822db319","command":"unload","payload":{}}
{"version":1,"id":"20b6d894-d810-4740-9d99-ccf7f0a72712","command":"shutdown","payload":{}}
```

Success responses have `{"version":1,"id":"<UUID>","ok":true,"result":{},"error":null}`; command errors use `ok:false`, `result:null` and `error:"..."`. Invalid request framing, envelope or protocol version is fatal, matching Beat This!. Invalid **commands and payloads** return errors but leave the process available for the next command.

Commands: `capabilities`, `load`, `warmup`, `status`, `infer`, `unload`, `shutdown`, plus `stream_start`, `stream_push`, `stream_finish`, `stream_cancel` on Zapformer ASR packages. Status phases: `started` (unloaded), `loaded` (weights loaded, not warmed), `ready` (warmup complete). `infer` requires `ready`. `unload` clears the model and the warm flag; `shutdown` replies and exits; EOF also exits. Only one model is resident. To select a different checkpoint package, the supervisor launches a different worker process.

## Snapshot-only inference

The NeuroMita Core copies the audio to its own input directory and sends a **single-leaf** reference and its SHA-256 digest. Audio is never accepted by an arbitrary client-supplied path. The worker rejects path traversal, symlinks, size mismatches, modified snapshots, and incorrect WAV/FLAC headers. It reads at most 64 MiB and decodes the verified bytes in memory. Example ASR request:

```json
{"version":1,"id":"329af929-aeb0-4089-8a99-ea72840ad103","command":"infer","payload":{"gigaam_version":1,"kind":"asr","input":{"file_name":"speech.wav","size_bytes":32044,"sha256":"<64 lowercase hex digits>"},"word_timestamps":false}}
```

An `asr` result has `gigaam_version`, `kind`, `duration_seconds`, `sample_rate_hz` (=16000), `text`, and `segments` (each contains `text`, `start`, `end`, optional `words`). One recording without VAD intervals must be at most 25 seconds. For longform, pass `segments:[{"start":0.5,"end":2.7},...]` from the host VAD; Python-style grouping is enabled by default (`group_vad:false` disables it). The worker never detects speech intervals itself.

Zapformer ASR snapshots additionally accept optional `head:"raw"` or `head:"e2e"` for dual-head checkpoints (default E2E) and `language` for conditioned checkpoints (tag, list, weighted map, or `"auto"`). These are request-scoped. Legacy models reject these explicit options. Native Rust clients can use `GigaAmService::start_stream` for packet inference; the stdio worker exposes the same stateful engine. The host continues to own VAD and utterance boundaries.

For an SSL package, use `kind:"embeddings"` with the same version and snapshot; it returns `frames`, `channels` and `output:{file_name,size_bytes,sha256}`. Float32 little-endian output is written to the host-owned `--output-dir` under the request UUID as `<UUID>.f32`, without overwriting pre-existing files. For Emo, use `kind:"emotion"`; the response contains `probabilities` keyed by emotion name. Neither of these operations accepts ASR-only timestamps or VAD intervals.

## Stateful utterance streaming

Streaming is advertised in `gigaam_capability.streaming` for registered Zapformer ASR packages; other families return null. A loaded checkpoint must actually declare the requested context mode. The wire envelope remains version 1. The worker owns at most one active utterance, not one model copy per stream. The host still owns VAD, PCM conversion, boundaries, and snapshot cleanup.

After load and warmup:

- `stream_start`: payload `{"gigaam_version":1,"future_context_ms":160,"language":"ru","word_timestamps":true}`. Language is optional and only valid for conditioned checkpoints; omit it on monolingual ones. Omit the context to select the checkpoint's largest declared context. The request UUID becomes the stream ID. Result: `{"stream_id":"<UUID>","next_packet_index":0}`.
- `stream_push`: payload `{"stream_id":"<UUID>","packet_index":0,"input":{"file_name":"packet.f32","size_bytes":5120,"sha256":"<digest>"}}`. This snapshot contains raw little-endian float32, mono, 16000 Hz samples, no header, no resampling or VAD. It uses the same leaf-name and digest checks as file inference. Packet size is at most 640000 bytes and utterance length at most 4800000 samples (300 seconds). Every sample must be finite.
- `stream_finish`: payload `{"stream_id":"<UUID>","packet_index":1}` for the next expected sequence number. Flushes the unfinished PCM and encoder context and releases stream state. No additional input is accepted after finish.
- `stream_cancel`: payload `{"stream_id":"<UUID>"}`. Discards state without flushing; the warmed model remains loaded.

Push and finish return `gigaam_version`, `stream_id`, `packet_index`, cumulative `accepted_samples`, cumulative `encoded_frames`, `finished`, and `heads`. Each head contains text publication delta, newly decoded token IDs, absolute utterance-relative frame indices, and optional word publication deltas (seconds relative to utterance start). Dual-head checkpoints return both `heads.raw` and `heads.e2e`; single-head checkpoints return `heads.single`. The host selects the display head; no second encoder invocation is required. A word or text fragment may only become available on a later packet or finish.

Packets must be contiguous in sequence order. Incorrect stream ID, sequence, malformed or corrupt PCM, or size limit errors do not consume a packet. An inference/finalization error discards the active stream; the host can open a fresh one. Starting another stream while one is active is rejected. Status includes `active_stream` (UUID or null). Unload, shutdown, and EOF release all stream state and the model. Transport retry after a lost successful response is not idempotent: recover via status or cancel/restart rather than replaying accepted PCM.

## Windows package manifest

The package metadata follows Beat This!'s `native-model.json` schema version 1 and install-plan convention. The per-model script `models/gigaam/package_windows.ps1` computes real SHA-256 digests from a supplied checkpoint and executable. It does not invent trusted digests or bundle user-specific weights. Checkpoints/tokenizers are separate installer assets. Example:

```powershell
.\models\gigaam\package_windows.ps1 -Version 0.1.0 -ModelId gigaam_v3_ctc -CheckpointPath 'D:\Models\v3_ctc.ckpt' -ExecutablePath 'D:\Build\gigaam-rs.exe' -OutputRoot 'D:\Stage\gigaam-v3-ctc' -Backend cuda
```

`-TokenizerPath` and optional upstream `-CheckpointUrl`/`-TokenizerUrl` are supported for model-specific install plans. The script is not a GPU correctness or release certification test; use separately compiled matching-backend executables.

## Host responsibilities

The supervisor owns worker start/restart, snapshot creation/deletion, model selection, timeouts, external VAD, cancellation by process termination, installer integrity and model/license policy. Requests are processed serially, just like Beat This!; there is no in-process cancellation of an active GPU kernel and no extra VAD or network service.
