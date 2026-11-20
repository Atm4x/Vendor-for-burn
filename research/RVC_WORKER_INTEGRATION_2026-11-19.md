# RVC worker integration against NeuroMita supervisor IPC v1 — 2026-11-19

## Source and compatibility

- RVC code source: exact full-history Git Bridge for PR #2 at `d08eb05f83e7d01c43292de973a1a56ac5a0ad06`. The same bundle contains `main` at `a68c6fdb10d8a711a2f36463b6fdb8a6a160cd49` and base `feat/rvc-burn-parity-stage1` at `5b5874752284aeb3fbef7ef5cbedc6015e6b2c9f`.
- Worker contract source: `Atm4x/Vendor-for-burn` branch `codex/gigaam-beat-worker-contract-20261118`, `models/gigaam/IPC_STDIO_WORKER.md` and `models/gigaam/src/ipc_worker.rs`; corroborated by branch `codex/beat-this-windows-worker` `models/beat_this/src/worker.rs`.
- This is the same `version/id/command/payload` envelope used by those workers. Do not confuse it with GigaAM's legacy `{"method":"..."}` protocol.

## Architecture

- Dedicated `rvc-worker` binary, explicit backend, packaged model ID and model directory at process launch. No implicit fallback and only one resident model. The application/supervisor owns lifetimes, snapshots, timeouts and process termination for cancellation.
- Commands: capabilities, load, warmup, status, infer, unload, shutdown. Phases: started, loaded, ready. Invalid framing/version exits; invalid command returns nonfatal error. Stdout only contains framed JSON responses.
- `infer` consumes a verified WAV input snapshot with SHA-256, size <=64 MiB and a single-leaf filename. Output uses an exclusive `<request-uuid>.wav` file with a recorded size and SHA-256. No arbitrary caller-supplied paths and no Python/ONNX.
- `conversion::convert_wav_bytes()` is a single inference route reused by the existing CLI and worker. Audio decode, 16k resample, 48 Hz high-pass, fixed-chunk RVC generation, RMS envelope and 40k mono PCM16 encode have not been independently reimplemented.
- Fixed assets are `voice.pth`, `hubert.pt`, `rmvpe.pt`, optional `index.index` in model-dir. One model identity per worker process; the worker package supplies its checkpoint files externally.

## Validated locally (Rust 1.100.0, Linux, offline)

- `cargo check -p tts-with-rvc-rs --bin rvc-worker --no-default-features --features cpu,cuda,wgpu --offline`: PASS.
- `cargo test -p tts-with-rvc-rs --lib worker::tests --no-default-features --features cpu,cuda,wgpu --offline`: 5/5 PASS, including exclusive output snapshot, SHA mismatch, dangerous filename, lifecycle, bad framing, wrong protocol version.
- `cargo test -p tts-with-rvc-rs --lib audio_io::tests --no-default-features --features cpu,cuda,wgpu --offline`: 3/3 PASS.
- `cargo build --locked --offline -p tts-with-rvc-rs --bin rvc-worker ...`: PASS. Python subprocess `smoke_worker.py` CPU without private model: PASS, verified no stdout text leakage.
- Existing CLI `cargo build --locked --offline --bin tts-with-rvc-rs` and `doctor`: PASS.
- Windows packaging syntax and Windows build are pending CI. Private-model E2E, CUDA JIT and audio parity are pending external hardware testing.

## Merge gate

Keep PR #2 **Draft** until the new Windows CI succeeds and a real installed model passes:
1. `rvc-worker.exe` starts with `--ipc-stdio --backend cuda --model-id rvc_v2_<name> --model-dir ... --input-dir ... --output-dir ...`.
2. `load`, `warmup`, `infer` and `unload` complete without unframed stdout output, stale memory/handles or hidden fallback.
3. Worker and CLI output WAV hashes/PCM match on a fixed input, same parameters/seed/chunk batching, and the RVC regression gates pass.
4. A supervisor import test on NeuroMita Rust application is performed; adapting Vendor-for-burn alone is not end-to-end integration.
5. All relevant Windows CI checks have succeeded against the exact proposed HEAD.

Do not mark ready-to-merge based on an offline `cargo check` or mock test alone. No weights, private WAV, index, or compiled binaries are stored in Git history.
