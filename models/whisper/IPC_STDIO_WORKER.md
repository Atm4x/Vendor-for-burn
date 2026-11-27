# Whisper ASR worker

The host supplies absolute artifact paths from `install-plan.json`. The plan pins the revision, size and SHA-256 of each downloaded file.
Transport uses UTF-8 JSON lines with `version: 1` and a UUID `id`. Commands are `capabilities`, `load`, `warmup`, `status`, `infer`, `unload` and `shutdown`.
Launch with `--ipc-stdio`, `--model-id` and `--backend cpu|cuda|wgpu`.

Pass `--checkpoint`, `--tokenizer`, `--generation-config` and `--input-dir`.
Send `asr_version: 1`, optional `language` (a language code or `auto`), and an `input` WAV snapshot.
The worker accepts the planned Hugging Face Safetensors variants and returns text, source duration and 16 kHz processing rate.
The decoder uses greedy inference and returns utterance text.

Send the launch model ID as **model_id** in the **load** payload. Successful lifecycle replies include **loaded: true** and **warm: true**, respectively.
