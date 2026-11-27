# OmniVoice TTS worker

The host supplies absolute artifact paths from `install-plan.json`. The plan pins the revision, size and SHA-256 of each downloaded file.
Transport uses UTF-8 JSON lines with `version: 1` and a UUID `id`. Commands are `capabilities`, `load`, `warmup`, `status`, `infer`, `unload` and `shutdown`.
Launch with `--ipc-stdio`, `--model-id` and `--backend cpu|cuda|wgpu`.

Pass `--checkpoint`, `--config`, `--tokenizer`, `--codec-checkpoint`, `--codec-config`, `--input-dir` and `--output-dir`.
Send `tts_version: 1`, `text`, `reference_text`, `steps`, `seed`, optional `language` and a `reference` WAV snapshot.
The result includes a 24 kHz PCM16 WAV snapshot. This adapter implements reference-based voice cloning.

Send the launch model ID as **model_id** in the **load** payload. Successful lifecycle replies include **loaded: true** and **warm: true**, respectively.
