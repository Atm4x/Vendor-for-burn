# RVC v2 — resident inference worker

`rvc-worker` is a standalone subprocess which loads explicitly supplied model
weights, warms them once, and accepts sequential voice-conversion requests.
It is not a NeuroMita model registry, installer, ASR/TTS module, or file finder.
It never downloads weights. One model is resident at a time; to change weights,
restart the process with new arguments (or unload and load the same weights).

## Launch configuration

Every asset path is explicit and absolute. No fixed filenames or model directory
structure are required. `--index` is optional; all other weights are required.
The worker name is optional (`rvc_v2` by default) and has no bearing on model
loading or the audio calculation.

```powershell
.\rvc-worker.exe --ipc-stdio `
  --backend cuda --gpu-id pci:0000:01:00.0 `
  --voice 'D:\Voices\CrazyMita.pth' `
  --hubert 'D:\SharedModels\hubert_base.pt' `
  --rmvpe 'D:\SharedModels\rmvpe.pt' `
  --index 'D:\Voices\CrazyMita.index' `
  --input-dir 'D:\Worker\Input' --output-dir 'D:\Worker\Output' `
  --batch-size 8
```

`--backend` selects a **compiled** Burn backend: `cpu`, `cuda`, or `wgpu`.
`--gpu-id` optionally resolves an exact physical adapter by PCI address (or
Windows LUID); it fails rather than choosing another GPU. Without it, the
backend's existing default device is used.

`--backend-path 'D:\CUDA\v12.8\bin'` specifies a directory for Windows DLL
search, or the absolute path to one runtime DLL. **Directory-only selection
cannot prove which libraries the engine actually loaded**. To pin specific
external CUDA libraries, repeat `--runtime-dll` with their absolute file paths:

```powershell
.\rvc-worker.exe --ipc-stdio --backend cuda `
  --runtime-dll 'D:\CUDA\v12.8\bin\nvrtc64_120_0.dll' `
  --runtime-dll 'D:\CUDA\v12.8\bin\cudart64_12.dll' `
  --voice 'D:\Voices\CrazyMita.pth' `
  --hubert 'D:\SharedModels\hubert_base.pt' `
  --rmvpe 'D:\SharedModels\rmvpe.pt' `
  --input-dir 'D:\Worker\Input' --output-dir 'D:\Worker\Output'
```

These filenames are illustrative; specify the DLLs actually shipped with the
chosen CUDA Toolkit. Explicit DLLs are loaded by absolute path **before** GPU
initialization. The Windows loader checks the module's actual path and rejects
conflicts with already-loaded DLLs of the same basename. Dependent DLLs can use
`--backend-path` as a search directory; they are not automatically enumerated or
pinned. DLLs are held for the lifetime of the worker. Passing external libraries
**does not** change the compiled Burn backend. `nvcuda.dll` is part of the
installed NVIDIA driver (not the CUDA Toolkit). Loading HIP/ROCm DLLs cannot
make this CUDA/WGPU build execute HIP kernels; a HIP-capable backend must be
compiled separately. No hidden library download or path discovery takes place.
Outside Windows explicit DLL loading is rejected.

Model assets must be existing, non-symlink regular files; input/output roots must
be existing absolute, non-symlink directories. These are process launch options,
not hardcoded names or worker-specific conventions.

## JSON Lines protocol

Worker receives one UTF-8 JSON object per line on stdin (1 MiB maximum) and
returns one JSON response per valid frame on stdout. All diagnostics go to
stderr. Frames use `version`, UUID `id`, `command`, and object `payload`.

```json
{"version":1,"id":"d693ac39-dd75-48f8-b5fb-5db199110159","command":"capabilities","payload":{}}
{"version":1,"id":"b2e2ecaa-9d58-4d9f-a345-6a3d2d84f31b","command":"load","payload":{}}
{"version":1,"id":"937a1d41-0e69-4c37-872f-d6c5e84a8cab","command":"warmup","payload":{}}
{"version":1,"id":"58e21c92-d329-43cd-bcff-0f604c15683f","command":"status","payload":{}}
{"version":1,"id":"e4164eae-165d-4935-bcfb-31e5d8dabd61","command":"infer","payload":{"input":{"file_name":"speech.wav","size_bytes":123456,"sha256":"<64 lowercase hex characters>"},"index_rate":0.75,"pitch_shift_semitones":0,"rms_mix_rate":0.5}}
{"version":1,"id":"225d20a8-758e-41ce-83de-ea2387b45fa2","command":"unload","payload":{}}
{"version":1,"id":"bb644729-6cfd-43b8-a755-b7d899b4e67e","command":"shutdown","payload":{}}
```

Success: `{"version":1,"id":"<UUID>","ok":true,"result":{},"error":null}`.
Errors: `{"version":1,"id":"<UUID>","ok":false,"result":null,"error":"..."}`.
No model ID, model name, RVC-specific envelope version, or `kind` parameter is
required. `load`, `warmup`, `unload`, and `shutdown` take empty payloads.
`status` reports `started`, `loaded`, or `ready`; `infer` requires `ready`.

`infer` accepts an input WAV snapshot described by a single-leaf ASCII filename,
length (max 64 MiB), and SHA-256. The worker validates the bytes once, processes
the verified buffer, writes a new output WAV to `--output-dir/<UUID>.wav`, and
returns `{file_name,size_bytes,sha256}` alongside input/output sample metadata.
Outputs cannot overwrite existing files. `index_rate` defaults to 0.75 when an
index is provided, otherwise 0; `pitch_shift_semitones` defaults to 0 and
`rms_mix_rate` defaults to 0.5.

`unload` drops the model; `shutdown` sends its reply and exits; EOF also exits.
There is no in-process GPU cancellation, multiple resident models, dynamic
backend switching, or automatic backend/GPU fallback. To cancel a hung GPU
operation, terminate the subprocess. The host owns the input/output folders,
request scheduling, process timeouts, and installed native runtime libraries.

## Validation

The smoke script checks the protocol without model weights using placeholder
files; to test full inference supply `--voice`, `--hubert`, `--rmvpe`, optional
`--index`, and `--input-wav`. Pure unit tests do not establish CUDA JIT or
cross-backend parity with private checkpoints on real hardware.
