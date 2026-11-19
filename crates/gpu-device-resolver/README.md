# gpu-device-resolver

Small Rust SDK for selecting an **exact physical GPU** in Burn/CubeCL without exposing runtime-specific device indices to the caller.

The caller owns an OS-provided identifier: `pci:0000:01:00.0`. A Windows session-local `luid:7856341201000000` is supported as a second, non-persistent selector. Both refer to physical devices; neither encodes a CUDA/WGPU index.

## API

```rust
use gpu_device_resolver::{Backend, GpuId, enumerate, resolve};

let pci: GpuId = "pci:0000:01:00.0".parse()?;
let selected = resolve(Backend::Cuda, pci)?;
let device = selected.device;
```

`enumerate(Backend::Wgpu)` returns names and available PCI/LUID identifiers. The worker chooses the backend, not the caller of the physical device selector. `resolve` uses Burn's backend-specific enumeration and CubeCL's physical identity, requiring **exactly one match**. No fallback to GPU 0, no matching on model name, no assumed equivalence of CUDA and WGPU indices.

On Windows, some WGPU DX12 adapters expose only a LUID. The SDK asks the OS via `D3DKMTOpenAdapterFromLuid` and `D3DKMTQueryAdapterInfo(KMTQAITYPE_ADAPTERADDRESS)` for its PCI bus/device/function. That query does not report the PCI domain, so the Windows mapping uses domain 0. When an identifier is not discoverable, the request **fails closed**. Windows LUID changes after reboot, and a PCI address can change with PCI topology; neither is a permanent serial number.

If a backend exposes multiple entries for the same physical card, the resolver refuses the ambiguous selection. For machines with virtualized GPUs or driver limitations that hide the physical identity, no attempt is made to guess by GPU brand or memory size.

## RVC

```text
rvc-worker.exe --ipc-stdio --backend cuda --gpu-id pci:0000:01:00.0 --model-id rvc_v2_crazymita --model-dir ... --input-dir ... --output-dir ...
tts-with-rvc-rs.exe list-gpus cuda
tts-with-rvc-rs.exe list-gpus wgpu
tts-with-rvc-rs.exe convert --device wgpu --gpu-id pci:0000:01:00.0 ...
```

The physical GPU ID is optional for backwards compatibility; when provided it is strictly enforced before loading model weights. The RVC worker `status` reports its requested `gpu_id`. `Device::identity()` is additionally captured by the resolver and the returned `ResolvedGpu.info` describes the selected hardware.

## Validation

Offline unit tests cover invalid selectors, exact matching between identical GPU models, duplicates, and no fallback. Linux compilation cannot prove Windows D3DKMT behavior or CUDA/WGPU selection on a multi-GPU Windows machine. The Windows CI checks compilation and those unit tests. Hardware acceptance requires `list-gpus`, positive and negative GPU selectors, and verification that the output was generated on the chosen card on a machine with at least two GPUs.
