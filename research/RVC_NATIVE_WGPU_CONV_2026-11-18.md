# Native RVC convolution on RTX 4060

Windows, full 89.939 s MitaKind input, CrazyMita v2, CPU FAISS 0.75, fixed 300-frame cores, 32 chunks. Cold iteration excluded. Times below cover synchronized resident inference. CUDA uses the existing NSF implicit GEMM. WGPU uses direct NSF. Both use persistent, coalesced, gate-parallel RMVPE GRU. Graph replay is disabled. Each output WAV matches its own backend's PR #2 baseline byte for byte.

| Backend | Batch | Warm seconds | Median seconds |
|---|---:|---|---:|
| wgpu | 2 | 7.788 / 7.822 / 7.822 | 7.822 |
| cuda | 2 | 4.336 / 4.308 / 4.313 | 4.313 |
| wgpu | 8 | 7.227 / 7.230 / 7.259 | 7.230 |
| cuda | 8 | 4.102 / 4.005 / 4.007 | 4.007 |

With the same direct NSF algorithm, commit 43a096a0 measured 10.614 s on CUDA and 11.018 s on WGPU at batch 2. NSF algorithm selection accounts for most of the difference against the fast CUDA configuration. Splitting fully valid spatial tiles from padded edges reduced the WGPU median to 8.087 s. The final implementation additionally shares filters over a larger spatial tile when the device supports it and specializes packed interior address calculations.

The direct WGPU reference for the earlier stage was 37.059 s. The optional native Vulkan compiler build still rejected the requested implicit GEMM tile on this host. Explicit window lowering took 13.483 s. Separate transpose GEMM took 13.906 s. Two polyphase layouts gave approximately 10.7 s and changed the PCM. These alternatives were excluded. AMD hardware was unavailable. Attention, HuBERT frontend and RMVPE U-Net reference fixtures passed on CUDA and WGPU.
