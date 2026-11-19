# Fish Speech 1.5 / CubeCL CONT39 — backend-neutral AR stream contracts

Date: 2026-11-19. Development branch: `codex/fish-speech`.

## Scope and outcome

CONT39 completes the remaining source-level C03 structural criterion: the stream coordinator, the buffer-binding identifiers, and KV-copy barriers are no longer restricted to TensorRT network objects. It does **not** claim CubeCL kernel numerical parity or GPU performance improvements. The CubeCL graph runtime and existing CPU/TensorRT generation routes remain separate actual execution implementations.

- `core/src/stream_binding.rs`: stable backend-neutral `Binding::{Value,Temp}`. `trtx_blueprint.rs` re-exports the same identity for compatibility.
- `core/src/ar_stream_plan.rs`: `ArNetworkSegment` contract (segment preflight, required external inputs, typed outputs); `ArStreamPhase<N>` and `ArStreamPlan<N>` can carry any backend's network type, preserving positional BF16 K/V validation, output bindings, and submission barriers. The existing `from_graph` TensorRT lowering remains the adapter for `TrtBlueprint`.
- `core/src/ar_stream_executor.rs`: `ArStreamDevice<N>` and `ArStreamExecutor<D,N>` compile and execute independently of the network representation; common scratch accounting, preflight of all network inputs, error poison/reset, stream ordering, KV copies and retention remain in one executor. The generic type parameter on the executor has a TensorRT default **only for source compatibility** with existing callers.
- `src/cuda_stream.rs`: explicit `ArStreamDevice<TrtBlueprint>` adapter; no change to its allocation, dispatch, stream or runtime behavior.
- New Core fake-network test converts a known stream plan to a separate `OpaqueNetwork` type, then checks no submissions on missing inputs, successful segment enqueue/KV-copy sequence, active KV-prefix growth and cleanup.

## Verification

| Test | Result |
| --- | --- |
| Core library | 207 passed, 2 ignored |
| Default library | 58 passed, 2 ignored |
| Joint CUDA + TensorRT + CubeCL `cargo check` | PASS |
| Feature CubeCL library | 72 passed, 2 ignored |
| Formatting / diff hygiene | PASS |
| Numeric CubeCL runtime and Windows/RTX | Unverified: host environment lacks the runtime necessary for device parity |

## Progress, fixed denominator 26

| Track | Complete | Left to implement or verify |
| --- | ---: | --- |
| C01 base executor | 3/4 | Actual kernel parity |
| C02 Transformer/BF16/GQA | 5/6 | Runtime-executed numerical parity |
| C03 AR contracts/runtime | 4/4 | Source-level contract complete; real GPU execution still unverified |
| C04 sampling / TTS | 0/4 | Native device sampling and end-to-end generation trace |
| C05 Firefly GPU | 0/4 | FSQ, ConvNeXt, HiFiGAN, parity |
| C06 RTX acceptance/optimization | 0/4 | Actual RTX launch, checkpoint parity, benchmarks, optimization |
| **Total** | **12/26 = 46.2%** | **14 gates** |

## Separate optimized TensorRT branch

The user supplied `codex/fish-speech-tensorrt` (HEAD `e7d6e1a57fefda44f07c78621fb231934eb3230d`). CONT39 began from `codex/fish-speech` (`993781d404e48b85bfbfe72112878a29a4a2fc1a`). The two heads differ. No diff was fetched via the GitHub connector or web, and **no merge was attempted** in this cycle: a trustworthy integration requires local Git tree-by-tree comparison after obtaining the exact revised TensorRT branch. Do not overwrite or replace the faster TensorRT implementation on an assumption of compatibility.

## Next

Start C04 with an actual device-resident sampling interface for semantic and acoustic logits. The current CubeCL generator returns logits to CPU. For numerical acceptance, obtain a running CubeCL runtime (CPU with Tracel LLVM 23.1, or Windows NVIDIA CUDA) and compare each operator against the CPU/Python reference before claiming parity.
