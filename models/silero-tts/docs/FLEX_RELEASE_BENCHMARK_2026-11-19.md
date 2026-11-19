# Silero V5 RU — Burn Flex CPU release benchmark

Date: 2026-11-19. Real weights: `v5_ru.pt` exported to SafeTensors. Rust 1.100.0, Burn 0.22.1, `--release --no-default-features --features reference-flex`; binary reports only `flex` available. CPU: AMD EPYC 9V74 virtualized Linux, cgroup quota **4 CPU cores** (`cpu.max: 400000 100000`), 5 visible vCPUs. PyTorch reference: 2.10.0+cpu, 4 Torch threads.

The Rust `elapsed_ms` timer includes forward inference and encoding samples into Base64 in the JSONL worker; it excludes model loading and subsequent JSONL transport. External wall-clock time includes JSON framing and transmission. PyTorch timing measures `apply_tts`, not Base64 serialization. All times are in seconds unless noted.

**RTF = inference wall time / generated-audio duration**; lower is faster. Throughput = 1 / RTF (seconds of audio processed per second).

## Rust Flex release — worker measurements

| Case | Generated audio | Worker time | RTF | Speed |
|---|---:|---:|---:|---:|
| one-char-8k #1 | 0.3250 s | 0.066 s | 0.20308 | 4.92x |
| one-char-8k #2 | 0.3250 s | 0.058 s | 0.17846 | 5.60x |
| one-char-8k #3 | 0.3250 s | 0.058 s | 0.17846 | 5.60x |
| one-char-8k #4 | 0.3250 s | 0.059 s | 0.18154 | 5.51x |
| one-char-24k #1 | 0.3250 s | 0.059 s | 0.18154 | 5.51x |
| one-char-48k #1 | 0.3250 s | 0.059 s | 0.18154 | 5.51x |
| phrase-8k #1 | 0.9875 s | 0.120 s | 0.12152 | 8.23x |
| phrase-8k #2 | 0.9875 s | 0.116 s | 0.11747 | 8.51x |
| Long Russian phrase, Flex #1 | 5.475 s | 0.399 s | 0.07288 | 13.72x |
| Long Russian phrase, Flex #2 | 5.475 s | 0.407 s | 0.07434 | 13.45x |
| Long Russian phrase, Flex #3 | 5.475 s | 0.412 s | 0.07525 | 13.29x |

Model load (short test): **1.279 s**. Model load (long test): **1.292 s**. Reuse confirmed: **8 successful short/medium calls** in a single process with no reload; clean worker exit code 0.

## PyTorch 2.10 CPU reference

Long sample: `Привет, мир! Привет, мир! Привет, мир! Привет, мир! Привет, мир! Привет, мир!`; 77 characters; sample rate 8 kHz; generated 43800 samples = 5.475 s.

- Flex warmed median (last two runs): **0.4095 s**, RTF **0.07479**, throughput **13.37x**.
- PyTorch warmed median (last two runs): **0.20325 s**, RTF **0.03712**, throughput **26.94x**.
- Flex/PyTorch latency ratio: **2.01x**.
- Long audio length parity: **exact**. Correlation: **0.999939962**. Max abs error: **0.0698864**, at roughly 0.651 seconds of the result; approximately **2.24%** of samples exceed max-abs threshold 0.003; RMSE **0.0015904**. This fails strict max-abs threshold 0.003, so full long-utterance numerical parity is NOT yet established.

## Short-sample PyTorch parity

| Case | Audio samples match | Max absolute PCM error | Correlation |
|---|---|---:|---:|
| one-char-8k | yes (2600) | 0.0001508 | 0.999999995 |
| one-char-24k | yes (7800) | 0.0000980 | 0.999999993 |
| one-char-48k | yes (15600) | 0.0001471 | 0.999999995 |
| phrase-8k | yes (7900) | 0.0014248 | 0.999999468 |


## Interpretation and limitations

1. Flex release demonstrates end-to-end TTS faster than real time without CubeCL LLVM JIT. The earlier roughly 4.1 s Flex measurements were not release measurements and should not be used as production RTF.
2. The original published Silero CPU throughput is not directly comparable: this environment has an AMD EPYC shared virtual CPU quota and the tests use distinct lengths and Python/Torch versions.
3. Worker timing includes inline Base64 encoding; raw tensor backend kernel timings were **not measured** in this experiment.
4. On the repeated long phrase, float32 sample errors peak in limited regions (max ~0.07; correlation ~0.99994). The discrepancy could originate from frontend or accumulated numeric differences. Attribution requires layer-by-layer parity; no root cause is established.
5. All results were measured locally; no CUDA, WGPU or CubeCL operations were invoked by the Flex-only binary.

## Reproducibility

The evidence ZIP contains `benchmark_release_flex.py` (standalone reusable JSONL worker benchmark), the actual JSON raw results, PyTorch parity script and results, worker log and WAV examples. No model weights or vendored dependencies are included. Build and performance scripts have no persistent changes to the CubeCL backend.
