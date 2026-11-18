# Silero V5 RU — CubeCL CPU release profiling and CSE A/B

**Date:** 2026-11-19  
**Reference:** `Atm4x/Vendor-for-burn`, `codex/silero-tts-cubecl-port-20261118`, Git baseline `37eea4447f2833d909c651ad4f9bad03368d4ed1`  
**Workload:** real `v5_ru.pt` weights exported to SafeTensors, Russian input `а`, `aidar`, `8000 Hz`, one persistent model instance  
**Backends:** Burn `0.22.1`, CubeCL CPU/LLVM `0.11.3`, Rust `1.100.0`, Tracel LLVM `23.1.0-4`; **no Flex**  
**Sampling:** `perf record -e cpu-clock:u -F 99`, user CPU, fixed 38-second wall-clock time limit, self samples, no call graphs (kernel restrictions)

## Release benchmark controls

- `cargo build --release --offline --no-default-features --features cubecl-cpu --bin silero-tts` completed with debug symbols included for profiling (`CARGO_PROFILE_RELEASE_DEBUG=1`). A clean release build took **6m10s** on the available 5-vCPU Linux environment; this is **build time, not inference latency**.
- Restored the latest Silero development branch via checked Git Bridge artifact and verified the SHA against the actual GitHub HEAD at export. All 389 model tensors were exported from the user's `v5_ru.pt` using the existing developer tool.
- The checked-in vendored crate contents contain 431 line-ending checksum discrepancies and one additional checksum discrepancy. Build-only local fixes were required and **were not committed**.
- A local **diagnostic-only** modification to `vendor/cubecl-llvm-0.11.3/src/shared/base.rs` guarded `SimpleCSEPass::with_memory()` with a runtime environment check for `SILERO_SKIP_SIMPLE_CSE`. In the default condition the original pass ran unchanged. The modified compiler and all dependent crates were rebuilt. Both controlled A/B runs used **the same patched release binary** and differed only by the environment variable.
- The input commands were `capabilities`, `load`, `infer` (accented text), `status`, `shutdown`. A watchdog sent SIGINT to the entire profiling process group after 38s and prevented unbounded runs. The profiler itself was smoke-tested with a 5s limit before the final A/B.
- Neither run produced PCM within the interval. Only `capabilities` and `load` replied; therefore **full-inference latency, final numerical parity, and a speedup factor are not established** by this A/B.

## Controlled 38-second A/B (one release binary)

| Metric | SimpleCSE enabled | SimpleCSE skipped |
|---|---:|---:|
| Actual wall time | 38.082 s | 38.068 s |
| CPU sampling events | 6,675 | 10,323 |
| Lost samples | 0 | 0 |
| JIT DSO share | 50.91% | 77.98% |
| `silero-tts` binary DSO share | 41.17% | 20.84% |
| libc share | 7.79% | 1.10% |
| `ScopedMap::get` (SimpleCSE) self share | 5.21% | 0% |
| `ScopedMap::insert` (SimpleCSE) self share | 3.10% | 0% |
| `DispatcherWorker::work` self share | 8.48% | 7.23% |
| Pliron `strictly_precedes_in_block` self share | 3.04% | 0.36% |
| Worker `load` response | PASS | PASS |
| Worker `infer` response by 38s | No | No |

**Interpretation:** avoiding SimpleCSE removes its own work (8.31% of baseline self samples in `get`/`insert`) and changes the share of samples in generated JIT kernels. A larger JIT share is **not** proof of increased inference throughput. Without a completed inference, intermediate progress counters, or an execution-time comparison for an equivalent completed graph, it is not correct to report a 1.5x speedup merely because more CPU samples were collected. The skipped pass may also increase work inside kernels by preserving redundant expressions. Do not enable the bypass in production on this evidence.

## Independent first 40-second optimized-build profile

An earlier non-A/B release run of 40s collected 7,322 samples: JIT 51.31%, `silero-tts` 41.66%, libc 6.83%; `ScopedMap::get` 5.30%, `ScopedMap::insert` 2.83%, dispatcher 10.22%. This is consistent with the later controlled release baseline. In contrast, the older **debug** profile showed 51.1% in Pliron routines and 9.4% in SimpleCSE/ScopedMap, reflecting the very different costs of an unoptimized executable. Never compare the older 509.9s debug inference to release latency directly.

## Current performance suspects, in priority order

1. **Generated CubeCL CPU JIT kernels and their scheduler.** They account for about half of samples in the release baseline, and close to four-fifths when CSE is skipped. The unsymbolized JIT mappings must be tied to kernel names, operation shapes, dispatch counts, generated code and thread-pool activity before assuming which tensor operator is slow.
2. **SimpleCSE/Pliron optimizer.** The map uses a `Vec` of scope entries, with linear equality-based lookup. CSE consumes measurable CPU time even in release. Evaluate a hash/index strategy for larger scopes, or scoped caching, separately from permanently disabling CSE; protect semantic parity.
3. **`SameIstft` overlap-add algorithm.** This is a separate, structurally inefficient path built from repeated tensor slices and `slice_assign`. A gather-style CubeCL output kernel with overlap normalization is appropriate, but this particular 38s sample does **not** isolate its execution time because inference did not finish.
4. **Cold JIT and kernel cache behavior.** Measure a warm second request after one full inference and report compiler time, optimized IR time, kernel launch count, worker thread idle time, and kernel execution independently.

## Reproduction

The committed `tools/profile_cpu_40s.py` takes `--binary`, `--model-dir`, `--perf`, `--output`, and `--seconds` (defaults to 40). It writes `cpu.data`, `worker.jsonl`, `summary.json`, `dso.txt`, and `symbols.txt`, and terminates the profiled process at the bounded wall-clock limit.

Example baseline:

```bash
python tools/profile_cpu_40s.py \
  --binary /path/to/target/release/silero-tts \
  --model-dir /path/to/exported-silero-weights \
  --perf /path/to/perf \
  --output ./profiles/release-baseline \
  --seconds 38
```

The `--skip-cse` switch sets a diagnostic environment variable that has **no effect on an ordinary, unmodified CubeCL build**; it works only when the local experimental guard described above is present. The standard source tree must retain the original pass.

## Acceptance gate for further optimization

Before reporting a performance improvement: complete inference and validate PCM against the PyTorch reference, compare the same input and device with a cold/warm split, and collect timestamps at individual model stages. A fixed-duration sampling trace alone is for **bottleneck identification**, not a successful-inference benchmark.
