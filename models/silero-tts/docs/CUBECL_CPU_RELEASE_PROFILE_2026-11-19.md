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
- Neither run produced PCM within the interval. Only `capabilities` and `load` replied; therefore **full-