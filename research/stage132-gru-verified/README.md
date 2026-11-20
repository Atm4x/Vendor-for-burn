# Stage132 — validated GRU performance, low-storage NSYS/NCU

The Stage131 RTX4060 run contained ten completed, synchronized 89.939 s benchmark executions and eight PCM comparisons. Later profile export failed after the Windows volume filled; the benchmark results were not lost.

The **production candidate remains opt-in**: persistent RMVPE GRU with coalesced recurrent weights and three parallel gates. Stage132 changes no GRU math or weight layout; it adds the synchronized warm NVTX range to the short-segment benchmark and corrects the profiling harness.

After extracting the Stage132 Windows CI artifact beside the `embed` directory, run:

    .\Run-Stage132-GRU-AB.ps1 -Runs 3 -Cycles 2 -Profiler all

Default paths are inherited from Stage131 (`tts-with-rvc-rs.exe`, `embed/MitaKind.wav`, `embed/CrazyMita.pth`, `hubert_base.pt`, `rmvpe.pt`, `CrazyMita.index`; local NVIDIA Nsight installations and `python`). Override the supplied path parameters if installed elsewhere.

The full ~90 s WAV is benchmarked **without profiler** in alternating control/coalesced/gate-parallel order. WAVs are bit-exact compared to each cycle's control. Profiler captures are separate, using a 3-second segment and two iterations, with only the second inside the new `stage102_rust_measured` NVTX interval. Thus NSYS confirms GRU GPU time and CUDA geometry on that segment; its number must not be substituted for the entire 90 s GRU workload. NCU targets the same GRU family and uses hardware replay; its duration is not a warm full-WAV time.

The new NCU exporter uses NVIDIA's `extras/python/ncu_report` API, not the CLI raw CSV, which previously printed `bad conversion` and produced headers without rows. SQLite files must pass a quick integrity check and contain one warm GRU launch in the short-segment capture. Missing/mismatched kernels fail closed. By default, rejected split-K variants (below 70 dB output SNR in Stage131) are not profiled.

Storage: Stage131 profiled five 90 s WAVs with NSYS, generating >100k kernel events per mode and exhausting disk space. Stage132 collects the compact segment separately from the full benchmark, checks >=2 GiB available before profiling, removes generated WAVs and uses streamed ZipArchive packaging. If any profiler fails, it still retains evidence and raises a failure rather than claiming success.
