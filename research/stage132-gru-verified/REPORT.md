# Stage132 — actual RTX4060 GRU evidence from user's Stage131 capture

Input: `20261117T220950012Z.zip`, full source MitaKind.wav 89.939297 s, 44100 Hz; 3 iterations/cycle, 2 alternating cycles, native inference warm medians exclude first iteration. All ten native benchmark JSONs and eight PCM comparisons are present. This report uses the exact contents of the supplied archive, not estimated kernel timing.

| Mode | Cycle 1 warm ms | Cycle 2 warm ms | Median of cycle medians ms | PCM |
|:--|--:|--:|--:|:--|
| control | 13109.960 | 12335.523 | 12722.741 | original |
| coalesced | 12544.403 | 11941.797 | 12243.100 | bit-exact |
| gate_parallel | 11861.004 | 11959.521 | 11910.262 | bit-exact |
| split2 | 12601.393 | 11848.330 | 12224.861 | 68.294 dB, reject |
| split4 | 12015.846 | 11834.976 | 11925.411 | 67.591 dB, reject |

Both gate-parallel cycle comparisons favor gate-parallel; median-of-two warm times is ~6.39% below control. This is encouraging but does not establish steady speed under every workload or prove that all the wall-time improvement comes from GRU.

Four NSYS SQLite files passed `PRAGMA quick_check` even though the long profiling run exhausted disk. For each supported path, `stage102_rust_measured` delimits one warm full-WAV inference; known GRU launch geometry from NCU and the NSYS name summary identifies precisely 17 launches. Warm GRU GPU kernel sums:

| Mode | GPU kernel ms / 17 warm launches | GPU speedup vs control |
|:--|--:|--:|
| control | 488.643 | 1.00× |
| coalesced | 117.497 | 4.16× |
| gate_parallel | 65.723 | 7.43× |
| split2 | 76.589 | 6.38×, numerical gate FAIL |

Nsight Compute `.ncu-rep` data was not lost: the user's `ncu_report` exporter recovered 2 launches and ~1676 metrics per valid kernel. Control vs gate-parallel short-segment NCU: `gpu__time_duration.avg` 37.801 / 36.274 ms vs 5.510 / 4.781 ms; `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 125,890,752 vs 15,790,272; 40 registers/thread in both; 256 vs 768 threads/block; SM active warps 16.67% vs 50.01% of peak. These NCU numbers are replay durations, not full-WAV elapsed time.

**Failure attribution:** four split-K PCM rejects were valid. `NCU bad conversion` affected CLI export only; SDK action data is intact. The `split2.nsys.sqlite` file is SQLite-valid but has zero StringIds due to incomplete export; its timing and geometry survive. `split4` NSYS failed before SQLite creation. Disk-full errors came from multiple full-90-second NSYS exports and ZIP packaging. Gate-parallel data itself is intact.

**Next performance priority:** keep gate-parallel as an opt-in candidate for broader RTX acceptance. The ~423 ms GRU GPU-time saving in the captured warm execution is substantial but still far below the remaining full-RVC ~12 s; direct attention now belongs on NSF ResBlocks, with per-shape CUDA profiles rather than further split-K experimentation.
