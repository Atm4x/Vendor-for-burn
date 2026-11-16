# Stage117 targeted ResBlock NCU (no recompilation)

Investigate the C128/K11/D5 first/second convolutions of RVC40 GeneratorNSF on RTX4060. Source is the original Git bundle, with Stage115-116 published in PR #2. The analytical stage report is STAGE117_REPORT.md.

NCU kernel-name filter: `regex:preact_residual_temporal_kernel`; matching launch skip = 34; capture 2. NVIDIA documents that launch-skip counts launches matching the kernel filter. This skips NVTX host-range scheduling ambiguity. Capture must be checked against actual launch configuration before asserting the geometry.

From the folder containing `tts-with-rvc-rs.exe` and `embed`, a direct CLI example (input/model paths are local):

```powershell
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe'
$env:RVC_NSF_RESBLOCK_TRANSPORT = '1'
$env:RVC_NSF_RESBLOCK_SHARED_NTC = '1'
$env:RVC_NVTX = '0'
& $ncu --target-processes all --kernel-name 'regex:preact_residual_temporal_kernel' --launch-skip 34 --launch-count 2 --set full --force-overwrite --export '.\stage117_c128_k11_d5' '.\tts-with-rvc-rs.exe' infer-wav-segment-benchmark '.\embed\CrazyMita.pth' '.\embed\hubert_base.pt' '.\embed\rmvpe.pt' '.\embed\CrazyMita.index' '.\embed\MitaKind.wav' '.\stage117_output.wav' 0 3 0.75 0 cuda 2
```

Remove the temporary WAV after the profile and restore prior environment variable values. Preserve the NCU report and logs. For numerical parity and the independent effect of NTC prepacking, use the absolute-path-fixed Stage116 runner with `-Profiler measure -Modes 'baseline,baseline_ntc,transport,transport_ntc' -Runs 5`.

Do not infer performance or numerical correctness from the profiler's replay wall times. The uploaded Stage116 archive lacks PCM parity; no new GPU kernel counters have been measured here.
