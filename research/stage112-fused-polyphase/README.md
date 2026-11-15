# Stage 112: single-launch fused two-tap ConvTranspose1d

The input layouts are NCT with cached prepacked `[Cin,K,Cout/groups]` weights. The operation is *not* an explicit GEMM: each 128-thread CTA calculates 16 output time positions in one stride phase for up to 32 output channels and writes output directly in NTC storage. Input and weight chunks of 32 reduction channels are loaded cooperatively into 12 KiB shared memory with two barriers per chunk.

The algorithm requires **F32**, `kernel_size == 2*stride`, `dilation == 1`, and `output_padding == 0`; unsupported shapes are rejected, not silently approximated. Grouped and odd-channel shapes are covered. Production defaults remain untouched.

`Run-RVC-Stage112-Fused-Polyphase.ps1` compares Direct, ILP4, and fused2tap in the same probe, checks full-output parity *before* NCU, optionally captures selected kernels and matched PyTorch/cuDNN F32/F16 measurements. It includes all tensor copies in synchronized call time but filters Nsight Compute to the compute kernel. The standalone PyTorch F16 result is not numerically interchangeable with the FP32 fused kernel.

Run:

```powershell
.\Run-RVC-Stage112-Fused-Polyphase.ps1 -Backend cuda -Profiler ncu -Runs 12 -NcuLaunchCount 2 -PythonExe 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe'
```

Output: `RVC_STAGE112_FUSED_POLYPHASE_RESULTS.zip`.
