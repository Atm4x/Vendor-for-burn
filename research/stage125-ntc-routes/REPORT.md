# Stage125 — evidence-based NTC dataflow route experiment

## Measured basis

User-supplied Irodori-TTS-burn-master and RTX3060Ti report: a packed dilation-residue plus D1 route reduced two K7 C192 T48000 D3/D9 FP32 convolutions from 11.693 to 7.914 ms (1.477x) on another GPU. It included a pack kernel. This is NOT evidence of an RVC speedup. Dashmag's register sliding window is depthwise Conv1d and does not transfer to our dense C128/K11 reduction.

Our Stage121 **fast control** NCU: first/second C128/K11 ~4.241/5.239 ms, L1 sectors ~211 million, high L1 hit, issue active ~68-74%. Stage118 dilation D1/D3/D5 times were comparable. Thus memory packing/residue penalties are not established for RVC. Do not reintroduce rejected implicit GEMM, shared-memory staging, Stage120 weight prefetch or Stage121 warp broadcast.

## Routes and guards

Control: Stage123 activation placement (producer emits LeakyReLU-activated value), transport and shared NTC, temporal tile4, persistent GRU. Linear: Stage124 fast-in-bounds kernel and linear progression of packed weight/input addresses; falls back to original code at boundaries. Residue: compile-time C128 K11 D>1 first Conv1d only, with Stage123 output_activated=true and input_activated=false, changes work-item time coordinates from consecutive 4 positions to four positions with step=dilation. It writes to the ORIGINAL output time positions, does not materialize residue packing and adds no CUDA kernel. Combined: linear+residue. Optional legacy: control without Stage123 activation.

All routes preserve the per-output K outer/Cin inner FP32 FMA order, packed [K,Cin,Cout], current NTC and output buffers. For T=29800, tile4: normal time tiles=7450, residue-order D3=7452 and D5=7450. Potential cache changes have not been proven to improve runtime.

## Verification

Local Rust 1.100.0 reference: 3/3 PASS for exact time permutation and both convolutions including padding, D3/D5, T up to 29800 for bijection, Cin/Cout up to 128 for FP32. Stage124 independent linear-interior reference: 2/2 PASS. Local offline cargo check with cpu,cuda,wgpu after restoring Stage123 source and applying both opt-in routes: PASS, including cubek-convolution, burn-cubecl, tts-with-rvc-rs. DOCS_RS=1 is used only locally to avoid offline LLVM-bundler download. GPU JIT, RTX-4060 correctness, SASS, Nsight counters and speed remain unmeasured.

Windows CI must compile source and package **one** artifact, without GPU. User tests all modes using one PowerShell runner, verifying strict PCM identity before profiling, and uploads resulting single ZIP. Review full NSYS timeline, launch geometry, NCU issue/branch/address work, registers, occupancy, L1/TEX and per-kernel GPU times. Reject slower routes. No claims of improvement until those results arrive.
