# Stage 33 — NSF Conv1d arithmetic R&D (CPU reference, CUDA design)

Date: 2026-11-10. Status: experimental, not enabled for production inference.

## Evidence from supplied PyTorch source

Inspected `aten/src/ATen/native/Convolution.cpp`, `ConvolutionMM2d.cpp`, `mkldnn/Conv.cpp`, and `cudnn/Conv_v8.cpp` in the user-supplied `pytorch-main.zip`. PyTorch dispatches suitable CPU convolutions into oneDNN/MKLDNN, and NVIDIA GPU convolutions into cuDNN. Its generic 1D convolution dispatch can view 1D as a 2D operation. This demonstrates that PyTorch's CPU performance is not obtained by executing convolution loops in interpreted Python. It does not establish that a single algorithm is optimal on all devices.

## Reduction identity

For cross-correlation with a three-tap filter, two neighboring output positions require six scalar products per input/output channel pair in the direct representation. Let `a,b,c,d` be four consecutive input samples, and `u,v,w` three filter coefficients. Define

- `p0 = u*(a-c)`;
- `p1 = ((u+v+w)/2)*(b+c)`;
- `p2 = ((u-v+w)/2)*(c-b)`;
- `p3 = w*(b-d)`.

Then exactly in real arithmetic:

- `y_even = p0+p1+p2 = u*a+v*b+w*c`;
- `y_odd = p1-p2-p3 = u*b+v*c+w*d`.

Thus there are **four multiplications instead of six** per two outputs (33.3% fewer), at the cost of transformations, additions and temporary storage. This is the established Winograd F(2,3) identity, not a claim to have invented Winograd. The implementation and tile/weight-plan engineering here are independent experimental work.

For any odd K, partition taps into groups of three, zero-extending the last incomplete group. Combine each segment's input-channel dimension with segment index into one GEMM reduction, for four transformed phases. For K=11, four segments lead to 16 products per pair of output positions, versus 22 in direct convolution (**27.3% fewer**). For dilation D, output locations are paired within each modulo-D lane; the code gathers these lanes into a shared spatial dimension for four GEMMs per tile. This preserves convolution semantics for stride=1 with symmetric same padding, arbitrary positive dilation, finite FP32 weights. The implementation is not general Conv1d; it deliberately targets NSF ResBlocks.

## Actual measurements

All times are from standalone Rust `-C opt-level=3 -C target-cpu=native` using the same `gemm` version as vendored Burn and one GEMM thread. Pretransformed weights are held across forwards. Timings include output creation, input transformations and kernel work; they exclude checkpoint loading and the one-time weight transformation. CPU scheduling noise varies between trials. These are **microkernel** results, not whole-model speedups.

| Real saved PyTorch activation / checkpoint weight | Direct GEMM | Winograd | Best observed speedup | Max difference vs PyTorch output |
|---|---:|---:|---:|---:|
| `resblocks.0.convs1.0`: 256→256, K3, D1, L240 | 0.960 ms | 0.752 ms | 1.28× | 4.29e-6 |
| `resblocks.5.convs1.2`: 128→128, K11, D5, L2400 | 8.983 ms | 8.559 ms | 1.05× | 1.06e-5 |
| `resblocks.11.convs2.2`: 32→32, K11, D1, L9600 | 2.694 ms | 3.645 ms | **0.74× (regression)** | 3.3e-6 |

Independent K3 plan benchmark showed that recomputing weight transforms each forward largely erases its speed benefit. The plan must therefore be persisted per loaded checkpoint. `nsf_winograd_f23.rs` tests cover different channel counts, even/odd output lengths and tiles, including the last incomplete output pair. `nsf_segmented_winograd.rs` checks K3/5/7/11 and dilation 1/2/5 on deterministic inputs. Each test suite passed locally.

A separate one-thread Python 2.10 CPU reference on the same real K3 activation had ~1.09 ms median with MKLDNN, but CPU contention/measurement scope differ; do not present this as end-to-end PyTorch superiority.

## Low-rank viability study

For trained weights reshaped `[Cout, Cin*K]`, 99% Frobenius energy requires rank 181/256 at K3 (256 channels), 119/128 at K11/D5, and 23/32 at K11/D1. A two-layer factorization with these ranks offers little or no arithmetic saving at the 99% energy threshold. Energy preservation itself does not guarantee waveform quality. No approximate weights were adopted.

## CPU / CUDA boundary

The algebraic transform is device-independent. A CUDA implementation must pretransform weights once on the device, fuse input transforms/gathers with blocked GEMM input loading, and fuse the output transform and bias into the store. Four separate GPU kernel launches per tile could negate the arithmetic advantage. The same is true for long K11 layers: measured CPU overhead outweighs reduced products, so dispatch decisions must be grounded in measured full-operator latency. There was **no CUDA execution** in this stage.

## Integration and next gates

1. Add immutable, backend-owned packed-weight plans at model load (not per forward and not global cache by pointer). Keep a canonical weight representation and explicitly invalidate packed plans if weights change.
2. Prototype a single launch fused input-transform → matrix microkernel → output-transform for K3, then measure on the real 256-channel stage. Compare to native Burn and PyTorch on matched CPU threads and warmups.
3. For K7/K11, investigate blocked output-stationary microkernels that reuse the full input tile across output channels; segmented Winograd as written is *not* a production optimization.
4. Separately implement/measure a CUDA device kernel (CubeCL or CUDA integration) with proper FP32/tensor-core accuracy. Do not infer CUDA performance from the CPU results.
5. Before enabling any kernel in NSF, run full checkpoint parity, source-noise control, multi-chunk WAV→WAV and whole-model time/memory benchmarks. Existing production path remains unchanged.

This work introduces no fallback routing in the production Conv1d/Conv2d code, no modifications to legacy scripts, and no private model/audio uploads.
