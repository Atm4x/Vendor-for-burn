# Stage 83 — RVC algorithm redesign, not another RMVPE cache

Date: 2026-11-13. Parent: local verified Stage81 (`34ec73ad`, corresponding to PR #2 published Stage81). No Stage82 unfinished source is included. All new behavior is experimental and opt-in.

## Start with the actual computational graph

The original Python reference (`tts-with-rvc.zip`, `tts-with-rvc-main (2).zip`) and the Rust full-WAV CLI do not use the same sequence geometry. Original Python config on an RTX4060 uses `is_half=True`, `x_pad=3`, `x_center=60`, `x_max=65`; it performs global 16-kHz high-pass filtering and F0 before chunked VC. Rust defaults to a 3.0-second core and approximately 0.4s context per side, tensor-batch 2, 32 chunks for the 89.94s test. This is **not** numerical parity and a direct WAV comparison across schedules is invalid. Do not claim the original Python latency until the pinned Stage80 PyTorch runner has passed on the user's CUDA environment.

Real RTX4060 Stage68 synchronized profile (14.014s, combined NSF GEMM + persistent GRU, 32 chunks; diagnostic synchronizations change runtime):

| Nonoverlapping phase | Observed time |
|---|---:|
| NSF | 5.164s |
| HuBERT | 3.109s (frontend Conv1d 2.612s) |
| RMVPE | 3.068s (UNet 2.439s) |
| TextEncoder | 1.830s |
| Flow | 0.528s |
| FAISS/pitch protection | 0.145s |

**Implication:** optimizing only attention cannot yield a 2–3x end-to-end speedup at the current 3s geometry. The optimistic Amdahl contribution of the entire TextEncoder plus non-frontend HuBERT is at most ~2.33s/14.01s (16.6%). The *structural* purpose of this cycle is to remove the quadratic activation-memory barrier and make Python-scale windows feasible for future investigation. This does not mean long windows are automatically faster: attention compute still grows as O(T²), and other layers/RVC random generation have different semantics.

## Architectural obstacle discovered

`hubert_transformer.rs` materializes `[B,H,T,T]` scores per layer; `attention.rs` materializes dense relative key and value expansions `[1,1,T,T,D]` even though the RVC relative embedding is active only for `|i-j| <= window`. At T=6000, H=12, FP32 HuBERT scores alone occupy 1.61 GiB; one relative key or value with D=64 occupies 8.58 GiB. At T=380, these same representative tensors are only 6.6 and 35.3 MiB respectively. Numbers describe individual intermediate buffers and do not predict peak resident GPU memory or complete runtime.

## New algorithm implemented: online softmax with sparse relative correction

For query q_i, key k_j and value v_j, the original relative model has

`score(i,j)=q_i·k_j + 1{|i-j|<=w} q_i·rK[j-i]`, and
`numerator(i)=Σ_j exp(score(i,j)) (v_j+1{|i-j|<=w} rV[j-i])`.

The transformed evaluation streams over finite key tiles and maintains `(m,l,o)` for each query row. For each key tile with scores `s`, define `m'=max(m,max(s))`, `a=exp(m-m')`, and `p=exp(s-m')`. Then

- `o' = a o + Σ(p_j*(v_j+rV_ij))`
- `l' = a l + Σ p_j`
- `m'` is retained, output is `o/l` after all tiles.

This is algebraically equivalent to dense softmax in real arithmetic and numerically stable; floating-point reassociation means **bitwise equality is not expected**. Relative key/value tensors are **never created** for off-diagonal key tiles that cannot overlap the width-`w` band. Computational work for position terms scales as `O(T*w*D)` plus tile-boundary overhead, not dense `O(T²*D)`. Standard absolute global attention still requires `O(T²*D)` arithmetic absent a change to the attention model; workspace decreases to `O(B*H*Qtile*Ktile + Qtile*Ktile*D)` per tile plus Q/K/V. This is **not** hidden approximate attention or dynamic padding tricks.

Source: `models/tts-with-rvc-rs/src/online_attention.rs`; wired into both HuBERT and TextEncoder through `RVC_ATTENTION_ONLINE=1`. Production dense path remains unchanged.

## Native accelerated branch: CubeCL FlashAttention for HuBERT

Inspection of vendored `burn-cubecl-0.22.1` found a full CubeK FlashAttention kernel. With `autotune` disabled, its standard strategy defaults to the dense fallback. `RVC_HUBERT_ATTENTION_FLASH=1` now selects the native CubeCL FlashAttention **directly** with a unit routine; unsupported shapes fail explicitly instead of silently running the dense fallback. HuBERT is unpadded in the current full-inference path; nontrivial key padding is rejected by this experimental branch. Query vectors are passed unscaled, letting the native kernel perform `1/sqrt(head_dim)` exactly once. This is a genuinely fused GPU kernel candidate, unlike chaining many small Burn tensor operations.

Using both flags yields FlashAttention for HuBERT and streamed relative attention for TextEncoder. No changes to model weights, RMVPE, NSF, F0, chunk scheduler or index search. The source remains compatible with non-Flash CPU and WGPU execution when the flags are unset.

## Research hypotheses and stop gates

1. CPU parity against dense/reference attention: max abs error on representative masked and relative fixtures <=1e-4; evaluate separate numeric and model-end-to-end errors.
2. CUDA execution: first run may JIT compile; report cold vs warm separately. Fused kernel must not silently degrade to dense.
3. Compare full 89.94s WAV with identical geometry and options; check F0 threshold flips, PCM16 RMSE, samples and waveform hashes. Do not require bitwise equality for regrouped FP math.
4. Benchmark ABBA order, not a single baseline-before-candidate pass, and inspect GPU clock/utilization.
5. Only then test 600 and 900 core frames. Large core may be slower because attention arithmetic remains quadratic. Do not enable `core=6000` until the existing 1000-frame caps, memory and model-specific equivalence have been addressed together.

Windows runner: `research/stage37-windows-gpu-bench/windows/Run-RVC-Full-Stage83.ps1`. Modes `baseline`, `online`, `flash`, `hybrid`, `all`. `all` uses ABBA-like order `baseline online flash hybrid hybrid flash online baseline` (two runs per process; compare time medians by mode). Results `RVC_STAGE83_FULL_WAV_RESULTS.zip`.

## Next architecture milestones (not implemented in this stage)

- Establish successful original Python FP16/FP32 baseline with the pinned Stage80 script. Compare model input geometry, F0 alignment, stochastic seed and final waveform separately.
- Model total inference latency as a function of segment length and attention tile geometry; derive/measure an optimal memory-constrained partitioning scheme instead of assuming the longest possible window wins.
- The *measured dominant* cost remains NSF high-rate residual convolutions (~5.16s Stage68); its next serious algorithmic candidate is a **fused, channel-tiled high-rate residual execution engine** rather than one more GEMM toggle. It must preserve six serial nonlinearities, all three branches, boundary padding and deterministic noise schedule.
- If a 2x+ end-to-end improvement is the goal, it will require attacking the NSF, HuBERT frontend and RMVPE together and/or eliminating unnecessary windowed inference at the model/schedule level. Attention alone has insufficient Amdahl headroom.
