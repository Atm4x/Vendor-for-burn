#ifdef __CUDACC_RTC__
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
#define INFINITY (__int_as_float(0x7f800000))
#else
#include <math.h>
#include <stdint.h>
#endif

__device__ __forceinline__ uint64_t fish_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

extern "C" __global__ void fish_fast_sample(
    const float* logits,
    const int* history_mask,
    int vocabulary,
    int logits_offset,
    int padded,
    float top_p,
    float temperature,
    float repetition_penalty,
    uint64_t seed,
    int codebook,
    int* output_token,
    int* output_column
) {
    extern __shared__ float score[];
    int* token_id = reinterpret_cast<int*>(score + padded);
    __shared__ int invalid;
    int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) invalid = 0;
    __syncthreads();

    for (int i = tid; i < padded; i += static_cast<int>(blockDim.x)) {
        float value = -INFINITY;
        if (i < vocabulary) {
            value = logits[logits_offset + i];
            if (!isfinite(value)) {
                atomicExch(&invalid, 1);
            } else if (history_mask[i]) {
                value = value < 0.0f ? value * repetition_penalty : value / repetition_penalty;
            }
        }
        score[i] = value;
        token_id[i] = i;
    }
    __syncthreads();
    if (invalid) {
        if (tid == 0) {
            *output_token = 0;
            output_column[codebook] = -1;
        }
        return;
    }

    for (int length = 2; length <= padded; length <<= 1) {
        for (int stride = length >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < padded; i += static_cast<int>(blockDim.x)) {
                int peer = i ^ stride;
                if (peer > i) {
                    float left = score[i];
                    float right = score[peer];
                    int left_id = token_id[i];
                    int right_id = token_id[peer];
                    bool before = (left > right) || ((left == right) && (left_id < right_id));
                    bool desc = ((i & length) == 0);
                    if (desc != before) {
                        score[i] = right;
                        score[peer] = left;
                        token_id[i] = right_id;
                        token_id[peer] = left_id;
                    }
                }
            }
            __syncthreads();
        }
    }

    __shared__ double warp_sums[32];
    float maximum = score[0];
    double partial = 0.0;
    for (int i = tid; i < vocabulary; i += blockDim.x)
        partial += exp(static_cast<double>(score[i] - maximum));
    for (int offset = 16; offset; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if ((tid & 31) == 0) warp_sums[tid >> 5] = partial;
    __syncthreads();
    if (tid != 0) return;
    double total = 0.0;
    for (int i = 0; i < blockDim.x / 32; ++i) total += warp_sums[i];
    if (!(total > 0.0) || !isfinite(total)) {
        *output_token = 0;
        output_column[codebook] = -1;
        return;
    }
    double cumulative = 0.0;
    double best = -1.0;
    int selected = -1;
    for (int rank = 0; rank < vocabulary; ++rank) {
        double likelihood = exp(static_cast<double>(score[rank] - maximum));
        cumulative += likelihood / total;
        if (rank != 0 && cumulative > static_cast<double>(top_p)) break;
        uint64_t random = fish_mix64(seed ^ (0xd1b54a32d192ed03ULL * static_cast<uint64_t>(codebook + 1)) ^
                                      (0x94d049bb133111ebULL * static_cast<uint64_t>(token_id[rank] + 1)));
        double uniform = (static_cast<double>(random >> 11) + 0.5) / 9007199254740992.0;
        double q = -log(uniform);
        double weight = exp(static_cast<double>((score[rank] - maximum) / temperature));
        double race = weight / q;
        if (race > best) {
            best = race;
            selected = token_id[rank];
        }
    }
    *output_token = selected;
    output_column[codebook] = selected;
}
