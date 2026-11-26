#ifdef __CUDACC_RTC__
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
#define INFINITY (__int_as_float(0x7f800000))
#else
#include <math.h>
#include <stdint.h>
#endif

__device__ __forceinline__ uint64_t fish_semantic_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float fish_semantic_adjust(float v, int seen, float penalty) {
    float result = seen ? (v < 0.0f ? v * penalty : v / penalty) : v;
    return result == 0.0f ? 0.0f : result;
}

__device__ __forceinline__ uint32_t fish_semantic_order(float value) {
    union { float f; uint32_t bits; } v;
    v.f = value;
    return (v.bits & 0x80000000u) ? ~v.bits : (v.bits ^ 0x80000000u);
}

__device__ __forceinline__ int fish_seen(const int* history, int i, int packed) {
    return packed ? ((static_cast<uint32_t>(history[i >> 5]) >> (i & 31)) & 1u) : history[i];
}

// Summarize vocabulary tiles in parallel. Local normalization keeps every
// exponential bounded when a model emits very large or very small logits.
extern "C" __global__ void fish_semantic_tiles(const float* logits, const int* history,
    int packed, int vocabulary, float penalty, double* tiles) {
    __shared__ float maxima[256];
    __shared__ double mass[256];
    int lane = threadIdx.x;
    int id = blockIdx.x * blockDim.x + lane;
    float score = id < vocabulary ? fish_semantic_adjust(logits[id], fish_seen(history, id, packed), penalty) : -INFINITY;
    maxima[lane] = id < vocabulary && !isfinite(score) ? INFINITY : score;
    __syncthreads();
    for (int n = 128; n; n >>= 1) {
        if (lane < n) maxima[lane] = fmaxf(maxima[lane], maxima[lane+n]);
        __syncthreads();
    }
    mass[lane] = id < vocabulary ? exp(static_cast<double>(score - maxima[0])) : 0.0;
    __syncthreads();
    for (int n = 128; n; n >>= 1) {
        if (lane < n) mass[lane] += mass[lane+n];
        __syncthreads();
    }
    if (!lane) {
        tiles[2 * blockIdx.x] = maxima[0];
        tiles[2 * blockIdx.x + 1] = mass[0];
    }
}

extern "C" __global__ void fish_semantic_sample(
    const float* logits,
    const int* history_mask,
    int packed,
    int vocabulary,
    float top_p,
    float temperature,
    float repetition_penalty,
    uint64_t seed,
    uint64_t semantic_step,
    int eos_id,
    int* result,
    int* output_column,
    int write_column,
    const double* tiles
) {
    __shared__ double sums[256];
    __shared__ int counts[256];
    __shared__ float maxima[256];
    __shared__ uint32_t threshold;
    __shared__ uint32_t midpoint;
    __shared__ int cutoff_id;
    __shared__ double total_mass;
    __shared__ double target_mass;
    __shared__ int invalid;
    int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) invalid = 0;
    __syncthreads();

    float local_max = -INFINITY;
    int tile_count = (vocabulary + 255) / 256;
    for (int tile = tid; tile < tile_count; tile += blockDim.x)
        local_max = fmaxf(local_max, static_cast<float>(tiles[tile * 2]));
    maxima[tid] = local_max;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride; stride >>= 1) {
        if (tid < stride) maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
        __syncthreads();
    }
    if (invalid || vocabulary <= 0 || !isfinite(maxima[0])) {
        if (tid == 0) { result[0] = -1; result[1] = -1; }
        return;
    }
    float maximum = maxima[0];
    // Retain the upper tail per lane. Use it only when it covers the full nucleus.
    constexpr int slots = 12;
    __shared__ uint32_t candidate_keys[slots][256];
    __shared__ double candidate_mass[slots][256];
    __shared__ int compact;
    if (tid == 0) compact = 1;
    __syncthreads();
    int candidates = 0;
    double retained = 0.0;
    double local_total = 0.0;
    for (int tile = tid; tile < tile_count; tile += blockDim.x)
        local_total += tiles[tile * 2 + 1] * exp(tiles[tile * 2] - maximum);
    for (int tile = 0; top_p < 1.0f && tile < tile_count; ++tile) {
        if (tiles[tile * 2] < maximum - 8.0f) continue;
        int i = tile * 256 + tid;
        if (i >= vocabulary) continue;
        float v = fish_semantic_adjust(logits[i], fish_seen(history_mask, i, packed), repetition_penalty);
        if (v >= maximum - 8.0f) {
            double mass = exp(static_cast<double>(v - maximum));
            if (candidates < slots) {
                candidate_keys[candidates][tid] = fish_semantic_order(v);
                candidate_mass[candidates][tid] = mass;
            }
            ++candidates;
            retained += mass;
        }
    }
    sums[tid] = local_total;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        total_mass = sums[0];
        target_mass = sums[0] * static_cast<double>(top_p);
    }
    __syncthreads();
    if (!(total_mass > 0.0) || !isfinite(total_mass)) {
        if (tid == 0) { result[0] = -1; result[1] = -1; }
        return;
    }

    if (top_p < 1.0f) {
        if (candidates > slots) atomicExch(&compact, 0);
        sums[tid] = retained;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (tid < stride) sums[tid] += sums[tid + stride];
            __syncthreads();
        }
        if (tid == 0 && sums[0] <= target_mass) compact = 0;
        __syncthreads();
    }
    // Exact float-score radix threshold: 32 scans, O(32 * vocabulary), not O(V^2).
    if (top_p < 1.0f) {
        uint64_t low = 0, high = 0xffffffffULL;
        for (int pass = 0; pass < 32; ++pass) {
            if (tid == 0) midpoint = static_cast<uint32_t>((low + high + 1ULL) / 2ULL);
            __syncthreads();
            double local_mass = 0.0;
            if (compact && midpoint <= fish_semantic_order(maximum - 8.0f)) {
                local_mass = tid == 0 ? total_mass : 0.0;
            } else if (compact) {
                for (int c = 0; c < candidates; ++c)
                    if (candidate_keys[c][tid] >= midpoint) local_mass += candidate_mass[c][tid];
            } else for (int i = tid; i < vocabulary; i += static_cast<int>(blockDim.x)) {
                float v = fish_semantic_adjust(logits[i], fish_seen(history_mask, i, packed), repetition_penalty);
                if (fish_semantic_order(v) >= midpoint)
                    local_mass += exp(static_cast<double>(v - maximum));
            }
            sums[tid] = local_mass;
            __syncthreads();
            for (int stride = static_cast<int>(blockDim.x) / 2; stride; stride >>= 1) {
                if (tid < stride) sums[tid] += sums[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                if (sums[0] > target_mass) low = midpoint;
                else high = static_cast<uint64_t>(midpoint) - 1ULL;
                threshold = static_cast<uint32_t>(low);
            }
            __syncthreads();
        }

        double above = 0.0;
        int equal = 0;
        for (int i = tid; i < vocabulary; i += static_cast<int>(blockDim.x)) {
            float v = fish_semantic_adjust(logits[i], fish_seen(history_mask, i, packed), repetition_penalty);
            uint32_t key = fish_semantic_order(v);
            if (key > threshold) above += exp(static_cast<double>(v - maximum));
            if (key == threshold) ++equal;
        }
        sums[tid] = above;
        counts[tid] = equal;
        __syncthreads();
        for (int stride = static_cast<int>(blockDim.x) / 2; stride; stride >>= 1) {
            if (tid < stride) {
                sums[tid] += sums[tid + stride];
                counts[tid] += counts[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            uint32_t bits = (threshold & 0x80000000u) ? (threshold ^ 0x80000000u) : ~threshold;
            union { uint32_t bits; float f; } score;
            score.bits = bits;
            double unit = exp(static_cast<double>(score.f - maximum));
            double remaining = fmax(0.0, target_mass - sums[0]);
            // The original cumulative top-p includes a tie exactly on the boundary.
            int allowed = static_cast<int>(floor(remaining / unit));
            if (sums[0] == 0.0 && allowed < 1) allowed = 1;
            if (allowed > counts[0]) allowed = counts[0];
            counts[0] = allowed;
        }
        __syncthreads();
        int wanted = counts[0];
        if (wanted == 0) {
            if (tid == 0) cutoff_id = -1;
            __syncthreads();
        } else {
            int low_id = 0, high_id = vocabulary - 1;
            while (low_id < high_id) {
                if (tid == 0) cutoff_id = low_id + (high_id - low_id) / 2;
                __syncthreads();
                int local_count = 0;
                for (int i = tid; i <= cutoff_id; i += static_cast<int>(blockDim.x)) {
                    float v = fish_semantic_adjust(logits[i], fish_seen(history_mask, i, packed), repetition_penalty);
                    if (fish_semantic_order(v) == threshold) ++local_count;
                }
                counts[tid] = local_count;
                __syncthreads();
                for (int stride = static_cast<int>(blockDim.x) / 2; stride; stride >>= 1) {
                    if (tid < stride) counts[tid] += counts[tid + stride];
                    __syncthreads();
                }
                if (counts[0] >= wanted) high_id = cutoff_id;
                else low_id = cutoff_id + 1;
                __syncthreads();
            }
            if (tid == 0) cutoff_id = low_id;
            __syncthreads();
        }
    } else {
        if (tid == 0) { threshold = 0; cutoff_id = vocabulary - 1; }
        __syncthreads();
    }

    // Independent exponential races. Weights need no global normalization.
    double best = -1.0;
    int best_id = -1;
    for (int i = tid; i < vocabulary; i += static_cast<int>(blockDim.x)) {
        float v = fish_semantic_adjust(logits[i], fish_seen(history_mask, i, packed), repetition_penalty);
        uint32_t key = fish_semantic_order(v);
        if (top_p < 1.0f && !(key > threshold || (key == threshold && i <= cutoff_id))) continue;
        uint64_t random = fish_semantic_mix64(seed ^
            (0xd1b54a32d192ed03ULL * (semantic_step + 1ULL)) ^
            (0x94d049bb133111ebULL * static_cast<uint64_t>(i + 1)));
        double uniform = (static_cast<double>(random >> 11) + 0.5) / 9007199254740992.0;
        double race = exp(static_cast<double>((v - maximum) / temperature)) / -log(uniform);
        if (race > best || (race == best && i < best_id)) { best = race; best_id = i; }
    }
    sums[tid] = best;
    counts[tid] = best_id;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride; stride >>= 1) {
        if (tid < stride && (sums[tid + stride] > sums[tid] ||
            (sums[tid + stride] == sums[tid] && counts[tid + stride] >= 0 &&
             (counts[tid] < 0 || counts[tid + stride] < counts[tid])))) {
            sums[tid] = sums[tid + stride];
            counts[tid] = counts[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[0] = counts[0];
        result[1] = (counts[0] >= 0 && counts[0] == eos_id) ? 1 : 0;
        if (write_column) output_column[0] = counts[0];
    }
}
