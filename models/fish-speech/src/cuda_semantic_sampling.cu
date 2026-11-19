#include <math.h>
#include <stdint.h>

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

extern "C" __global__ void fish_semantic_sample(
    const float* logits,
    const int* history_mask,
    int vocabulary,
    float top_p,
    float temperature,
    float repetition_penalty,
    uint64_t seed,
    uint64_t semantic_step,
    int eos_id,
    int* result
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
    for (int i = tid; i < vocabulary; i += static_cast<int>(blockDim.x)) {
        float v = logits[i];
        if (!isfinite(v)) {
            atomicExch(&invalid, 1);
            continue;
        }
        float adjusted = fish_semantic_adjust(v, history_mask[i], repetition_penalty);
        if (isfinite(adjusted) && adjusted > local_max) local_max = adjusted;
        if (!isfinite(adjusted)) atomicExch(&invalid, 1);
    }
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
    double local_total = 0.0;
    for (int i = tid; i < vocabulary; i += static_cast<int>(blockDim.x)) {
        float v = fish_semantic_adjust(logits[i], history_mask[i], repetition_penalty);
        local_total += exp(static_cast<double>(v - maximum));
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

    // Exact float-score radix threshold: 32 scans, O(32 * vocabulary), not O(V^2).
    if (top_p < 1.0f) {
        uint64_t low = 0, high = 0xffffffffULL;
        for (int pass = 0; pass < 32; ++pass) {
            if (tid == 0) midpoint = static_cast<uint32_t>((low + high + 1ULL) / 2ULL);
            __syncthreads();
            double local_mass = 0.0;
            for (int i = tid; i < vocabulary; i += static_cast<int>(blockDim.x)) {
                float v = fish_semantic_adjust(logits[i], history_mask[i], repetition_penalty);
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
            float v = fish_semantic_adjust(logits[i], history_mask[i], repetition_penalty);
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
                    float v = fish_semantic_adjust(logits[i], history_mask[i], repetition_penalty);
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
        float v = fish_semantic_adjust(logits[i], history_mask[i], repetition_penalty);
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
    }
}
