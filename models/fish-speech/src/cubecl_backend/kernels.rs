use cubecl::prelude::*;
use half::bf16;

#[cube(launch)]
pub(super) fn add_kernel(a: &[f32], b: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

#[cube(launch)]
pub(super) fn mul_kernel(a: &[f32], b: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS];
    }
}

#[cube(launch)]
pub(super) fn silu_kernel(a: &[f32], out: &mut [f32]) {
    if ABSOLUTE_POS < out.len() {
        let x = a[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = x / (1.0 + (-x).exp());
    }
}

#[cube(launch)]
pub(super) fn token_embedding_kernel(
    tokens: &[i32],
    weights: &[f32],
    out: &mut [f32],
    #[comptime] dimension: usize,
    #[comptime] vocabulary: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let token = tokens[ABSOLUTE_POS / dimension];
        if token >= 0 && token < vocabulary as i32 {
            out[ABSOLUTE_POS] = weights[(token as usize) * dimension + ABSOLUTE_POS % dimension];
        } else {
            out[ABSOLUTE_POS] = 0.0;
        }
    }
}

#[cube(launch)]
pub(super) fn fish_embedding_kernel(
    tokens: &[i32],
    words: &[f32],
    codebook: &[f32],
    semantic_lut: &[i32],
    out: &mut [f32],
    #[comptime] frames: usize,
    #[comptime] dimension: usize,
    #[comptime] num_codebooks: usize,
    #[comptime] codebook_size: usize,
    #[comptime] vocabulary: usize,
    #[comptime] shared: bool,
) {
    if ABSOLUTE_POS < out.len() {
        let feature = ABSOLUTE_POS % dimension;
        let frame = (ABSOLUTE_POS / dimension) % frames;
        let batch = ABSOLUTE_POS / (frames * dimension);
        let token = tokens[(batch * (num_codebooks + 1)) * frames + frame];
        if token < 0 || token >= vocabulary as i32 {
            out[ABSOLUTE_POS] = 0.0;
        } else {
            let word = token as usize;
            let mut sum = words[word * dimension + feature];
            if semantic_lut[word] == 1 {
                for codebook_index in 0..num_codebooks {
                    let code =
                        tokens[(batch * (num_codebooks + 1) + codebook_index + 1) * frames + frame];
                    if code >= 0 && code < codebook_size as i32 {
                        let offset = if shared {
                            codebook_index * codebook_size
                        } else {
                            codebook_index * 0
                        };
                        sum += codebook[(offset + code as usize) * dimension + feature];
                    }
                }
            }
            out[ABSOLUTE_POS] = sum;
        }
    }
}

#[cube(launch)]
pub(super) fn split_qkv_kernel(
    packed: &[f32],
    out: &mut [f32],
    #[comptime] frames: usize,
    #[comptime] head_count: usize,
    #[comptime] head_dim: usize,
    #[comptime] packed_width: usize,
    #[comptime] part_offset: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let dim = ABSOLUTE_POS % head_dim;
        let frame = (ABSOLUTE_POS / head_dim) % frames;
        let head = (ABSOLUTE_POS / (head_dim * frames)) % head_count;
        let batch = ABSOLUTE_POS / (head_dim * frames * head_count);
        out[ABSOLUTE_POS] =
            packed[(batch * frames + frame) * packed_width + part_offset + head * head_dim + dim];
    }
}

#[cube(launch)]
pub(super) fn rope_kernel(
    input: &[f32],
    cosine: &[f32],
    signed_sine: &[f32],
    out: &mut [f32],
    #[comptime] frames: usize,
    #[comptime] head_dim: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let feature = ABSOLUTE_POS % head_dim;
        let frame = (ABSOLUTE_POS / head_dim) % frames;
        let partner = if feature % 2 == 0 {
            feature + 1
        } else {
            feature - 1
        };
        let neighbor = ABSOLUTE_POS - feature + partner;
        let coefficient = frame * head_dim + feature;
        out[ABSOLUTE_POS] =
            input[ABSOLUTE_POS] * cosine[coefficient] + input[neighbor] * signed_sine[coefficient];
    }
}

#[cube(launch)]
pub(super) fn kv_update_kernel(
    previous: &[bf16],
    fresh: &[f32],
    out: &mut [bf16],
    #[comptime] heads: usize,
    #[comptime] capacity: usize,
    #[comptime] frames: usize,
    #[comptime] dimension: usize,
    #[comptime] position: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let feature = ABSOLUTE_POS % dimension;
        let frame = (ABSOLUTE_POS / dimension) % capacity;
        let head = (ABSOLUTE_POS / (dimension * capacity)) % heads;
        let batch = ABSOLUTE_POS / (dimension * capacity * heads);
        if frame >= position && frame < position + frames {
            let src = ((batch * heads + head) * frames + frame - position) * dimension + feature;
            out[ABSOLUTE_POS] = bf16::cast_from(fresh[src]);
        } else {
            out[ABSOLUTE_POS] = previous[ABSOLUTE_POS];
        }
    }
}

#[cube(launch)]
pub(super) fn kv_prefix_kernel(
    cache: &[bf16],
    out: &mut [bf16],
    #[comptime] capacity: usize,
    #[comptime] active: usize,
    #[comptime] dimension: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let feature = ABSOLUTE_POS % dimension;
        let frame = (ABSOLUTE_POS / dimension) % active;
        let batch_head = ABSOLUTE_POS / (dimension * active);
        out[ABSOLUTE_POS] = cache[(batch_head * capacity + frame) * dimension + feature];
    }
}

#[cube(launch)]
pub(super) fn causal_gqa_kernel(
    query: &[f32],
    key: &[bf16],
    value: &[bf16],
    out: &mut [f32],
    #[comptime] q_heads: usize,
    #[comptime] kv_heads: usize,
    #[comptime] frames: usize,
    #[comptime] active: usize,
    #[comptime] dimension: usize,
    #[comptime] first_position: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let feature = ABSOLUTE_POS % dimension;
        let head = (ABSOLUTE_POS / dimension) % q_heads;
        let frame = (ABSOLUTE_POS / (dimension * q_heads)) % frames;
        let batch = ABSOLUTE_POS / (dimension * q_heads * frames);
        let kv_head = head / (q_heads / kv_heads);
        let visible = first_position + frame + 1;
        let q_offset = ((batch * q_heads + head) * frames + frame) * dimension;
        let mut max_score = query[q_offset] * 0.0 - 3.4e38f32;
        let scale = 1.0 / (dimension as f32).sqrt();
        for t in 0..visible {
            let k_offset = ((batch * kv_heads + kv_head) * active + t) * dimension;
            let mut score = 0.0f32;
            for d in 0..dimension {
                score += query[q_offset + d] * f32::cast_from(key[k_offset + d]);
            }
            score *= scale;
            if score > max_score {
                max_score = score;
            }
        }
        let mut denominator = 0.0f32;
        let mut numerator = 0.0f32;
        for t in 0..visible {
            let k_offset = ((batch * kv_heads + kv_head) * active + t) * dimension;
            let mut score = 0.0f32;
            for d in 0..dimension {
                score += query[q_offset + d] * f32::cast_from(key[k_offset + d]);
            }
            let probability = (score * scale - max_score).exp();
            denominator += probability;
            numerator += probability * f32::cast_from(value[k_offset + feature]);
        }
        out[ABSOLUTE_POS] = numerator / denominator;
    }
}

#[cube(launch)]
pub(super) fn linear_kernel(
    a: &[f32],
    w: &[f32],
    out: &mut [f32],
    #[comptime] input_features: usize,
    #[comptime] output_features: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let row = ABSOLUTE_POS / output_features;
        let col = ABSOLUTE_POS % output_features;
        let mut sum = 0.0f32;
        for k in 0..input_features {
            sum += a[row * input_features + k] * w[col * input_features + k];
        }
        out[ABSOLUTE_POS] = sum;
    }
}

#[cube(launch)]
pub(super) fn add_bias_kernel(bias: &[f32], out: &mut [f32], #[comptime] width: usize) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] += bias[ABSOLUTE_POS % width];
    }
}

#[cube(launch)]
pub(super) fn rms_norm_kernel(
    a: &[f32],
    gamma: &[f32],
    out: &mut [f32],
    #[comptime] width: usize,
    #[comptime] epsilon_bits: u32,
) {
    let rows = out.len() / width;
    if ABSOLUTE_POS < rows {
        let start = ABSOLUTE_POS * width;
        let mut squared = 0.0f32;
        for k in 0..width {
            let x = a[start + k];
            squared += x * x;
        }
        let scale = 1.0 / (squared / (width as f32) + f32::from_bits(epsilon_bits)).sqrt();
        for k in 0..width {
            out[start + k] = a[start + k] * scale * gamma[k];
        }
    }
}

#[cube(launch)]
pub(super) fn last_token_kernel(
    a: &[f32],
    out: &mut [f32],
    #[comptime] sequence: usize,
    #[comptime] width: usize,
) {
    if ABSOLUTE_POS < out.len() {
        let batch = ABSOLUTE_POS / width;
        let feature = ABSOLUTE_POS % width;
        out[ABSOLUTE_POS] = a[(batch * sequence + sequence - 1) * width + feature];
    }
}
