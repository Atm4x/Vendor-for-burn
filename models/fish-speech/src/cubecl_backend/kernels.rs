use cubecl::prelude::*;

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

#[cube]
fn kv_value(value: f32, #[comptime] emulate_bf16: bool) -> f32 {
    if emulate_bf16 {
        let bits = u32::reinterpret(value);
        let magnitude = bits & 0x7fff_ffff;
        let rounded = if magnitude > 0x7f80_0000 {
            bits | 0x0040_0000
        } else {
            bits + 0x7fff + ((bits >> 16) & 1)
        };
        f32::reinterpret(rounded & 0xffff_0000)
    } else {
        value
    }
}

#[cube(launch)]
pub(super) fn kv_write_kernel<F: Float>(
    fresh: &[f32],
    cache: &mut [F],
    #[comptime] capacity: usize,
    #[comptime] frames: usize,
    #[comptime] dimension: usize,
    #[comptime] position: usize,
    #[comptime] emulate_bf16: bool,
) {
    if ABSOLUTE_POS < fresh.len() {
        let row = ABSOLUTE_POS / dimension;
        let feature = ABSOLUTE_POS % dimension;
        let destination =
            ((row / frames) * capacity + position + row % frames) * dimension + feature;
        cache[destination] = F::cast_from(kv_value(fresh[ABSOLUTE_POS], emulate_bf16));
    }
}

#[cube(launch)]
pub(super) fn kv_update_kernel<F: Float>(
    previous: &[F],
    fresh: &[f32],
    out: &mut [F],
    #[comptime] heads: usize,
    #[comptime] capacity: usize,
    #[comptime] frames: usize,
    #[comptime] dimension: usize,
    #[comptime] position: usize,
    #[comptime] emulate_bf16: bool,
) {
    if ABSOLUTE_POS < out.len() {
        let feature = ABSOLUTE_POS % dimension;
        let frame = (ABSOLUTE_POS / dimension) % capacity;
        let head = (ABSOLUTE_POS / (dimension * capacity)) % heads;
        let batch = ABSOLUTE_POS / (dimension * capacity * heads);
        if frame >= position && frame < position + frames {
            let src = ((batch * heads + head) * frames + frame - position) * dimension + feature;
            out[ABSOLUTE_POS] = F::cast_from(kv_value(fresh[src], emulate_bf16));
        } else {
            out[ABSOLUTE_POS] = previous[ABSOLUTE_POS];
        }
    }
}

#[cube(launch)]
pub(super) fn kv_prefix_kernel<F: Float>(
    cache: &[F],
    out: &mut [F],
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
pub(super) fn gqa_scores_kernel<F: Float>(
    query: &[f32],
    key: &[F],
    scores: &mut [f32],
    #[comptime] q_heads: usize,
    #[comptime] kv_heads: usize,
    #[comptime] frames: usize,
    #[comptime] active: usize,
    #[comptime] dimension: usize,
    #[comptime] first_position: usize,
) {
    if ABSOLUTE_POS < scores.len() {
        let step = ABSOLUTE_POS % active;
        let row = ABSOLUTE_POS / active;
        let frame = row % frames;
        let head = (row / frames) % q_heads;
        let batch = row / (frames * q_heads);
        let mut score = 0.0f32;
        if step <= first_position + frame {
            let kv_head = head / (q_heads / kv_heads);
            let key_start = ((batch * kv_heads + kv_head) * active + step) * dimension;
            let query_start = row * dimension;
            for d in 0..dimension {
                score += query[query_start + d] * f32::cast_from(key[key_start + d]);
            }
            score *= 1.0 / (dimension as f32).sqrt();
        }
        scores[ABSOLUTE_POS] = score;
    }
}

#[cube(launch)]
pub(super) fn gqa_softmax_kernel(
    scores: &mut [f32],
    #[comptime] frames: usize,
    #[comptime] active: usize,
    #[comptime] first_position: usize,
) {
    let row = ABSOLUTE_POS;
    if row < scores.len() / active {
        let visible = first_position + row % frames + 1;
        let start = row * active;
        let mut maximum = scores[start];
        for step in 1..visible {
            let score = scores[start + step];
            if score > maximum {
                maximum = score;
            }
        }
        let mut total = 0.0f32;
        for step in 0..visible {
            let weight = (scores[start + step] - maximum).exp();
            scores[start + step] = weight;
            total += weight;
        }
        for step in 0..visible {
            scores[start + step] /= total;
        }
    }
}

#[cube(launch)]
pub(super) fn gqa_values_kernel<F: Float>(
    scores: &[f32],
    value: &[F],
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
        let score_row = (batch * q_heads + head) * frames + frame;
        let kv_head = head / (q_heads / kv_heads);
        let visible = first_position + frame + 1;
        let value_start = (batch * kv_heads + kv_head) * active * dimension;
        let mut sum = 0.0f32;
        for step in 0..visible {
            sum += scores[score_row * active + step]
                * f32::cast_from(value[value_start + step * dimension + feature]);
        }
        out[ABSOLUTE_POS] = sum;
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
pub(super) fn linear_plane_kernel(
    a: &[f32],
    w: &[f32],
    out: &mut [f32],
    #[comptime] input_features: usize,
    #[comptime] output_features: usize,
    #[comptime] plane_width: usize,
) {
    let index = ABSOLUTE_POS / plane_width;
    let lane = UNIT_POS_X as usize % plane_width;
    let mut sum = 0.0f32;
    if index < out.len() {
        let row = index / output_features;
        let column = index % output_features;
        let mut k = lane;
        while k < input_features {
            sum += a[row * input_features + k] * w[column * input_features + k];
            k += plane_width;
        }
    }
    let reduced = plane_sum(sum);
    if lane == 0 && index < out.len() {
        out[index] = reduced;
    }
}

#[cube]
fn unpack_bf16(word: u32, odd: usize) -> f32 {
    let bits = if odd == 0 {
        word << 16
    } else {
        word & 0xffff_0000
    };
    f32::reinterpret(bits)
}

#[cube(launch)]
pub(super) fn linear_packed_plane_kernel(
    a: &[f32],
    w: &[u32],
    out: &mut [f32],
    #[comptime] input_features: usize,
    #[comptime] output_features: usize,
    #[comptime] plane_width: usize,
) {
    let index = ABSOLUTE_POS / plane_width;
    let lane = UNIT_POS_X as usize % plane_width;
    let mut sum = 0.0f32;
    if index < out.len() {
        let row = index / output_features;
        let column = index % output_features;
        let mut pair = lane;
        while pair < input_features / 2 {
            let word = w[column * (input_features / 2) + pair];
            let offset = row * input_features + pair * 2;
            sum += a[offset] * unpack_bf16(word, 0);
            sum += a[offset + 1] * unpack_bf16(word, 1);
            pair += plane_width;
        }
    }
    let reduced = plane_sum(sum);
    if lane == 0 && index < out.len() {
        out[index] = reduced;
    }
}

#[cube(launch)]
pub(super) fn linear_packed_tile_kernel(
    a: &[f32],
    w: &[u32],
    out: &mut [f32],
    #[comptime] rows: usize,
    #[comptime] input_features: usize,
    #[comptime] output_features: usize,
) {
    let x = UNIT_POS_X as usize;
    let y = UNIT_POS_Y as usize;
    let row = CUBE_POS_Y as usize * 16 + y;
    let column = CUBE_POS_X as usize * 16 + x;
    let weight_row = CUBE_POS_X as usize * 16 + y;
    let mut left = Shared::<[f32]>::new_slice(256usize);
    let mut right = Shared::<[f32]>::new_slice(272usize);
    let mut sum = 0.0f32;
    let mut offset = 0usize;
    while offset < input_features {
        let k = offset + x;
        left[y * 16 + x] = if row < rows && k < input_features {
            a[row * input_features + k]
        } else {
            0.0f32
        };
        right[x * 17 + y] = if weight_row < output_features && k < input_features {
            unpack_bf16(w[weight_row * (input_features / 2) + k / 2], k % 2)
        } else {
            0.0f32
        };
        sync_cube();
        #[unroll]
        for i in 0..16 {
            sum += left[y * 16 + i] * right[i * 17 + x];
        }
        sync_cube();
        offset += 16;
    }
    if row < rows && column < output_features {
        out[row * output_features + column] = sum;
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
pub(super) fn rms_norm_plane_kernel(
    a: &[f32],
    gamma: &[f32],
    out: &mut [f32],
    #[comptime] width: usize,
    #[comptime] epsilon_bits: u32,
    #[comptime] plane_width: usize,
) {
    let row = ABSOLUTE_POS / plane_width;
    let lane = UNIT_POS_X as usize % plane_width;
    let mut squared = 0.0f32;
    let start = row * width;
    if start < out.len() {
        let mut k = lane;
        while k < width {
            let x = a[start + k];
            squared += x * x;
            k += plane_width;
        }
    }
    let sum = plane_sum(squared);
    if start < out.len() {
        let scale = 1.0 / (sum / width as f32 + f32::from_bits(epsilon_bits)).sqrt();
        let mut k = lane;
        while k < width {
            out[start + k] = a[start + k] * scale * gamma[k];
            k += plane_width;
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
