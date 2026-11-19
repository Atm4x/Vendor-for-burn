use cubecl::prelude::*;

#[cube(launch)]
pub(super) fn grouped_conv_kernel(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    #[comptime] in_channels: usize,
    #[comptime] out_channels: usize,
    #[comptime] in_frames: usize,
    #[comptime] out_frames: usize,
    #[comptime] kernel: usize,
    #[comptime] stride: usize,
    #[comptime] dilation: usize,
    #[comptime] groups: usize,
    #[comptime] pad_left: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let t = ABSOLUTE_POS % out_frames;
        let ch = (ABSOLUTE_POS / out_frames) % out_channels;
        let batch = ABSOLUTE_POS / (out_frames * out_channels);
        let in_per_group = in_channels / groups;
        let out_per_group = out_channels / groups;
        let group = ch / out_per_group;
        let mut sum = bias[ch];
        for local_c in 0..in_per_group {
            let input_ch = group * in_per_group + local_c;
            for k in 0..kernel {
                let padded = t * stride + k * dilation;
                if padded >= pad_left {
                    let src_t = padded - pad_left;
                    if src_t < in_frames {
                        sum += input[(batch * in_channels + input_ch) * in_frames + src_t]
                            * weight[(ch * in_per_group + local_c) * kernel + k];
                    }
                }
            }
        }
        output[ABSOLUTE_POS] = sum;
    }
}

#[cube(launch)]
pub(super) fn layer_norm_kernel(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    epsilon: &[f32],
    output: &mut [f32],
    #[comptime] channels: usize,
    #[comptime] frames: usize,
) {
    if ABSOLUTE_POS < output.len() / channels {
        let t = ABSOLUTE_POS % frames;
        let batch = ABSOLUTE_POS / frames;
        let mut mean = 0.0f32;
        for c in 0..channels {
            mean += input[(batch * channels + c) * frames + t];
        }
        mean /= f32::cast_from(channels);
        let mut variance = 0.0f32;
        for c in 0..channels {
            let centered = input[(batch * channels + c) * frames + t] - mean;
            variance += centered * centered;
        }
        variance /= f32::cast_from(channels);
        let inv = 1.0f32 / (variance + epsilon[0]).sqrt();
        for c in 0..channels {
            let offset = (batch * channels + c) * frames + t;
            output[offset] = (input[offset] - mean) * inv * weight[c] + bias[c];
        }
    }
}

#[cube(launch)]
pub(super) fn linear_channels_kernel(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    #[comptime] in_channels: usize,
    #[comptime] out_channels: usize,
    #[comptime] frames: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let t = ABSOLUTE_POS % frames;
        let ch = (ABSOLUTE_POS / frames) % out_channels;
        let batch = ABSOLUTE_POS / (frames * out_channels);
        let mut sum = bias[ch];
        for c in 0..in_channels {
            sum += weight[ch * in_channels + c] * input[(batch * in_channels + c) * frames + t];
        }
        output[ABSOLUTE_POS] = sum;
    }
}

#[cube(launch)]
pub(super) fn gelu_kernel(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        let x = input[ABSOLUTE_POS];
        let z = x * 0.7071067811865476f32;
        let absolute = z.abs();
        let t = 1.0f32 / (1.0f32 + 0.3275911f32 * absolute);
        let p = ((((1.061405429f32 * t - 1.453152027f32) * t + 1.421413741f32) * t
            - 0.284496736f32)
            * t
            + 0.254829592f32)
            * t;
        let erf = if z < 0.0f32 {
            -(1.0f32 - p * (-(absolute * absolute)).exp())
        } else {
            1.0f32 - p * (-(absolute * absolute)).exp()
        };
        output[ABSOLUTE_POS] = 0.5f32 * x * (1.0f32 + erf);
    }
}

#[cube(launch)]
pub(super) fn layer_scale_residual_kernel(
    original: &[f32],
    projected: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    #[comptime] channels: usize,
    #[comptime] frames: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let ch = (ABSOLUTE_POS / frames) % channels;
        output[ABSOLUTE_POS] = original[ABSOLUTE_POS] + projected[ABSOLUTE_POS] * gamma[ch];
    }
}
