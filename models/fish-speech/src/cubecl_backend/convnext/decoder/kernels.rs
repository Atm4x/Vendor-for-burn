use cubecl::prelude::*;

/// Output-gather transposed convolution with the same trimmed boundary as Firefly CPU.
#[cube(launch)]
pub(super) fn transconv_kernel(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    #[comptime] in_channels: usize,
    #[comptime] out_channels: usize,
    #[comptime] in_frames: usize,
    #[comptime] out_frames: usize,
    #[comptime] kernel: usize,
    #[comptime] stride: usize,
    #[comptime] dilation: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let t = ABSOLUTE_POS % out_frames;
        let c = (ABSOLUTE_POS / out_frames) % out_channels;
        let b = ABSOLUTE_POS / (out_channels * out_frames);
        let mut sum = bias[c];
        for channel in 0..in_channels {
            // Reverse-k enumeration visits contributing input positions in forward-time order.
            for reverse in 0..kernel {
                let k = kernel - 1 - reverse;
                let offset = k * dilation;
                if t >= offset {
                    let pos = t - offset;
                    if pos % stride == 0 {
                        let input_t = pos / stride;
                        if input_t < in_frames {
                            sum += input[(b * in_channels + channel) * in_frames + input_t]
                                * weights[(channel * out_channels + c) * kernel + k];
                        }
                    }
                }
            }
        }
        output[ABSOLUTE_POS] = sum;
    }
}

#[cube(launch)]
pub(super) fn decoder_activation_kernel(input: &[f32], output: &mut [f32], #[comptime] tanh: bool) {
    if ABSOLUTE_POS < output.len() {
        let x = input[ABSOLUTE_POS];
        output[ABSOLUTE_POS] = if tanh {
            x.tanh()
        } else {
            x / (1.0f32 + (-x).exp())
        };
    }
}

#[cube(launch)]
pub(super) fn decoder_mean_first_kernel(
    input: &[f32],
    output: &mut [f32],
    #[comptime] branches: usize,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] / f32::cast_from(branches);
    }
}

#[cube(launch)]
pub(super) fn decoder_mean_add_kernel(
    previous: &[f32],
    input: &[f32],
    output: &mut [f32],
    #[comptime] branches: usize,
) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] =
            previous[ABSOLUTE_POS] + input[ABSOLUTE_POS] / f32::cast_from(branches);
    }
}
