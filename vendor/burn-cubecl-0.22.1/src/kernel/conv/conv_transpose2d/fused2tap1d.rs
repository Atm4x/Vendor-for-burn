//! Cooperative, single-launch two-tap polyphase ConvTranspose1d for F32 inference.
//!
//! Each CTA produces a 16-time by 32-output-channel tile for one stride phase.
//! Input and filter tiles are staged cooperatively in shared memory. A lane
//! owns one output channel and four temporal positions, reducing 32 input
//! channels at a time without any global im2col/gather/scatter allocation.

use crate::{
    kernel::{utils::address_type},
    ops::{numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::{Shape, ops::conv::calculate_conv_transpose_output_size};
use cubecl::{prelude::*, std::tensor::layout::linear::LinearViewMut};
use cubek::convolution::components::ConvSetupError;

const TIME_TILE: usize = 16;
const CHANNEL_TILE: usize = 32;
const REDUCTION_TILE: usize = 32;
const THREADS: usize = 128;

#[derive(CubeType, CubeLaunch)]
struct FusedTwoTapArgs {
    out_length: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    time_tiles: usize,
    channel_tiles: usize,
}

#[cube(launch, address_type = "dynamic")]
fn fused_two_tap_kernel(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: ComptimeOption<&[f32]>,
    mut output: LinearViewMut<'_, f32>,
    args: FusedTwoTapArgs,
) {
    let thread = UNIT_POS as usize;
    let channel_lane = UNIT_POS_X as usize;
    let time_lane = UNIT_POS_Y as usize;
    let block = CUBE_POS as usize;
    let channel_tile = block % args.channel_tiles;
    let temporal_tile = (block / args.channel_tiles) % args.time_tiles;
    let phase = (block / (args.channel_tiles * args.time_tiles)) % args.stride;
    let group = (block / (args.channel_tiles * args.time_tiles * args.stride)) % args.groups;
    let batch = block / (args.channel_tiles * args.time_tiles * args.stride * args.groups);

    let cout_group = weight.shape(2);
    let cout = cout_group * args.groups;
    let cin_group = input.shape(2) / args.groups;
    let input_length = input.shape(1);
    let oc_local = channel_tile * CHANNEL_TILE + channel_lane;
    let oc = group * cout_group + oc_local;
    let q_base = temporal_tile * TIME_TILE;
    let filter_phase = (phase + args.padding) % args.stride;
    let source_base = (phase + args.padding) / args.stride + q_base;

    let mut input_shared = Shared::<[f32]>::new_slice(2 * TIME_TILE * REDUCTION_TILE);
    let mut weight_shared = Shared::<[f32]>::new_slice(2 * REDUCTION_TILE * CHANNEL_TILE);
    let mut accum0 = 0.0f32;
    let mut accum1 = 0.0f32;
    let mut accum2 = 0.0f32;
    let mut accum3 = 0.0f32;

    let channel_blocks = (cin_group + REDUCTION_TILE - 1) / REDUCTION_TILE;
    for ci_block in 0..channel_blocks {
        for slot in 0..8usize {
            let index = thread + slot * THREADS;
            let tap = index / (TIME_TILE * REDUCTION_TILE);
            let local = index % (TIME_TILE * REDUCTION_TILE);
            let q = local / REDUCTION_TILE;
            let ci = ci_block * REDUCTION_TILE + local % REDUCTION_TILE;
            let source = source_base + q;
            let valid = if tap == 0 {
                source < input_length
            } else {
                source > 0 && source - 1 < input_length
            };
            let mut value = 0.0f32;
            if ci < cin_group && valid {
                let time_index = if tap == 0 { source } else { source - 1 };
                value = input[
                    batch * input.stride(0)
                        + time_index * input.stride(1)
                        + (group * cin_group + ci) * input.stride(2)
                ];
            }
            input_shared[index] = value;
        }
        for slot in 0..16usize {
            let index = thread + slot * THREADS;
            let tap = index / (REDUCTION_TILE * CHANNEL_TILE);
            let local = index % (REDUCTION_TILE * CHANNEL_TILE);
            let ci = ci_block * REDUCTION_TILE + local / CHANNEL_TILE;
            let output_channel = channel_tile * CHANNEL_TILE + local % CHANNEL_TILE;
            let mut value = 0.0f32;
            if ci < cin_group && output_channel < cout_group {
                value = weight[
                    (group * cin_group + ci) * weight.stride(0)
                        + (filter_phase + tap * args.stride) * weight.stride(1)
                        + output_channel * weight.stride(2)
                ];
            }
            weight_shared[index] = value;
        }
        sync_cube();
        for ci in 0..REDUCTION_TILE {
            let w0 = weight_shared[ci * CHANNEL_TILE + channel_lane];
            let w1 = weight_shared[REDUCTION_TILE * CHANNEL_TILE + ci * CHANNEL_TILE + channel_lane];
            let q0 = time_lane;
            let q1 = time_lane + 4;
            let q2 = time_lane + 8;
            let q3 = time_lane + 12;
            accum0 += input_shared[q0 * REDUCTION_TILE + ci] * w0;
            accum0 += input_shared[TIME_TILE * REDUCTION_TILE + q0 * REDUCTION_TILE + ci] * w1;
            accum1 += input_shared[q1 * REDUCTION_TILE + ci] * w0;
            accum1 += input_shared[TIME_TILE * REDUCTION_TILE + q1 * REDUCTION_TILE + ci] * w1;
            accum2 += input_shared[q2 * REDUCTION_TILE + ci] * w0;
            accum2 += input_shared[TIME_TILE * REDUCTION_TILE + q2 * REDUCTION_TILE + ci] * w1;
            accum3 += input_shared[q3 * REDUCTION_TILE + ci] * w0;
            accum3 += input_shared[TIME_TILE * REDUCTION_TILE + q3 * REDUCTION_TILE + ci] * w1;
        }
        sync_cube();
    }

    let mut bias_value = 0.0f32;
    #[comptime]
    match bias {
        ComptimeOption::Some(b) => {
            if oc_local < cout_group {
                bias_value = b[oc];
            }
        }
        ComptimeOption::None => {}
    }
    if oc_local < cout_group {
        let q0 = q_base + time_lane;
        let q1 = q0 + 4;
        let q2 = q0 + 8;
        let q3 = q0 + 12;
        let time0 = phase + q0 * args.stride;
        let time1 = phase + q1 * args.stride;
        let time2 = phase + q2 * args.stride;
        let time3 = phase + q3 * args.stride;
        let out_base = batch * args.out_length * cout + oc;
        if time0 < args.out_length { output.write(out_base + time0 * cout, accum0 + bias_value); }
        if time1 < args.out_length { output.write(out_base + time1 * cout, accum1 + bias_value); }
        if time2 < args.out_length { output.write(out_base + time2 * cout, accum2 + bias_value); }
        if time3 < args.out_length { output.write(out_base + time3 * cout, accum3 + bias_value); }
    }
}

/// Executes only on K=2*stride, F32 input and prepacked [Cin,K,Cout/group]
/// weights, with no intermediate buffers. All other geometries are rejected.
pub fn conv_transpose1d_fused_two_tap_packed(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    padding_out: usize,
    groups: usize,
) -> Result<CubeTensor, ConvSetupError> {
    if groups == 0 || stride == 0 || dilation != 1 || padding_out != 0
        || input.dtype != burn_backend::DType::F32
        || weight.dtype != burn_backend::DType::F32
        || bias.as_ref().is_some_and(|b| b.dtype != burn_backend::DType::F32)
    {
        return Err(ConvSetupError::Unknown);
    }
    let [batch, cin, in_length] = input.meta.shape().dims();
    let [weight_cin, kernel, cout_group] = weight.meta.shape().dims();
    if batch == 0 || cin == 0 || in_length == 0 || cin != weight_cin
        || cin % groups != 0 || cout_group == 0 || kernel != 2 * stride
    {
        return Err(ConvSetupError::Unknown);
    }
    let out_length = calculate_conv_transpose_output_size(
        kernel, stride, padding, padding_out, dilation, in_length,
    );
    if out_length == 0 { return Err(ConvSetupError::Unknown); }
    if input.client.properties().hardware.max_units_per_cube < THREADS as u32 {
        return Err(ConvSetupError::Unknown);
    }
    let cout = cout_group * groups;
    let input = permute_nchw_to_nhwc(input);
    let output = empty_device_dtype(
        input.client.clone(), input.device.clone(),
        Shape::new([batch, out_length, cout]), input.dtype,
    );
    let time_tiles = out_length.div_ceil(stride).div_ceil(TIME_TILE);
    let channel_tiles = cout_group.div_ceil(CHANNEL_TILE);
    let total_blocks = batch * groups * stride * time_tiles * channel_tiles;
    let count = CubeCount::new_1d(total_blocks as u32);
    let dim = CubeDim::new_2d(32, 4);
    let client = input.client.clone();
    fused_two_tap_kernel::launch(
        &client, count, dim,
        address_type!(input, weight, bias, output),
        input.into_tensor_arg(),
        weight.into_tensor_arg(),
        bias.map(|b| b.into_buffer_arg()).into(),
        output.clone().into_linear_view(),
        FusedTwoTapArgsLaunch::new(out_length, stride, padding, groups, time_tiles, channel_tiles),
    );
    Ok(permute_nhwc_to_nchw(output))
}
