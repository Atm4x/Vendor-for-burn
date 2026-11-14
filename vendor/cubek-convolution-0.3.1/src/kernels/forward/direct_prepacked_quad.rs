//! Experimental four-position temporal tile for shared CubeCL packed Conv1d.
//!
//! Direct GPU arithmetic; keeps four output accumulator vectors live while
//! reusing each packed weight vector once per input/kernel step. This may
//! increase register pressure and is measured separately from TemporalPair.
//!
//! Weight layout: [groups, kernel, channels_in_per_group, channels_out_per_group].
//! Packing is performed once when a model is loaded; the inner convolution loop
//! uses one contiguous SIMD vector of weights for each input value, with no GEMM.

use cubecl::{
    calculate_cube_count_elemwise,
    client::Client,
    prelude::*,
    std::tensor::layout::linear::{LinearViewMut, linear_view},
    tensor_vector_size_parallel,
};

use super::direct::DirectTensors;
use crate::{components::ConvSetupError, launch::ConvolutionArgs};

#[cube(launch_unchecked, address_type = "dynamic")]
fn prepacked_temporal_quad_kernel<E: Numeric, NOut: Size>(
    input: &Tensor<E>,
    weight: &Tensor<Vector<E, NOut>>,
    bias: ComptimeOption<&[Vector<E, NOut>]>,
    mut output: LinearViewMut<'_, Vector<E, NOut>>,
    out_len: u32,
    stride: u32,
    dilation: u32,
    padding: i32,
    #[define(E)] _dtype: ElemType,
    #[comptime] fast_interior: bool,
    #[comptime] linear_indexing: bool,
) {
    let out_vector = output.vector_size();
    let cout_group = weight.shape(3) as u32;
    let groups = weight.shape(0) as u32;
    let cout = groups * cout_group;
    let vectors_per_time = cout / out_vector as u32;
    let tiles = (out_len + 3) / 4;
    let total = input.shape(0) as u32 * tiles * vectors_per_time;
    let unit = ABSOLUTE_POS as u32;
    if unit >= total {
        terminate!();
    }

    let vector_index = unit % vectors_per_time;
    let tile_linear = unit / vectors_per_time;
    let b = tile_linear / tiles;
    let t0 = tile_linear % tiles * 4;
    let t1 = t0 + 1;
    let t2 = t0 + 2;
    let t3 = t0 + 3;
    let oc = vector_index * out_vector as u32;
    let g = oc / cout_group;
    let local_oc = oc % cout_group;
    let cin_group = weight.shape(2) as u32;
    let kernel = weight.shape(1) as u32;
    let in_len = input.shape(1) as u32;

    let base_bias: ComptimeOption<Vector<E, NOut>> = bias.map(|x| x[vector_index as usize]);
    let mut sum0 = base_bias.unwrap_or_else(|| Vector::zero());
    let mut sum1 = sum0;
    let mut sum2 = sum0;
    let mut sum3 = sum0;
    let first_input = (t0 * stride) as i32 - padding;
    let last_input = (t3 * stride + (kernel - 1) * dilation) as i32 - padding;
    let fully_interior = t3 < out_len && first_input >= 0 && last_input < in_len as i32;
    if fast_interior && fully_interior {
        let batch_base = b as usize * input.stride(0);
        let channel_base = (g * cin_group) as usize;
        for k in 0..kernel {
            let i0 = ((t0 * stride + k * dilation) as i32 - padding) as usize;
            let i1 = i0 + stride as usize;
            let i2 = i1 + stride as usize;
            let i3 = i2 + stride as usize;
            let x0 = batch_base + i0 * input.stride(1) + channel_base;
            let x1 = batch_base + i1 * input.stride(1) + channel_base;
            let x2 = batch_base + i2 * input.stride(1) + channel_base;
            let x3 = batch_base + i3 * input.stride(1) + channel_base;
            let weight_base =
                g as usize * weight.stride(0) + k as usize * weight.stride(1) + local_oc as usize;
            if linear_indexing {
                let mut w_index = weight_base / out_vector;
                let w_step = weight.stride(2) / out_vector;
                let mut x_index0 = x0;
                let mut x_index1 = x1;
                let mut x_index2 = x2;
                let mut x_index3 = x3;
                for _ in 0..cin_group {
                    let w = weight[w_index];
                    sum0 += w * Vector::new(input[x_index0]);
                    sum1 += w * Vector::new(input[x_index1]);
                    sum2 += w * Vector::new(input[x_index2]);
                    sum3 += w * Vector::new(input[x_index3]);
                    w_index += w_step;
                    x_index0 += 1;
                    x_index1 += 1;
                    x_index2 += 1;
                    x_index3 += 1;
                }
            } else {
                for ci in 0..cin_group {
                    let w = weight[(weight_base + ci as usize * weight.stride(2)) / out_vector];
                    sum0 += w * Vector::new(input[x0 + ci as usize]);
                    sum1 += w * Vector::new(input[x1 + ci as usize]);
                    sum2 += w * Vector::new(input[x2 + ci as usize]);
                    sum3 += w * Vector::new(input[x3 + ci as usize]);
                }
            }
        }
    } else {
        for k in 0..kernel {
            let i0 = (t0 * stride + k * dilation) as i32 - padding;
            let i1 = (t1 * stride + k * dilation) as i32 - padding;
            let i2 = (t2 * stride + k * dilation) as i32 - padding;
            let i3 = (t3 * stride + k * dilation) as i32 - padding;
            let valid0 = i0 >= 0 && (i0 as u32) < in_len;
            let valid1 = t1 < out_len && i1 >= 0 && (i1 as u32) < in_len;
            let valid2 = t2 < out_len && i2 >= 0 && (i2 as u32) < in_len;
            let valid3 = t3 < out_len && i3 >= 0 && (i3 as u32) < in_len;
            if valid0 || valid1 || valid2 || valid3 {
                for ci in 0..cin_group {
                    let weight_index = g as usize * weight.stride(0)
                        + k as usize * weight.stride(1)
                        + ci as usize * weight.stride(2)
                        + local_oc as usize;
                    let w = weight[weight_index / out_vector];
                    if valid0 {
                        let x_index = b as usize * input.stride(0)
                            + i0 as usize * input.stride(1)
                            + (g * cin_group + ci) as usize;
                        sum0 += w * Vector::new(input[x_index]);
                    }
                    if valid1 {
                        let x_index = b as usize * input.stride(0)
                            + i1 as usize * input.stride(1)
                            + (g * cin_group + ci) as usize;
                        sum1 += w * Vector::new(input[x_index]);
                    }
                    if valid2 {
                        let x_index = b as usize * input.stride(0)
                            + i2 as usize * input.stride(1)
                            + (g * cin_group + ci) as usize;
                        sum2 += w * Vector::new(input[x_index]);
                    }
                    if valid3 {
                        let x_index = b as usize * input.stride(0)
                            + i3 as usize * input.stride(1)
                            + (g * cin_group + ci) as usize;
                        sum3 += w * Vector::new(input[x_index]);
                    }
                }
            }
        }
    }
    let base = b as usize * out_len as usize * vectors_per_time as usize;
    output.write(
        base + t0 as usize * vectors_per_time as usize + vector_index as usize,
        sum0,
    );
    if t1 < out_len {
        output.write(
            base + t1 as usize * vectors_per_time as usize + vector_index as usize,
            sum1,
        );
    }
    if t2 < out_len {
        output.write(
            base + t2 as usize * vectors_per_time as usize + vector_index as usize,
            sum2,
        );
    }
    if t3 < out_len {
        output.write(
            base + t3 as usize * vectors_per_time as usize + vector_index as usize,
            sum3,
        );
    }
}

/// Takes the same input/output NHWC tensors as Direct, but the weight binding
/// must already be packed to [groups, K, CinGroup, CoutGroup].
pub fn launch_prepacked_temporal_quad(
    client: &Client,
    tensors: DirectTensors,
    args: ConvolutionArgs<1>,
    groups: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    launch_quad_impl(client, tensors, args, groups, dtype, false, false)
}

pub fn launch_prepacked_temporal_quad_interior(
    client: &Client,
    tensors: DirectTensors,
    args: ConvolutionArgs<1>,
    groups: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    launch_quad_impl(client, tensors, args, groups, dtype, true, false)
}

pub fn launch_prepacked_temporal_quad_linear(
    client: &Client,
    tensors: DirectTensors,
    args: ConvolutionArgs<1>,
    groups: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    launch_quad_impl(client, tensors, args, groups, dtype, true, true)
}

fn launch_quad_impl(
    client: &Client,
    tensors: DirectTensors,
    args: ConvolutionArgs<1>,
    groups: usize,
    dtype: ElemType,
    fast_interior: bool,
    linear_indexing: bool,
) -> Result<(), ConvSetupError> {
    let DirectTensors {
        input,
        weight,
        bias,
        out,
    } = tensors;
    if groups == 0 || weight.shape.len() != 4 || out.shape.len() != 3 || input.shape.len() != 3 {
        return Err(ConvSetupError::Groups(groups));
    }
    if weight.shape[0] != groups
        || weight.shape[3] * groups != out.shape[2]
        || weight.shape[2] * groups != input.shape[2]
    {
        return Err(ConvSetupError::Unknown);
    }
    if out.shape.iter().any(|&n| n == 0) {
        return Ok(());
    }

    let mut grouped_shape = out.shape.clone();
    grouped_shape[2] = weight.shape[3];
    let vector_size_out = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &grouped_shape,
        &out.strides,
        2,
    );
    if weight.shape[3] % vector_size_out != 0 {
        return Err(ConvSetupError::Unknown);
    }
    let work_items = out.shape[0] * out.shape[1].div_ceil(4) * (out.shape[2] / vector_size_out);
    let cube_dim = CubeDim::new(client, work_items);
    let cube_count = calculate_cube_count_elemwise(client, work_items, cube_dim);
    let out_len = out.shape[1] as u32;
    let address_type = input
        .required_address_type(dtype.size())
        .max(weight.required_address_type(dtype.size()))
        .max(out.required_address_type(dtype.size()));
    unsafe {
        prepacked_temporal_quad_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size_out,
            input.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.map(|b| b.into_buffer_arg()).into(),
            linear_view(out),
            out_len,
            args.stride[0] as u32,
            args.dilation[0] as u32,
            args.padding[0] as i32,
            dtype,
            fast_interior,
            linear_indexing,
        );
    }
    Ok(())
}
