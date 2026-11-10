//! Experimental shared output-channel packed Conv1d for constant-weight inference.
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
fn prepacked_temporal_kernel<E: Numeric, NOut: Size>(
    input: &Tensor<E>,
    weight: &Tensor<Vector<E, NOut>>,
    bias: ComptimeOption<&[Vector<E, NOut>]>,
    mut output: LinearViewMut<'_, Vector<E, NOut>>,
    out_len: u32,
    stride: u32,
    dilation: u32,
    padding: i32,
    #[define(E)] _dtype: ElemType,
) {
    let out_vector = output.vector_size();
    let cout_group = weight.shape(3) as u32;
    let groups = weight.shape(0) as u32;
    let cout = groups * cout_group;
    let vectors_per_time = cout / out_vector as u32;
    let tiles = (out_len + 1) / 2;
    let total = input.shape(0) as u32 * tiles * vectors_per_time;
    let unit = ABSOLUTE_POS as u32;
    if unit >= total {
        terminate!();
    }

    let vector_index = unit % vectors_per_time;
    let tile_linear = unit / vectors_per_time;
    let b = tile_linear / tiles;
    let t0 = tile_linear % tiles * 2;
    let t1 = t0 + 1;
    let oc = vector_index * out_vector as u32;
    let g = oc / cout_group;
    let local_oc = oc % cout_group;
    let cin_group = weight.shape(2) as u32;
    let kernel = weight.shape(1) as u32;
    let in_len = input.shape(1) as u32;

    let base_bias: ComptimeOption<Vector<E, NOut>> = bias.map(|x| x[vector_index as usize]);
    let mut sum0 = base_bias.unwrap_or_else(|| Vector::zero());
    let mut sum1 = sum0;
    for k in 0..kernel {
        let i0 = (t0 * stride + k * dilation) as i32 - padding;
        let i1 = (t1 * stride + k * dilation) as i32 - padding;
        let valid0 = i0 >= 0 && (i0 as u32) < in_len;
        let valid1 = t1 < out_len && i1 >= 0 && (i1 as u32) < in_len;
        if valid0 || valid1 {
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
}

/// Takes the same input/output NHWC tensors as Direct, but the weight binding
/// must already be packed to [groups, K, CinGroup, CoutGroup].
pub fn launch_prepacked_temporal_pair(
    client: &Client,
    tensors: DirectTensors,
    args: ConvolutionArgs<1>,
    groups: usize,
    dtype: ElemType,
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
    let work_items = out.shape[0] * out.shape[1].div_ceil(2) * (out.shape[2] / vector_size_out);
    let cube_dim = CubeDim::new(client, work_items);
    let cube_count = calculate_cube_count_elemwise(client, work_items, cube_dim);
    let out_len = out.shape[1] as u32;
    let address_type = input
        .required_address_type(dtype.size())
        .max(weight.required_address_type(dtype.size()))
        .max(out.required_address_type(dtype.size()));
    unsafe {
        prepacked_temporal_kernel::launch_unchecked(
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
        );
    }
    Ok(())
}
