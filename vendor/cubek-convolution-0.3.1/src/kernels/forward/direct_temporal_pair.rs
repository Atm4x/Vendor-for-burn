//! Experimental shared CubeCL Conv1d kernel: one work item accumulates two adjacent
//! output positions and loads each filter vector once for both accumulators.
//!
//! This is independent of CPU/GPU runtimes. It is intentionally not selected by
//! the production launcher until runtime parity and performance are measured.

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
fn temporal_pair_kernel<E: Numeric, NIn: Size, NOut: Size>(
    input: &Tensor<Vector<E, NIn>>,
    weight: &Tensor<Vector<E, NIn>>,
    bias: ComptimeOption<&[Vector<E, NOut>]>,
    mut output: LinearViewMut<'_, Vector<E, NOut>>,
    out_len: u32,
    stride: u32,
    dilation: u32,
    padding: i32,
    channels_per_group: u32,
    #[define(E)] _dtype: ElemType,
) {
    let out_channels = weight.shape(0) as u32;
    let out_vector = output.vector_size();
    let in_vector = input.vector_size();
    let channel_vectors = out_channels / out_vector as u32;
    let pairs = (out_len + 1) / 2;
    let total = input.shape(0) as u32 * pairs * channel_vectors;
    let unit = ABSOLUTE_POS as u32;
    if unit >= total {
        terminate!();
    }

    let channel_vector = unit % channel_vectors;
    let pair_linear = unit / channel_vectors;
    let batch = pair_linear / pairs;
    let t0 = (pair_linear % pairs) * 2;
    let t1 = t0 + 1;
    let oc_start = channel_vector * out_vector as u32;
    let group = oc_start / channels_per_group;
    let in_channels_per_group = weight.shape(2) as u32;
    let in_channel_start = group * in_channels_per_group;
    let kernel_size = weight.shape(1) as u32;
    let input_length = input.shape(1) as u32;

    let base_bias: ComptimeOption<Vector<E, NOut>> = bias.map(|b| b[channel_vector as usize]);
    let mut acc0: Vector<E, NOut> = base_bias.unwrap_or_else(|| Vector::zero());
    let mut acc1 = acc0;

    for k in 0..kernel_size {
        let index0 = (t0 * stride + k * dilation) as i32 - padding;
        let index1 = (t1 * stride + k * dilation) as i32 - padding;
        let valid0 = index0 >= 0 && (index0 as u32) < input_length;
        let valid1 = index1 >= 0 && (index1 as u32) < input_length && t1 < out_len;
        if valid0 || valid1 {
            for ci in range_stepped(0, in_channels_per_group, in_vector as u32) {
                let mut val0 = Vector::<E, NIn>::zero();
                let mut val1 = Vector::<E, NIn>::zero();
                if valid0 {
                    let input_index = batch as usize * input.stride(0)
                        + index0 as usize * input.stride(1)
                        + in_channel_start as usize
                        + ci as usize;
                    val0 = input[input_index / in_vector as usize];
                }
                if valid1 {
                    let input_index = batch as usize * input.stride(0)
                        + index1 as usize * input.stride(1)
                        + in_channel_start as usize
                        + ci as usize;
                    val1 = input[input_index / in_vector as usize];
                }

                #[unroll]
                for lane in 0..out_vector {
                    let oc = oc_start + lane as u32;
                    let weight_index = oc as usize * weight.stride(0)
                        + k as usize * weight.stride(1)
                        + ci as usize;
                    let filter = weight[weight_index / in_vector as usize];
                    let mut total0 = acc0.extract(lane);
                    let mut total1 = acc1.extract(lane);
                    #[unroll]
                    for j in 0..in_vector {
                        let w = filter.extract(j);
                        if valid0 {
                            total0 += val0.extract(j) * w;
                        }
                        if valid1 {
                            total1 += val1.extract(j) * w;
                        }
                    }
                    acc0.insert(lane, total0);
                    acc1.insert(lane, total1);
                }
            }
        }
    }

    let base = batch as usize * out_len as usize * channel_vectors as usize;
    output.write(
        base + t0 as usize * channel_vectors as usize + channel_vector as usize,
        acc0,
    );
    if t1 < out_len {
        output.write(
            base + t1 as usize * channel_vectors as usize + channel_vector as usize,
            acc1,
        );
    }
}

/// Experimental direct Conv1d. The caller supplies NHWC tensors with contiguous
/// channel axes, as required by the existing `launch_direct` path.
pub fn launch_temporal_pair(
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
    let dim_c = input.shape.len() - 1;
    let out_channels = weight.shape[0];
    if groups == 0 || out_channels % groups != 0 {
        return Err(ConvSetupError::Groups(groups));
    }
    if out.shape[0] == 0 || out.shape[1] == 0 || out_channels == 0 {
        return Ok(());
    }
    let channels_per_group = out_channels / groups;

    let mut grouped_shape = out.shape.clone();
    grouped_shape[dim_c] = channels_per_group;
    let vector_size_out = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &grouped_shape,
        &out.strides,
        dim_c,
    );
    if channels_per_group % vector_size_out != 0 {
        return Err(ConvSetupError::Unknown);
    }
    let vector_size_in = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &weight.shape,
        &weight.strides,
        weight.shape.len() - 1,
    );

    let pairs = out.shape[1].div_ceil(2);
    let work_items = out.shape[0] * pairs * (out_channels / vector_size_out);
    let cube_dim = CubeDim::new(client, work_items);
    let cube_count = calculate_cube_count_elemwise(client, work_items, cube_dim);
    let out_len = out.shape[1] as u32;
    let address_type = input
        .required_address_type(dtype.size())
        .max(weight.required_address_type(dtype.size()))
        .max(out.required_address_type(dtype.size()));

    unsafe {
        temporal_pair_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size_in,
            vector_size_out,
            input.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.map(|b| b.into_buffer_arg()).into(),
            linear_view(out),
            out_len,
            args.stride[0] as u32,
            args.dilation[0] as u32,
            args.padding[0] as i32,
            channels_per_group as u32,
            dtype,
        );
    }
    Ok(())
}
