//! Single-input-channel Conv1d optimized for NSF harmonic source injection.
//!
//! Prepack the constant weights once into [1, kernel, Cout], then evaluate
//! two neighboring output samples from each filter vector in one pass.
//! This removes the generic 2D convolution path and its layout conversions.

use crate::{
    kernel::{into_contiguous_aligned, utils::address_type},
    ops::{max_vector_size_many, numeric::empty_device_dtype, permute, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{
    Shape,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
use cubecl::{
    calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearViewMut,
};
use cubek::convolution::components::ConvSetupError;

#[derive(CubeType, CubeLaunch)]
struct NoiseArgs {
    out_length: usize,
    stride: usize,
    dilation: usize,
    padding: i32,
}

#[cube(launch, address_type = "dynamic")]
fn noise_temporal_pair_kernel<E: Numeric, N: Size>(
    input: &Tensor<E>,
    weight: &Tensor<Vector<E, N>>,
    bias: ComptimeOption<&[Vector<E, N>]>,
    mut output: LinearViewMut<'_, Vector<E, N>>,
    args: NoiseArgs,
    #[define(E)] _dtype: ElemType,
) {
    let line = weight.vector_size();
    let cout = weight.shape(2);
    let channel_vectors = cout / line;
    let pair_count = (args.out_length + 1) / 2;
    let u = ABSOLUTE_POS;
    let channel_vector = u % channel_vectors;
    let pair_linear = u / channel_vectors;
    let batch = pair_linear / pair_count;
    let t0 = pair_linear % pair_count * 2;
    let t1 = t0 + 1;
    if batch >= input.shape(0) {
        terminate!();
    }

    let bias_sum: ComptimeOption<Vector<E, N>> = bias.map(|b| b[channel_vector]);
    let mut sum0 = bias_sum.unwrap_or_else(|| Vector::zero());
    let mut sum1 = sum0;
    let kernel = weight.shape(1);
    let in_len = input.shape(2) as i32;
    let base0 = (t0 * args.stride) as i32 - args.padding;
    let base1 = base0 + args.stride as i32;
    let batch_base = batch * input.stride(0);
    let oc = channel_vector * line;
    for k in 0..kernel {
        let x0 = base0 + (k * args.dilation) as i32;
        let x1 = base1 + (k * args.dilation) as i32;
        let valid0 = x0 >= 0 && x0 < in_len;
        let valid1 = t1 < args.out_length && x1 >= 0 && x1 < in_len;
        if valid0 || valid1 {
            let w = weight[(k * weight.stride(1) + oc) / line];
            if valid0 {
                let a = input[batch_base + x0 as usize * input.stride(2)];
                sum0 += Vector::broadcast(a) * w;
            }
            if valid1 {
                let a = input[batch_base + x1 as usize * input.stride(2)];
                sum1 += Vector::broadcast(a) * w;
            }
        }
    }
    let base_out = batch * args.out_length * channel_vectors + channel_vector;
    output.write(base_out + t0 * channel_vectors, sum0);
    if t1 < args.out_length {
        output.write(base_out + t1 * channel_vectors, sum1);
    }
}

/// Convert [Cout, 1, K] to a channel-contiguous [1, K, Cout] tensor once per model.
pub fn conv1d_single_input_prepack_weight(weight: CubeTensor) -> CubeTensor {
    into_contiguous_aligned(permute(weight, &[1, 2, 0]))
}

/// Perform single-input-channel 1D convolution using prepacked immutable weights.
/// The operation is equivalent to Conv1d for Cin=1 and groups=1.
pub fn conv1d_single_input_packed(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<1>,
) -> Result<CubeTensor, ConvSetupError> {
    let [batch, cin, time] = input.meta.shape().dims();
    let [packed_cin, kernel, cout] = weight.meta.shape().dims();
    if cin != 1
        || packed_cin != 1
        || cout == 0
        || kernel == 0
        || options.groups != 1
        || options.stride[0] == 0
        || options.dilation[0] == 0
    {
        return Err(ConvSetupError::Unknown);
    }
    let out_lengths = calculate_conv_output_sizes(
        &[kernel],
        &options.stride,
        &options.padding,
        &options.dilation,
        &[time],
    );
    let out_len = out_lengths[0];
    if out_len == 0 || batch == 0 {
        return Err(ConvSetupError::Unknown);
    }
    let out = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, out_len, cout]),
        input.dtype,
    );
    let line = max_vector_size_many(&[&weight, &out], 2).min(match &bias {
        Some(bias) => max_vector_size_many(&[bias], 0),
        None => VectorSize::MAX,
    });
    let work_items = batch * out_len.div_ceil(2) * (cout / line);
    let dim = CubeDim::new(&input.client, work_items);
    let count = calculate_cube_count_elemwise(&input.client, work_items, dim);
    let dtype = input.dtype;
    let client = input.client.clone();
    noise_temporal_pair_kernel::launch(
        &client,
        count,
        dim,
        address_type!(input, weight, bias, out),
        line,
        input.into_tensor_arg(),
        weight.into_tensor_arg(),
        bias.map(|v| v.into_buffer_arg()).into(),
        out.clone().into_linear_view(),
        NoiseArgsLaunch::new(
            out_len,
            options.stride[0],
            options.dilation[0],
            options.padding_begin()[0] as i32,
        ),
        dtype_to_storage_type(dtype),
    );
    Ok(permute_nhwc_to_nchw(out))
}
