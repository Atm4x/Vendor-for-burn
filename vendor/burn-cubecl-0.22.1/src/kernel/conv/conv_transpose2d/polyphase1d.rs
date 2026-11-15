//! One-pass polyphase ConvTranspose1d for inference.
//!
//! Each lane owns an output-channel vector and two output samples with the same
//! stride phase. Both outputs consume one filter vector: no zero-inserted input,
//! im2col buffer, scatter atomics, or inner-loop modulus/division. The NCT API
//! is preserved through metadata-only dimension permutations.

use crate::{
    kernel::{into_contiguous_aligned, utils::address_type},
    ops::{
        max_vector_size_many, numeric::empty_device_dtype, permute_nchw_to_nhwc,
        permute_nhwc_to_nchw,
    },
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{Shape, ops::conv::calculate_conv_transpose_output_size};
use cubecl::{
    calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearViewMut,
};
use cubek::convolution::components::ConvSetupError;

#[derive(CubeType, CubeLaunch)]
struct PolyphaseArgs {
    out_length: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    pairs_per_phase: usize,
}

#[cube(launch, address_type = "dynamic")]
fn polyphase_pair_kernel<E: Numeric, N: Size>(
    input: &Tensor<E>,
    weight: &Tensor<Vector<E, N>>,
    bias: ComptimeOption<&[Vector<E, N>]>,
    mut output: LinearViewMut<'_, Vector<E, N>>,
    args: PolyphaseArgs,
    #[define(E)] _dtype: ElemType,
) {
    let line = weight.vector_size();
    let out_channels_per_group = weight.shape(2);
    let out_channels = out_channels_per_group * args.groups;
    let vectors_per_time = out_channels / line;
    let vector_index = ABSOLUTE_POS % vectors_per_time;
    let temporal_index = ABSOLUTE_POS / vectors_per_time;
    let phase_pair = temporal_index % args.pairs_per_phase;
    let batch_phase = temporal_index / args.pairs_per_phase;
    let phase = batch_phase % args.stride;
    let batch = batch_phase / args.stride;
    let t0 = phase + phase_pair * args.stride * 2;
    let t1 = t0 + args.stride;
    if batch >= input.shape(0) || t0 >= args.out_length {
        terminate!();
    }

    let oc = vector_index * line;
    let group = oc / out_channels_per_group;
    let local_oc = oc % out_channels_per_group;
    let cin_per_group = input.shape(2) / args.groups;
    let cin_begin = group * cin_per_group;
    let in_length = input.shape(1);
    let kernel = weight.shape(1);
    let base = t0 + args.padding;
    let filter_phase = base % args.stride;
    let source_t = base / args.stride;
    let taps = (kernel + args.stride - 1 - filter_phase) / args.stride;

    let bvec: ComptimeOption<Vector<E, N>> = bias.map(|b| b[vector_index]);
    let mut sum0 = bvec.unwrap_or_else(|| Vector::zero());
    let mut sum1 = sum0;
    for tap in 0..taps {
        let filter_k = filter_phase + tap * args.stride;
        let valid0 = source_t >= tap && source_t - tap < in_length;
        let valid1 = t1 < args.out_length && source_t + 1 >= tap && source_t + 1 - tap < in_length;
        if valid0 || valid1 {
            let mut x0 = 0usize.runtime();
            let mut x1 = 0usize.runtime();
            if valid0 {
                x0 = batch * input.stride(0)
                    + (source_t - tap) * input.stride(1)
                    + cin_begin * input.stride(2);
            }
            if valid1 {
                x1 = batch * input.stride(0)
                    + (source_t + 1 - tap) * input.stride(1)
                    + cin_begin * input.stride(2);
            }
            let wbase = filter_k * weight.stride(1) + local_oc;
            for ci in 0..cin_per_group {
                let w = weight[(wbase + (cin_begin + ci) * weight.stride(0)) / line];
                if valid0 {
                    let a = input[x0 + ci * input.stride(2)];
                    sum0 += Vector::broadcast(a) * w;
                }
                if valid1 {
                    let a = input[x1 + ci * input.stride(2)];
                    sum1 += Vector::broadcast(a) * w;
                }
            }
        }
    }
    let out_base = batch * args.out_length * vectors_per_time + vector_index;
    output.write(out_base + t0 * vectors_per_time, sum0);
    if t1 < args.out_length {
        output.write(out_base + t1 * vectors_per_time, sum1);
    }
}

#[derive(CubeType, CubeLaunch)]
struct PolyphaseSingleArgs {
    out_length: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    samples_per_phase: usize,
}

/// One output time sample per lane: twice the CTAs of temporal-pair, one live
/// accumulator and no cross-sample dependency. The hot interior is bounds-free.
#[cube(launch, address_type = "dynamic")]
fn polyphase_single_kernel<E: Numeric, N: Size>(
    input: &Tensor<E>,
    weight: &Tensor<Vector<E, N>>,
    bias: ComptimeOption<&[Vector<E, N>]>,
    mut output: LinearViewMut<'_, Vector<E, N>>,
    args: PolyphaseSingleArgs,
    #[define(E)] _dtype: ElemType,
) {
    let line = weight.vector_size();
    let cout_group = weight.shape(2);
    let cout = cout_group * args.groups;
    let vectors_per_time = cout / line;
    let channel_vector = ABSOLUTE_POS % vectors_per_time;
    let temporal_index = ABSOLUTE_POS / vectors_per_time;
    let phase_time = temporal_index % args.samples_per_phase;
    let batch_phase = temporal_index / args.samples_per_phase;
    let phase = batch_phase % args.stride;
    let batch = batch_phase / args.stride;
    let time = phase + phase_time * args.stride;
    if batch >= input.shape(0) || time >= args.out_length {
        terminate!();
    }

    let oc = channel_vector * line;
    let group = oc / cout_group;
    let local_oc = oc % cout_group;
    let cin_group = input.shape(2) / args.groups;
    let cin_begin = group * cin_group;
    let input_length = input.shape(1);
    let kernel = weight.shape(1);
    let base = time + args.padding;
    let filter_phase = base % args.stride;
    let source_time = base / args.stride;
    let taps = (kernel + args.stride - 1 - filter_phase) / args.stride;
    let bias_value: ComptimeOption<Vector<E, N>> = bias.map(|b| b[channel_vector]);
    let mut sum = bias_value.unwrap_or_else(|| Vector::zero());
    let weight_stride0 = weight.stride(0);
    let weight_stride1 = weight.stride(1);
    let input_stride0 = input.stride(0);
    let input_stride1 = input.stride(1);
    let input_stride2 = input.stride(2);
    let weight_channel_base = cin_begin * weight_stride0 + local_oc;
    let input_channel_base = batch * input_stride0 + cin_begin * input_stride2;

    let enough_source = if taps > 0 {
        source_time >= taps - 1
    } else {
        false
    };
    if enough_source && source_time < input_length {
        for tap in 0..taps {
            let source_index = source_time - tap;
            let xbase = input_channel_base + source_index * input_stride1;
            let wbase = weight_channel_base + (filter_phase + tap * args.stride) * weight_stride1;
            for ci in 0..cin_group {
                let w = weight[(wbase + ci * weight_stride0) / line];
                let x = input[xbase + ci * input_stride2];
                sum += Vector::broadcast(x) * w;
            }
        }
    } else {
        for tap in 0..taps {
            if source_time >= tap && source_time - tap < input_length {
                let xbase = input_channel_base + (source_time - tap) * input_stride1;
                let wbase =
                    weight_channel_base + (filter_phase + tap * args.stride) * weight_stride1;
                for ci in 0..cin_group {
                    let w = weight[(wbase + ci * weight_stride0) / line];
                    let x = input[xbase + ci * input_stride2];
                    sum += Vector::broadcast(x) * w;
                }
            }
        }
    }
    let out_base = batch * args.out_length * vectors_per_time + channel_vector;
    output.write(out_base + time * vectors_per_time, sum);
}

/// Alternative full-output FP32 polyphase kernel, specializing in maximum warp
/// availability rather than cross-time weight reuse. Uses already packed weights.
pub fn conv_transpose1d_polyphase_single_packed(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    padding_out: usize,
    groups: usize,
) -> Result<CubeTensor, ConvSetupError> {
    if groups == 0 || stride == 0 || dilation != 1 {
        return Err(ConvSetupError::Unknown);
    }
    let [batch, cin, in_len] = input.meta.shape().dims();
    let [weight_cin, kernel, cout_group] = weight.meta.shape().dims();
    if cin != weight_cin || cin % groups != 0 || cout_group == 0 || kernel == 0 {
        return Err(ConvSetupError::Unknown);
    }
    let out_len = calculate_conv_transpose_output_size(
        kernel,
        stride,
        padding,
        padding_out,
        dilation,
        in_len,
    );
    if out_len == 0 || batch == 0 || cin == 0 {
        return Err(ConvSetupError::Unknown);
    }
    let cout = cout_group * groups;
    let input = permute_nchw_to_nhwc(input);
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
    let samples_per_phase = out_len.div_ceil(stride);
    let work_items = batch * stride * samples_per_phase * (cout / line);
    let dim = CubeDim::new(&input.client, work_items);
    let count = calculate_cube_count_elemwise(&input.client, work_items, dim);
    let dtype = input.dtype;
    let client = input.client.clone();
    polyphase_single_kernel::launch(
        &client,
        count,
        dim,
        address_type!(input, weight, bias, out),
        line,
        input.into_tensor_arg(),
        weight.into_tensor_arg(),
        bias.map(|b| b.into_buffer_arg()).into(),
        out.clone().into_linear_view(),
        PolyphaseSingleArgsLaunch::new(out_len, stride, padding, groups, samples_per_phase),
        dtype_to_storage_type(dtype),
    );
    Ok(permute_nhwc_to_nchw(out))
}

/// A dedicated FP32-friendly transposed 1D convolution, using output-channel
/// vectorization and two time positions of the same stride phase per lane.
/// Input is [N, Cin, T], weight [Cin, Cout/groups, K], output [N, Cout, Tout].
/// This is an exact spatial-domain algorithm, not a GEMM or Winograd approximation.
pub fn conv_transpose1d_polyphase(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    padding_out: usize,
    groups: usize,
) -> Result<CubeTensor, ConvSetupError> {
    let packed_weight = conv_transpose1d_prepack_weight(weight);
    conv_transpose1d_polyphase_packed(
        input,
        packed_weight,
        bias,
        stride,
        padding,
        dilation,
        padding_out,
        groups,
    )
}

/// Prepack fixed weights once when constructing an inference model.
/// Converts [Cin, Cout/groups, K] to channel-contiguous [Cin, K, Cout/groups].
pub fn conv_transpose1d_prepack_weight(weight: CubeTensor) -> CubeTensor {
    into_contiguous_aligned(permute_nchw_to_nhwc(weight))
}

/// Use the cached prepacked weights, avoiding a layout conversion per inference.
pub fn conv_transpose1d_polyphase_packed(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    padding_out: usize,
    groups: usize,
) -> Result<CubeTensor, ConvSetupError> {
    if groups == 0 || stride == 0 || dilation != 1 {
        return Err(ConvSetupError::Unknown);
    }
    let [batch, cin, in_len] = input.meta.shape().dims();
    let [weight_cin, kernel, cout_group] = weight.meta.shape().dims();
    if cin != weight_cin || cin % groups != 0 || cout_group == 0 || kernel == 0 {
        return Err(ConvSetupError::Unknown);
    }
    let out_len = calculate_conv_transpose_output_size(
        kernel,
        stride,
        padding,
        padding_out,
        dilation,
        in_len,
    );
    if out_len == 0 || batch == 0 || cin == 0 {
        return Err(ConvSetupError::Unknown);
    }
    let cout = cout_group * groups;
    let input = permute_nchw_to_nhwc(input);
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
    let pairs_per_phase = out_len.div_ceil(stride * 2);
    let work_items = batch * stride * pairs_per_phase * (cout / line);
    let dim = CubeDim::new(&input.client, work_items);
    let count = calculate_cube_count_elemwise(&input.client, work_items, dim);
    let dtype = input.dtype;
    let client = input.client.clone();
    polyphase_pair_kernel::launch(
        &client,
        count,
        dim,
        address_type!(input, weight, bias, out),
        line,
        input.into_tensor_arg(),
        weight.into_tensor_arg(),
        bias.map(|b| b.into_buffer_arg()).into(),
        out.clone().into_linear_view(),
        PolyphaseArgsLaunch::new(out_len, stride, padding, groups, pairs_per_phase),
        dtype_to_storage_type(dtype),
    );
    Ok(permute_nhwc_to_nchw(out))
}
