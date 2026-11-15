//! Phase-domain batched matrix multiplication for inference ConvTranspose1d.
//!
//! The transposed convolution is partitioned by output phase modulo stride.
//! A single gather kernel builds [N, G*S, T, Cin/G*ceil(K/S)]; a single
//! batched matmul computes all phases; one scatter kernel applies bias and
//! restores the NCT view. The static weight matrix is packed once per model.
//! This exposes the dense K reduction to the CubeCL matmul scheduler and its
//! hardware-accelerated FP16 kernels rather than issuing one global load per
//! scalar multiply in each output lane.

use crate::{
    kernel::{
        matmul::{MatmulStrategy, matmul},
        utils::address_type,
    },
    ops::{numeric::empty_device_dtype, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::{
    Shape, TensorMetadata, cubecl::dtype_to_storage_type,
    ops::conv::calculate_conv_transpose_output_size,
};
use cubecl::{
    calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearViewMut,
};
use cubek::convolution::components::ConvSetupError;

#[derive(CubeType, CubeLaunch)]
struct Geometry {
    cin_group: usize,
    cout_group: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    reduction_taps: usize,
    input_time: usize,
    output_time: usize,
    phase_time: usize,
    groups: usize,
}

#[cube(launch, address_type = "dynamic")]
fn pack_phase_weight<E: Numeric>(
    weight: &Tensor<E>,
    mut output: LinearViewMut<'_, E>,
    args: Geometry,
    #[define(E)] _dtype: ElemType,
) {
    let p = ABSOLUTE_POS;
    let reduction = args.cin_group * args.reduction_taps;
    let total = args.groups * args.stride * reduction * args.cout_group;
    if p >= total {
        terminate!();
    }
    let oc = p % args.cout_group;
    let red = (p / args.cout_group) % reduction;
    let phase_group = p / (args.cout_group * reduction);
    let group = phase_group / args.stride;
    let phase = phase_group % args.stride;
    let channel = red / args.reduction_taps;
    let tap = red % args.reduction_taps;
    let kernel_phase = (phase + args.padding) % args.stride;
    let kernel_index = kernel_phase + tap * args.stride;
    let mut w = E::from_int(0);
    if kernel_index < args.kernel {
        let weight_index = (group * args.cin_group + channel) * weight.stride(0)
            + oc * weight.stride(1)
            + kernel_index * weight.stride(2);
        w = weight[weight_index];
    }
    output.write(p, w);
}

#[cube(launch, address_type = "dynamic")]
fn split_tf32_matrix(
    input: &Tensor<f32>,
    mut high: LinearViewMut<'_, f32>,
    mut low: LinearViewMut<'_, f32>,
    length: usize,
) {
    let p = ABSOLUTE_POS;
    if p >= length {
        terminate!();
    }
    let value = input[p];
    let bits = u32::reinterpret(value);
    let rounded_bits = (bits + 0x0000_0fff_u32 + ((bits >> 13) & 1_u32)) & 0xffff_e000_u32;
    let rounded = f32::reinterpret(rounded_bits);
    high.write(p, rounded);
    low.write(p, value - rounded);
}

fn split_tf32_tensor(input: CubeTensor) -> (CubeTensor, CubeTensor) {
    let shape = input.meta.shape().clone();
    let elements = shape.iter().product();
    let high = empty_device_dtype(
        input.client.clone(), input.device.clone(), shape.clone(), input.dtype,
    );
    let low = empty_device_dtype(
        input.client.clone(), input.device.clone(), shape, input.dtype,
    );
    let dim = CubeDim::new(&input.client, elements);
    let count = calculate_cube_count_elemwise(&input.client, elements, dim);
    let client = input.client.clone();
    split_tf32_matrix::launch(
        &client, count, dim, address_type!(input, high, low),
        input.into_tensor_arg(), high.clone().into_linear_view(),
        low.clone().into_linear_view(), elements,
    );
    (high, low)
}

#[cube(launch, address_type = "dynamic")]
fn gather_phases<E: Numeric>(
    input: &Tensor<E>,
    mut columns: LinearViewMut<'_, E>,
    args: Geometry,
    #[define(E)] _dtype: ElemType,
) {
    let p = ABSOLUTE_POS;
    let reduction = args.cin_group * args.reduction_taps;
    let phase_batch_size = args.groups * args.stride;
    let total = input.shape(0) * phase_batch_size * args.phase_time * reduction;
    if p >= total {
        terminate!();
    }
    let red = p % reduction;
    let phase_time = (p / reduction) % args.phase_time;
    let phase_group = (p / (reduction * args.phase_time)) % phase_batch_size;
    let batch = p / (reduction * args.phase_time * phase_batch_size);
    let group = phase_group / args.stride;
    let phase = phase_group % args.stride;
    let channel = red / args.reduction_taps;
    let tap = red % args.reduction_taps;
    let source = ((phase + args.padding) / args.stride) as i32 + phase_time as i32 - tap as i32;
    let mut x = E::from_int(0);
    if source >= 0 && source < args.input_time as i32 {
        let input_index = batch * input.stride(0)
            + (group * args.cin_group + channel) * input.stride(1)
            + (source as usize) * input.stride(2);
        x = input[input_index];
    }
    columns.write(p, x);
}

#[cube(launch, address_type = "dynamic")]
fn scatter_phases<E: Numeric>(
    matrix: &Tensor<E>,
    correction_a: ComptimeOption<&Tensor<E>>,
    correction_b: ComptimeOption<&Tensor<E>>,
    bias: ComptimeOption<&Tensor<E>>,
    mut output: LinearViewMut<'_, E>,
    args: Geometry,
    #[define(E)] _dtype: ElemType,
) {
    let p = ABSOLUTE_POS;
    let cout = args.groups * args.cout_group;
    let total = matrix.shape(0) * args.output_time * cout;
    if p >= total {
        terminate!();
    }
    let oc = p % cout;
    let time = (p / cout) % args.output_time;
    let batch = p / (cout * args.output_time);
    let group = oc / args.cout_group;
    let local_oc = oc % args.cout_group;
    let phase = time % args.stride;
    let phase_time = time / args.stride;
    let idx = batch * matrix.stride(0)
        + (group * args.stride + phase) * matrix.stride(1)
        + phase_time * matrix.stride(2)
        + local_oc * matrix.stride(3);
    let mut value = matrix[idx];
    #[comptime]
    match correction_a {
        ComptimeOption::Some(correction) => value += correction[idx],
        ComptimeOption::None => {}
    }
    #[comptime]
    match correction_b {
        ComptimeOption::Some(correction) => value += correction[idx],
        ComptimeOption::None => {}
    }
    #[comptime]
    match bias {
        ComptimeOption::Some(bias) => output.write(p, value + bias[oc]),
        ComptimeOption::None => output.write(p, value),
    }
}

fn make_geometry(
    weight: &CubeTensor,
    stride: usize,
    padding: usize,
    groups: usize,
    input_time: usize,
    output_time: usize,
) -> Result<GeometryLaunch, ConvSetupError> {
    let [cin, cout_group, kernel] = weight.meta.shape().dims();
    if stride == 0
        || groups == 0
        || cin == 0
        || cin % groups != 0
        || cout_group == 0
        || kernel == 0
        || output_time == 0
    {
        return Err(ConvSetupError::Unknown);
    }
    Ok(GeometryLaunch::new(
        cin / groups,
        cout_group,
        kernel,
        stride,
        padding,
        kernel.div_ceil(stride),
        input_time,
        output_time,
        output_time.div_ceil(stride),
        groups,
    ))
}

pub fn prepack_conv_transpose1d_phase_gemm_weight(
    weight: CubeTensor,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<CubeTensor, ConvSetupError> {
    let [cin, cout_group, kernel] = weight.meta.shape().dims();
    let geometry = make_geometry(&weight, stride, padding, groups, 1, stride)?;
    let reduction = (cin / groups) * kernel.div_ceil(stride);
    let out = empty_device_dtype(
        weight.client.clone(),
        weight.device.clone(),
        Shape::new([1, groups * stride, reduction, cout_group]),
        weight.dtype,
    );
    let work = groups * stride * reduction * cout_group;
    let dim = CubeDim::new(&weight.client, work);
    let count = calculate_cube_count_elemwise(&weight.client, work, dim);
    let client = weight.client.clone();
    pack_phase_weight::launch(
        &client,
        count,
        dim,
        address_type!(weight, out),
        weight.into_tensor_arg(),
        out.clone().into_linear_view(),
        geometry,
        dtype_to_storage_type(out.dtype),
    );
    Ok(out)
}

pub fn prepack_conv_transpose1d_phase_gemm_weight_compensated(
    weight: CubeTensor,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<(CubeTensor, CubeTensor), ConvSetupError> {
    if weight.dtype != burn_backend::DType::F32 {
        return Err(ConvSetupError::Unknown);
    }
    let packed = prepack_conv_transpose1d_phase_gemm_weight(weight, stride, padding, groups)?;
    Ok(split_tf32_tensor(packed))
}

pub fn conv_transpose1d_phase_gemm_packed(
    input: CubeTensor,
    packed_weight: CubeTensor,
    bias: Option<CubeTensor>,
    stride: usize,
    padding: usize,
    kernel: usize,
    dilation: usize,
    output_padding: usize,
    groups: usize,
) -> Result<CubeTensor, ConvSetupError> {
    conv_transpose1d_phase_gemm_impl(
        input, packed_weight, None, bias, stride, padding,
        kernel, dilation, output_padding, groups,
    )
}

pub fn conv_transpose1d_phase_gemm_packed_compensated(
    input: CubeTensor,
    packed_weight_high: CubeTensor,
    packed_weight_low: CubeTensor,
    bias: Option<CubeTensor>,
    stride: usize,
    padding: usize,
    kernel: usize,
    dilation: usize,
    output_padding: usize,
    groups: usize,
) -> Result<CubeTensor, ConvSetupError> {
    if input.dtype != burn_backend::DType::F32
        || packed_weight_high.dtype != burn_backend::DType::F32
        || packed_weight_low.dtype != burn_backend::DType::F32
        || packed_weight_high.meta.shape() != packed_weight_low.meta.shape()
    {
        return Err(ConvSetupError::Unknown);
    }
    conv_transpose1d_phase_gemm_impl(
        input, packed_weight_high, Some(packed_weight_low), bias,
        stride, padding, kernel, dilation, output_padding, groups,
    )
}

fn conv_transpose1d_phase_gemm_impl(
    input: CubeTensor,
    packed_weight: CubeTensor,
    packed_weight_low: Option<CubeTensor>,
    bias: Option<CubeTensor>,
    stride: usize,
    padding: usize,
    kernel: usize,
    dilation: usize,
    output_padding: usize,
    groups: usize,
) -> Result<CubeTensor, ConvSetupError> {
    if dilation != 1 || stride == 0 || groups == 0 || kernel == 0 {
        return Err(ConvSetupError::Unknown);
    }
    let [batch, cin, input_time] = input.meta.shape().dims();
    let [one, phase_groups, reduction, cout_group] = packed_weight.meta.shape().dims();
    let output_time = calculate_conv_transpose_output_size(
        kernel,
        stride,
        padding,
        output_padding,
        dilation,
        input_time,
    );
    if batch == 0
        || cin == 0
        || cin % groups != 0
        || one != 1
        || phase_groups != groups * stride
        || cout_group == 0
        || output_time == 0
        || reduction != cin / groups * kernel.div_ceil(stride)
        || input.dtype != packed_weight.dtype
        || bias
            .as_ref()
            .is_some_and(|b| b.dtype != input.dtype || b.meta.shape()[0] != groups * cout_group)
    {
        return Err(ConvSetupError::Unknown);
    }
    let phase_time = output_time.div_ceil(stride);
    let geometry = GeometryLaunch::new(
        cin / groups,
        cout_group,
        kernel,
        stride,
        padding,
        kernel.div_ceil(stride),
        input_time,
        output_time,
        phase_time,
        groups,
    );
    let input_dtype = input.dtype;
    let columns = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, groups * stride, phase_time, reduction]),
        input.dtype,
    );
    let gather_work = batch * groups * stride * phase_time * reduction;
    let dim = CubeDim::new(&input.client, gather_work);
    let count = calculate_cube_count_elemwise(&input.client, gather_work, dim);
    let client = input.client.clone();
    gather_phases::launch(
        &client,
        count,
        dim,
        address_type!(input, columns),
        input.into_tensor_arg(),
        columns.clone().into_linear_view(),
        GeometryLaunch::new(
            cin / groups,
            cout_group,
            kernel,
            stride,
            padding,
            kernel.div_ceil(stride),
            input_time,
            output_time,
            phase_time,
            groups,
        ),
        dtype_to_storage_type(columns.dtype),
    );
    let (matrix, correction_a, correction_b) = if let Some(weight_low) = packed_weight_low {
        let (columns_high, columns_low) = split_tf32_tensor(columns);
        let matrix = matmul(
            columns_high.clone(), packed_weight.clone(), None,
            MatmulStrategy::default(), input_dtype,
        ).map_err(|_| ConvSetupError::Unknown)?;
        let correction_a = matmul(
            columns_high, weight_low, None,
            MatmulStrategy::default(), input_dtype,
        ).map_err(|_| ConvSetupError::Unknown)?;
        let correction_b = matmul(
            columns_low, packed_weight, None,
            MatmulStrategy::default(), input_dtype,
        ).map_err(|_| ConvSetupError::Unknown)?;
        (matrix, Some(correction_a), Some(correction_b))
    } else {
        let matrix = matmul(
            columns, packed_weight, None,
            MatmulStrategy::default(), input_dtype,
        ).map_err(|_| ConvSetupError::Unknown)?;
        (matrix, None, None)
    };
    let cout = groups * cout_group;
    let output = empty_device_dtype(
        matrix.client.clone(),
        matrix.device.clone(),
        Shape::new([batch, output_time, cout]),
        matrix.dtype,
    );
    let scatter_work = batch * output_time * cout;
    let dim = CubeDim::new(&matrix.client, scatter_work);
    let count = calculate_cube_count_elemwise(&matrix.client, scatter_work, dim);
    let client = matrix.client.clone();
    scatter_phases::launch(
        &client,
        count,
        dim,
        address_type!(matrix, correction_a, correction_b, bias, output),
        matrix.into_tensor_arg(),
        correction_a.map(|v| v.into_tensor_arg()).into(),
        correction_b.map(|v| v.into_tensor_arg()).into(),
        bias.map(|b| b.into_tensor_arg()).into(),
        output.clone().into_linear_view(),
        geometry,
        dtype_to_storage_type(output.dtype),
    );
    Ok(permute_nhwc_to_nchw(output))
}
