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
    let value = matrix[idx];
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
    let matrix = matmul(
        columns,
        packed_weight,
        None,
        MatmulStrategy::default(),
        input_dtype,
    )
    .map_err(|_| ConvSetupError::Unknown)?;
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
        address_type!(matrix, bias, output),
        matrix.into_tensor_arg(),
        bias.map(|b| b.into_tensor_arg()).into(),
        output.clone().into_linear_view(),
        geometry,
        dtype_to_storage_type(output.dtype),
    );
    Ok(permute_nhwc_to_nchw(output))
}
