use cubecl::{
    calculate_cube_count_elemwise,
    client::Client,
    prelude::*,
    std::tensor::layout::linear::{LinearViewMut, linear_view},
    tensor_vector_size_parallel,
};

use crate::components::ConvSetupError;

#[cube(launch_unchecked, address_type = "dynamic")]
fn preact_residual_temporal_kernel<E: Float, NOut: Size>(
    input: &Tensor<E>,
    weight: &Tensor<Vector<E, NOut>>,
    bias: ComptimeOption<&[Vector<E, NOut>]>,
    residual: ComptimeOption<&[Vector<E, NOut>]>,
    mut output: LinearViewMut<'_, Vector<E, NOut>>,
    out_len: u32,
    stride: u32,
    dilation: u32,
    padding: i32,
    #[define(E)] _dtype: ElemType,
) {
    let lanes = output.vector_size() as u32;
    let channels = weight.shape(3) as u32;
    let vectors_per_position = channels / lanes;
    let pairs = (out_len + 1) / 2;
    let unit = ABSOLUTE_POS as u32;
    if unit >= input.shape(0) as u32 * pairs * vectors_per_position {
        terminate!();
    }

    let output_vector = unit % vectors_per_position;
    let pair = unit / vectors_per_position;
    let batch = pair / pairs;
    let t0 = (pair % pairs) * 2;
    let t1 = t0 + 1;
    let cin = weight.shape(2) as u32;
    let kernel_size = weight.shape(1) as u32;
    let length = input.shape(1) as u32;
    let mut sum0 = Vector::<E, NOut>::zero();
    let mut sum1 = Vector::<E, NOut>::zero();

    for k in 0..kernel_size {
        let i0 = (t0 * stride + k * dilation) as i32 - padding;
        let i1 = (t1 * stride + k * dilation) as i32 - padding;
        let valid0 = i0 >= 0 && (i0 as u32) < length;
        let valid1 = t1 < out_len && i1 >= 0 && (i1 as u32) < length;
        if valid0 || valid1 {
            for ci in 0..cin {
                let w_offset = k as usize * weight.stride(1)
                    + ci as usize * weight.stride(2)
                    + (output_vector * lanes) as usize;
                let w = weight[w_offset / lanes as usize];
                if valid0 {
                    let x_offset = batch as usize * input.stride(0)
                        + i0 as usize * input.stride(1)
                        + ci as usize * input.stride(2);
                    let x = input[x_offset];
                    let activated = if x >= E::from_int(0) {
                        x
                    } else {
                        x * (E::from_int(1) / E::from_int(10))
                    };
                    sum0 += w * Vector::new(activated);
                }
                if valid1 {
                    let x_offset = batch as usize * input.stride(0)
                        + i1 as usize * input.stride(1)
                        + ci as usize * input.stride(2);
                    let x = input[x_offset];
                    let activated = if x >= E::from_int(0) {
                        x
                    } else {
                        x * (E::from_int(1) / E::from_int(10))
                    };
                    sum1 += w * Vector::new(activated);
                }
            }
        }
    }

    let base = batch as usize * out_len as usize * vectors_per_position as usize;
    let index0 = base + t0 as usize * vectors_per_position as usize + output_vector as usize;
    let index1 = base + t1 as usize * vectors_per_position as usize + output_vector as usize;
    let base_bias: ComptimeOption<Vector<E, NOut>> = bias.map(|b| b[output_vector as usize]);
    let b = base_bias.unwrap_or_else(|| Vector::zero());
    sum0 += b;
    sum1 += b;
    let base_residual: ComptimeOption<Vector<E, NOut>> = residual.map(|r| r[index0]);
    sum0 += base_residual.unwrap_or_else(|| Vector::zero());
    if t1 < out_len {
        let next_residual: ComptimeOption<Vector<E, NOut>> = residual.map(|r| r[index1]);
        sum1 += next_residual.unwrap_or_else(|| Vector::zero());
    }
    output.write(index0, sum0);
    if t1 < out_len {
        output.write(index1, sum1);
    }
}

pub fn launch_preact_residual_temporal(
    client: &Client,
    input: TensorBinding,
    weight: TensorBinding,
    bias: Option<TensorBinding>,
    residual: Option<TensorBinding>,
    output: TensorBinding,
    stride: usize,
    dilation: usize,
    padding: usize,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    if input.shape.len() != 3
        || weight.shape.len() != 4
        || output.shape.len() != 3
        || weight.shape[0] != 1
        || input.shape[2] != weight.shape[2]
        || weight.shape[3] != output.shape[2]
        || input.shape[0] != output.shape[0]
        || stride != 1
        || input.shape[1] != output.shape[1]
        || input.shape.iter().any(|&n| n == 0)
        || weight.shape.iter().any(|&n| n == 0)
    {
        return Err(ConvSetupError::Unknown);
    }
    if let Some(r) = &residual {
        if r.shape != output.shape
            || r.strides[2] != 1
            || r.strides[1] != output.shape[2]
            || r.strides[0] != output.shape[1] * output.shape[2]
        {
            return Err(ConvSetupError::Unknown);
        }
    }
    if weight.strides[3] != 1
        || weight.strides[2] != weight.shape[3]
        || weight.strides[1] != weight.shape[2] * weight.shape[3]
    {
        return Err(ConvSetupError::Unknown);
    }
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &output.shape,
        &output.strides,
        2,
    );
    if output.shape[2] % vector_size != 0 {
        return Err(ConvSetupError::Unknown);
    }
    let work = input.shape[0] * output.shape[1].div_ceil(2) * (output.shape[2] / vector_size);
    let cube_dim = CubeDim::new(client, work);
    let cube_count = calculate_cube_count_elemwise(client, work, cube_dim);
    let mut address_type = input
        .required_address_type(dtype.size())
        .max(weight.required_address_type(dtype.size()))
        .max(output.required_address_type(dtype.size()));
    if let Some(r) = &residual {
        address_type = address_type.max(r.required_address_type(dtype.size()));
    }
    let length = input.shape[1] as u32;
    unsafe {
        preact_residual_temporal_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            input.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.map(|x| x.into_buffer_arg()).into(),
            residual.map(|x| x.into_buffer_arg()).into(),
            linear_view(output),
            length,
            stride as u32,
            dilation as u32,
            padding as i32,
            dtype,
        );
    }
    Ok(())
}
