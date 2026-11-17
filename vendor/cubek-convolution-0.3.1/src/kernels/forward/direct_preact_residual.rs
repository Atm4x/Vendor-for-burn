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
    #[comptime] temporal_tile: u32,
    #[comptime] broadcast_inputs: bool,
    #[comptime] input_activated: bool,
    #[comptime] output_activated: bool,
    stride: u32,
    dilation: u32,
    padding: i32,
    #[define(E)] _dtype: ElemType,
) {
    let lanes = output.vector_size() as u32;
    let channels = weight.shape(3) as u32;
    let vectors_per_position = channels / lanes;
    let pairs = (out_len + temporal_tile - 1) / temporal_tile;
    let unit = ABSOLUTE_POS as u32;
    if unit >= input.shape(0) as u32 * pairs * vectors_per_position {
        terminate!();
    }

    let output_vector = unit % vectors_per_position;
    let pair = unit / vectors_per_position;
    let batch = pair / pairs;
    let t0 = (pair % pairs) * temporal_tile;
    let t1 = t0 + 1;
    let t2 = t0 + 2;
    let t3 = t0 + 3;
    let cin = weight.shape(2) as u32;
    let kernel_size = weight.shape(1) as u32;
    let length = input.shape(1) as u32;
    let mut sum0 = Vector::<E, NOut>::zero();
    let mut sum1 = Vector::<E, NOut>::zero();
    let mut sum2 = Vector::<E, NOut>::zero();
    let mut sum3 = Vector::<E, NOut>::zero();

    for k in 0..kernel_size {
        let i0 = (t0 * stride + k * dilation) as i32 - padding;
        let i1 = (t1 * stride + k * dilation) as i32 - padding;
        let i2 = (t2 * stride + k * dilation) as i32 - padding;
        let i3 = (t3 * stride + k * dilation) as i32 - padding;
        let valid0 = i0 >= 0 && (i0 as u32) < length;
        let valid1 = t1 < out_len && i1 >= 0 && (i1 as u32) < length;
        let valid2 = temporal_tile == 4 && t2 < out_len && i2 >= 0 && (i2 as u32) < length;
        let valid3 = temporal_tile == 4 && t3 < out_len && i3 >= 0 && (i3 as u32) < length;
        if valid0 || valid1 || valid2 || valid3 {
            for ci in 0..cin {
                let w_offset = k as usize * weight.stride(1)
                    + ci as usize * weight.stride(2)
                    + (output_vector * lanes) as usize;
                let w = weight[w_offset / lanes as usize];
                if valid0 {
                    let x_offset = batch as usize * input.stride(0)
                        + i0 as usize * input.stride(1)
                        + ci as usize * input.stride(2);
                    let x = if broadcast_inputs && PLANE_DIM == 32 {
                        let mut lane_value = E::from_int(0);
                        if UNIT_POS_PLANE == 0 {
                            lane_value = input[x_offset];
                        }
                        plane_broadcast(lane_value, 0u32)
                    } else {
                        input[x_offset]
                    };
                    let activated = if input_activated || x >= E::from_int(0) {
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
                    let x = if broadcast_inputs && PLANE_DIM == 32 {
                        let mut lane_value = E::from_int(0);
                        if UNIT_POS_PLANE == 0 {
                            lane_value = input[x_offset];
                        }
                        plane_broadcast(lane_value, 0u32)
                    } else {
                        input[x_offset]
                    };
                    let activated = if input_activated || x >= E::from_int(0) {
                        x
                    } else {
                        x * (E::from_int(1) / E::from_int(10))
                    };
                    sum1 += w * Vector::new(activated);
                }
                if valid2 {
                    let x_offset = batch as usize * input.stride(0)
                        + i2 as usize * input.stride(1)
                        + ci as usize * input.stride(2);
                    let x = if broadcast_inputs && PLANE_DIM == 32 {
                        let mut lane_value = E::from_int(0);
                        if UNIT_POS_PLANE == 0 {
                            lane_value = input[x_offset];
                        }
                        plane_broadcast(lane_value, 0u32)
                    } else {
                        input[x_offset]
                    };
                    let activated = if input_activated || x >= E::from_int(0) {
                        x
                    } else {
                        x * (E::from_int(1) / E::from_int(10))
                    };
                    sum2 += w * Vector::new(activated);
                }
                if valid3 {
                    let x_offset = batch as usize * input.stride(0)
                        + i3 as usize * input.stride(1)
                        + ci as usize * input.stride(2);
                    let x = if broadcast_inputs && PLANE_DIM == 32 {
                        let mut lane_value = E::from_int(0);
                        if UNIT_POS_PLANE == 0 {
                            lane_value = input[x_offset];
                        }
                        plane_broadcast(lane_value, 0u32)
                    } else {
                        input[x_offset]
                    };
                    let activated = if input_activated || x >= E::from_int(0) {
                        x
                    } else {
                        x * (E::from_int(1) / E::from_int(10))
                    };
                    sum3 += w * Vector::new(activated);
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
    if temporal_tile == 4 {
        sum2 += b;
        sum3 += b;
    }
    let base_residual: ComptimeOption<Vector<E, NOut>> = residual.map(|r| r[index0]);
    sum0 += base_residual.unwrap_or_else(|| Vector::zero());
    if t1 < out_len {
        let next_residual: ComptimeOption<Vector<E, NOut>> = residual.map(|r| r[index1]);
        sum1 += next_residual.unwrap_or_else(|| Vector::zero());
    }
    if temporal_tile == 4 {
        let index2 = base + t2 as usize * vectors_per_position as usize + output_vector as usize;
        let index3 = base + t3 as usize * vectors_per_position as usize + output_vector as usize;
        if t2 < out_len {
            let residual2: ComptimeOption<Vector<E, NOut>> = residual.map(|r| r[index2]);
            sum2 += residual2.unwrap_or_else(|| Vector::zero());
            if output_activated {
                sum2 = select_many(
                    sum2.greater_equal(&Vector::<E, NOut>::zero()),
                    sum2,
                    sum2 * Vector::new(E::from_int(1) / E::from_int(10)),
                );
            }
            output.write(index2, sum2);
        }
        if t3 < out_len {
            let residual3: ComptimeOption<Vector<E, NOut>> = residual.map(|r| r[index3]);
            sum3 += residual3.unwrap_or_else(|| Vector::zero());
            if output_activated {
                sum3 = select_many(
                    sum3.greater_equal(&Vector::<E, NOut>::zero()),
                    sum3,
                    sum3 * Vector::new(E::from_int(1) / E::from_int(10)),
                );
            }
            output.write(index3, sum3);
        }
    }
    if output_activated {
        sum0 = select_many(
            sum0.greater_equal(&Vector::<E, NOut>::zero()),
            sum0,
            sum0 * Vector::new(E::from_int(1) / E::from_int(10)),
        );
    }
    output.write(index0, sum0);
    if t1 < out_len {
        if output_activated {
            sum1 = select_many(
                sum1.greater_equal(&Vector::<E, NOut>::zero()),
                sum1,
                sum1 * Vector::new(E::from_int(1) / E::from_int(10)),
            );
        }
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
    temporal_tile: usize,
    broadcast_inputs: bool,
    input_activated: bool,
    output_activated: bool,
    dtype: ElemType,
) -> Result<(), ConvSetupError> {
    if output_activated && residual.is_some() {
        return Err(ConvSetupError::Unknown);
    }
    if input.shape.len() != 3
        || weight.shape.len() != 4
        || output.shape.len() != 3
        || weight.shape[0] != 1
        || input.shape[2] != weight.shape[2]
        || weight.shape[3] != output.shape[2]
        || input.shape[0] != output.shape[0]
        || (temporal_tile != 2 && temporal_tile != 4)
        || (broadcast_inputs && (temporal_tile != 4 || output.shape[2] != 128))
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
    let supported_vectors = client.io_optimized_vector_sizes(dtype.size()).collect::<Vec<_>>();
    let vector_size = if broadcast_inputs {
        if !supported_vectors.contains(&4) {
            return Err(ConvSetupError::Unknown);
        }
        4
    } else {
        tensor_vector_size_parallel(
            supported_vectors.into_iter(),
            &output.shape,
            &output.strides,
            2,
        )
    };
    if output.shape[2] % vector_size != 0 {
        return Err(ConvSetupError::Unknown);
    }
    let work = input.shape[0] * output.shape[1].div_ceil(temporal_tile) * (output.shape[2] / vector_size);
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
            temporal_tile as u32,
            broadcast_inputs,
            input_activated,
            output_activated,
            stride as u32,
            dilation as u32,
            padding as i32,
            dtype,
        );
    }
    Ok(())
}
