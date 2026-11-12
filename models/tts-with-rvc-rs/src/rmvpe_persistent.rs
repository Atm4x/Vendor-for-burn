use burn::cubecl;
use burn::cubecl::prelude::*;
use burn::tensor::{DType, Tensor};
use burn_cubecl::tensor::CubeTensor;

type CubeBackend = burn_cubecl::Cube;

#[cube(launch_unchecked)]
fn bigru_kernel(
    input_gates: &[f32],
    hidden_weight: &[f32],
    hidden_bias: &[f32],
    output: &mut [f32],
    time: usize,
    hidden_size: usize,
    gate_batch_stride: usize,
    gate_time_stride: usize,
    gate_feature_stride: usize,
    weight_direction_stride: usize,
    weight_row_stride: usize,
    weight_column_stride: usize,
    bias_direction_stride: usize,
    bias_feature_stride: usize,
) {
    let lane = UNIT_POS as usize;
    let batch = CUBE_POS_X as usize;
    let direction = CUBE_POS_Y as usize;
    let reverse = direction == 1;
    let mut hidden = Shared::<[f32]>::new_slice(256usize);
    hidden[lane] = 0.0_f32;
    sync_cube();
    let bias_start = direction * bias_direction_stride;
    let weight_start = direction * weight_direction_stride;

    for step in 0..time {
        let frame = if reverse { time - 1 - step } else { step };
        let gate_start = batch * gate_batch_stride
            + frame * gate_time_stride
            + direction * hidden_size * 3 * gate_feature_stride;
        let mut hr = hidden_bias[bias_start + lane * bias_feature_stride];
        let mut hz = hidden_bias[bias_start + (hidden_size + lane) * bias_feature_stride];
        let mut hn = hidden_bias[bias_start + (2 * hidden_size + lane) * bias_feature_stride];
        for column in 0..hidden_size {
            let value = hidden[column];
            hr += value * hidden_weight[weight_start + lane * weight_row_stride + column * weight_column_stride];
            hz += value * hidden_weight[weight_start + (hidden_size + lane) * weight_row_stride + column * weight_column_stride];
            hn += value * hidden_weight[weight_start + (2 * hidden_size + lane) * weight_row_stride + column * weight_column_stride];
        }
        let reset = 1.0_f32 / (1.0_f32 + (0.0_f32 - (input_gates[gate_start + lane * gate_feature_stride] + hr)).exp());
        let update = 1.0_f32 / (1.0_f32 + (0.0_f32 - (input_gates[gate_start + (hidden_size + lane) * gate_feature_stride] + hz)).exp());
        let candidate = (input_gates[gate_start + (2 * hidden_size + lane) * gate_feature_stride] + reset * hn).tanh();
        let next = (1.0_f32 - update) * candidate + update * hidden[lane];
        sync_cube();
        hidden[lane] = next;
        output[(batch * time + frame) * hidden_size * 2 + direction * hidden_size + lane] = next;
        sync_cube();
    }
}

pub(crate) fn forward(
    gates: Tensor<3>,
    weight: Tensor<3>,
    bias: Tensor<2>,
) -> Result<Tensor<3>, String> {
    let [batch, time, gates_count] = gates.dims();
    let [directions, weight_rows, hidden] = weight.dims();
    if batch == 0 || time == 0 || hidden != 256 || directions != 2
        || gates_count != hidden * 6 || weight_rows != hidden * 3
        || bias.dims() != [2, hidden * 3]
    {
        return Err("persistent BiGRU requires [B,T,1536] gates and [2,768,256] weights".into());
    }
    if batch > u32::MAX as usize {
        return Err("persistent BiGRU batch exceeds CUDA grid limits".into());
    }
    let input = gates.try_into_primitive::<CubeBackend>()
        .map_err(|e| format!("persistent BiGRU gates require CubeCL: {e:?}"))?;
    let weight = weight.try_into_primitive::<CubeBackend>()
        .map_err(|e| format!("persistent BiGRU weights require CubeCL: {e:?}"))?;
    let bias = bias.try_into_primitive::<CubeBackend>()
        .map_err(|e| format!("persistent BiGRU bias requires CubeCL: {e:?}"))?;
    if input.dtype != DType::F32 || weight.dtype != DType::F32 || bias.dtype != DType::F32
        || input.meta.is_tiled() || weight.meta.is_tiled() || bias.meta.is_tiled()
    {
        return Err("persistent BiGRU requires non-tiled FP32 CubeCL tensors".into());
    }
    let strides = input.meta.strides();
    let weights_strides = weight.meta.strides();
    let bias_strides = bias.meta.strides();
    let bytes = batch.checked_mul(time).and_then(|v| v.checked_mul(hidden * 2))
        .and_then(|v| v.checked_mul(4)).ok_or("persistent BiGRU output overflow")?;
    let output = input.client.empty(bytes);
    unsafe {
        bigru_kernel::launch_unchecked(
            &input.client,
            CubeCount::Static(batch as u32, 2, 1),
            CubeDim::new_1d(hidden as u32),
            BufferArg::from_raw_parts(input.handle.clone(), input.handle.size_in_used() as usize / 4),
            BufferArg::from_raw_parts(weight.handle.clone(), weight.handle.size_in_used() as usize / 4),
            BufferArg::from_raw_parts(bias.handle.clone(), bias.handle.size_in_used() as usize / 4),
            BufferArg::from_raw_parts(output.clone(), bytes / 4),
            time,
            hidden,
            strides[0], strides[1], strides[2],
            weights_strides[0], weights_strides[1], weights_strides[2],
            bias_strides[0], bias_strides[1],
        );
    }
    Ok(Tensor::from_primitive::<CubeBackend>(CubeTensor::new_contiguous(
        input.client, input.device, [batch, time, hidden * 2].into(), output, DType::F32,
    )))
}
