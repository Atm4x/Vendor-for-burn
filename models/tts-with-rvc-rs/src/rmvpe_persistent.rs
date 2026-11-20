use burn::cubecl;
use burn::cubecl::prelude::*;
use burn::tensor::{DType, Tensor};
use burn_cubecl::{kernel::into_contiguous, tensor::CubeTensor};

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
    #[comptime] weight_coalesced: bool,
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
            let first = if weight_coalesced {
                weight_start + column * weight_row_stride + lane * weight_column_stride
            } else {
                weight_start + lane * weight_row_stride + column * weight_column_stride
            };
            let gate_stride = if weight_coalesced {
                hidden_size * weight_column_stride
            } else {
                hidden_size * weight_row_stride
            };
            hr += value * hidden_weight[first];
            hz += value * hidden_weight[first + gate_stride];
            hn += value * hidden_weight[first + 2 * gate_stride];
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

#[cube(launch_unchecked)]
fn bigru_gate_parallel_kernel(
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
    let thread = UNIT_POS as usize;
    let lane = thread % hidden_size;
    let gate = thread / hidden_size;
    let batch = CUBE_POS_X as usize;
    let direction = CUBE_POS_Y as usize;
    let reverse = direction == 1;
    let mut hidden = Shared::<[f32]>::new_slice(256usize);
    let mut recurrent = Shared::<[f32]>::new_slice(768usize);
    if gate == 0 {
        hidden[lane] = 0.0_f32;
    }
    sync_cube();

    let bias_start = direction * bias_direction_stride;
    let weight_start = direction * weight_direction_stride;
    let gate_offset = gate * hidden_size;
    for step in 0..time {
        let frame = if reverse { time - 1 - step } else { step };
        let gate_start = batch * gate_batch_stride
            + frame * gate_time_stride
            + direction * hidden_size * 3 * gate_feature_stride;
        let mut projected = hidden_bias[bias_start + (gate_offset + lane) * bias_feature_stride];
        for column in 0..hidden_size {
            let value = hidden[column];
            let first = weight_start + column * weight_row_stride
                + (gate_offset + lane) * weight_column_stride;
            projected += value * hidden_weight[first];
        }
        recurrent[gate_offset + lane] = projected;
        sync_cube();
        if gate == 0 {
            let hr = recurrent[lane];
            let hz = recurrent[hidden_size + lane];
            let hn = recurrent[2 * hidden_size + lane];
            let reset = 1.0_f32 / (1.0_f32 + (0.0_f32 - (input_gates[gate_start + lane * gate_feature_stride] + hr)).exp());
            let update = 1.0_f32 / (1.0_f32 + (0.0_f32 - (input_gates[gate_start + (hidden_size + lane) * gate_feature_stride] + hz)).exp());
            let candidate = (input_gates[gate_start + (2 * hidden_size + lane) * gate_feature_stride] + reset * hn).tanh();
            let next = (1.0_f32 - update) * candidate + update * hidden[lane];
            hidden[lane] = next;
            output[(batch * time + frame) * hidden_size * 2 + direction * hidden_size + lane] = next;
        }
        sync_cube();
    }
}

#[cube(launch_unchecked)]
fn bigru_split_kernel(
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
    #[comptime] split: usize,
) {
    let thread = UNIT_POS as usize;
    let lane = thread % hidden_size;
    let shard = thread / hidden_size;
    let batch = CUBE_POS_X as usize;
    let direction = CUBE_POS_Y as usize;
    let reverse = direction == 1;
    let mut hidden = Shared::<[f32]>::new_slice(256usize);
    let mut partial_r = Shared::<[f32]>::new_slice(1024usize);
    let mut partial_z = Shared::<[f32]>::new_slice(1024usize);
    let mut partial_n = Shared::<[f32]>::new_slice(1024usize);
    if shard == 0 {
        hidden[lane] = 0.0_f32;
    }
    sync_cube();
    let bias_start = direction * bias_direction_stride;
    let weight_start = direction * weight_direction_stride;
    let shard_width = hidden_size / split;

    for step in 0..time {
        let frame = if reverse { time - 1 - step } else { step };
        let gate_start = batch * gate_batch_stride
            + frame * gate_time_stride
            + direction * hidden_size * 3 * gate_feature_stride;
        let mut hr = if shard == 0 { hidden_bias[bias_start + lane * bias_feature_stride] } else { 0.0_f32 };
        let mut hz = if shard == 0 { hidden_bias[bias_start + (hidden_size + lane) * bias_feature_stride] } else { 0.0_f32 };
        let mut hn = if shard == 0 { hidden_bias[bias_start + (2 * hidden_size + lane) * bias_feature_stride] } else { 0.0_f32 };
        for inner in 0..shard_width {
            let column = shard * shard_width + inner;
            let value = hidden[column];
            let first = weight_start + column * weight_row_stride + lane * weight_column_stride;
            let gate_stride = hidden_size * weight_column_stride;
            hr += value * hidden_weight[first];
            hz += value * hidden_weight[first + gate_stride];
            hn += value * hidden_weight[first + 2 * gate_stride];
        }
        let slot = shard * hidden_size + lane;
        partial_r[slot] = hr;
        partial_z[slot] = hz;
        partial_n[slot] = hn;
        sync_cube();
        if shard == 0 {
            let mut total_r = partial_r[lane];
            let mut total_z = partial_z[lane];
            let mut total_n = partial_n[lane];
            for part in 1..split {
                let offset = part * hidden_size + lane;
                total_r += partial_r[offset];
                total_z += partial_z[offset];
                total_n += partial_n[offset];
            }
            let reset = 1.0_f32 / (1.0_f32 + (0.0_f32 - (input_gates[gate_start + lane * gate_feature_stride] + total_r)).exp());
            let update = 1.0_f32 / (1.0_f32 + (0.0_f32 - (input_gates[gate_start + (hidden_size + lane) * gate_feature_stride] + total_z)).exp());
            let candidate = (input_gates[gate_start + (2 * hidden_size + lane) * gate_feature_stride] + reset * total_n).tanh();
            let next = (1.0_f32 - update) * candidate + update * hidden[lane];
            hidden[lane] = next;
            output[(batch * time + frame) * hidden_size * 2 + direction * hidden_size + lane] = next;
        }
        sync_cube();
    }
}

pub(crate) fn pack_hidden_weight(weight: Tensor<3>) -> Result<Tensor<3>, String> {
    if weight.dims() != [2, 768, 256] {
        return Err("RMVPE GRU coalesced prepack requires [2,768,256] hidden weights".into());
    }
    let transposed = weight.swap_dims(1, 2)
        .try_into_primitive::<CubeBackend>()
        .map_err(|error| format!("RMVPE GRU coalesced prepack requires CubeCL: {error:?}"))?;
    let packed = into_contiguous(transposed);
    if !packed.is_contiguous() || packed.meta.shape().as_slice() != [2, 256, 768] {
        return Err("RMVPE GRU coalesced prepack did not materialize expected layout".into());
    }
    Ok(Tensor::from_primitive::<CubeBackend>(packed))
}

pub(crate) fn forward(
    gates: Tensor<3>,
    weight: Tensor<3>,
    bias: Tensor<2>,
) -> Result<Tensor<3>, String> {
    let [batch, time, gates_count] = gates.dims();
    let weight_dims = weight.dims();
    let weight_coalesced = weight_dims == [2, 256, 768];
    let hidden = 256;
    if batch == 0 || time == 0
        || (weight_dims != [2, 768, 256] && !weight_coalesced)
        || gates_count != hidden * 6 || bias.dims() != [2, hidden * 3]
    {
        return Err("persistent BiGRU requires [B,T,1536] gates and [2,768,256] or [2,256,768] weights".into());
    }
    if batch > u32::MAX as usize {
        return Err("persistent BiGRU batch exceeds CUDA grid limits".into());
    }
    let column_split: usize = std::env::var("RVC_RMVPE_GRU_COLUMN_SPLIT")
        .unwrap_or_else(|_| "1".to_owned())
        .parse().map_err(|_| "RMVPE GRU COLUMN_SPLIT must be 1, 2 or 4")?;
    if !matches!(column_split, 1 | 2 | 4) || (column_split > 1 && !weight_coalesced) {
        return Err("RMVPE GRU split-K requires RVC_RMVPE_GRU_WEIGHT_COALESCED=1 and COLUMN_SPLIT=2 or 4".into());
    }
    let gate_parallel = std::env::var("RVC_RMVPE_GRU_GATE_PARALLEL").as_deref() == Ok("1");
    if gate_parallel && (column_split != 1 || !weight_coalesced) {
        return Err("RMVPE GRU gate-parallel requires coalesced weights and COLUMN_SPLIT=1".into());
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
    if gate_parallel {
        unsafe {
            bigru_gate_parallel_kernel::launch_unchecked(
                &input.client,
                CubeCount::Static(batch as u32, 2, 1),
                CubeDim::new_1d((hidden * 3) as u32),
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
    } else if column_split > 1 {
        unsafe {
            bigru_split_kernel::launch_unchecked(
                &input.client,
                CubeCount::Static(batch as u32, 2, 1),
                CubeDim::new_1d((hidden * column_split) as u32),
                BufferArg::from_raw_parts(input.handle.clone(), input.handle.size_in_used() as usize / 4),
                BufferArg::from_raw_parts(weight.handle.clone(), weight.handle.size_in_used() as usize / 4),
                BufferArg::from_raw_parts(bias.handle.clone(), bias.handle.size_in_used() as usize / 4),
                BufferArg::from_raw_parts(output.clone(), bytes / 4),
                time,
                hidden,
                strides[0], strides[1], strides[2],
                weights_strides[0], weights_strides[1], weights_strides[2],
                bias_strides[0], bias_strides[1],
                column_split,
            );
        }
    } else {
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
            weight_coalesced,
            );
        }
    }
    Ok(Tensor::from_primitive::<CubeBackend>(CubeTensor::new_contiguous(
        input.client, input.device, [batch, time, hidden * 2].into(), output, DType::F32,
    )))
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn cpu_kernel_split2_split4_matches_original_gru_within_float_tolerance() {
        let device = Device::cpu();
        let weights = (0..2 * 768 * 256)
            .map(|i| ((i * 31 + 17) % 401) as f32 / 32000.0 - 0.006)
            .collect::<Vec<_>>();
        let input = (0..2 * 3 * 1536)
            .map(|i| ((i * 43 + 19) % 211) as f32 / 900.0 - 0.11)
            .collect::<Vec<_>>();
        let bias = (0..2 * 768)
            .map(|i| ((i * 73 + 29) % 97) as f32 / 10000.0 - 0.004)
            .collect::<Vec<_>>();
        let input = Tensor::from_data(TensorData::new(input, [2, 3, 1536]), &device);
        let bias = Tensor::from_data(TensorData::new(bias, [2, 768]), &device);
        let original = Tensor::from_data(TensorData::new(weights, [2, 768, 256]), &device);
        let packed = pack_hidden_weight(original.clone()).unwrap();

        unsafe { std::env::remove_var("RVC_RMVPE_GRU_COLUMN_SPLIT") };
        let base = forward(input.clone(), original, bias.clone())
            .unwrap().into_data().try_into_vec::<f32>().unwrap();
        let coalesced = forward(input.clone(), packed.clone(), bias.clone())
            .unwrap().into_data().try_into_vec::<f32>().unwrap();
        for (a, b) in base.iter().zip(&coalesced) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        unsafe { std::env::set_var("RVC_RMVPE_GRU_GATE_PARALLEL", "1") };
        let gate_parallel = forward(input.clone(), packed.clone(), bias.clone())
            .unwrap().into_data().try_into_vec::<f32>().unwrap();
        for (a, b) in base.iter().zip(&gate_parallel) {
            assert_eq!(a.to_bits(), b.to_bits(), "gate-parallel recurrence must be bit-exact");
        }
        unsafe { std::env::remove_var("RVC_RMVPE_GRU_GATE_PARALLEL") };
        for split in [2, 4] {
            unsafe { std::env::set_var("RVC_RMVPE_GRU_COLUMN_SPLIT", split.to_string()) };
            let candidate = forward(input.clone(), packed.clone(), bias.clone())
                .unwrap().into_data().try_into_vec::<f32>().unwrap();
            let max_abs = base.iter().zip(candidate).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
            assert!(max_abs < 1e-4, "CPU split={split} max_abs={max_abs}");
        }
        unsafe { std::env::remove_var("RVC_RMVPE_GRU_COLUMN_SPLIT") };
    }
}
