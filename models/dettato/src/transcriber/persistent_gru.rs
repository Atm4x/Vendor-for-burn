use burn::cubecl::{self, prelude::*};
use burn::tensor::{DType, Tensor};
use burn_cubecl::{Cube, CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use burn_fusion::{
    ExecutionError,
    custom::{CustomOpIr, OperationFn, OperationIr, OperationOutput, StreamId, TensorIr},
};

#[cube(launch_unchecked)]
fn lane_gru_kernel(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    time: usize,
    hidden_size: usize,
    #[comptime] reverse: bool,
) {
    let lane = UNIT_POS as usize;
    let batch = CUBE_POS_X as usize;
    let gate_width = hidden_size * 3;
    let mut state = Shared::<[f32]>::new_slice(256usize);
    state[lane] = 0.0_f32;
    sync_cube();

    for step in 0..time {
        let frame = if reverse { time - 1 - step } else { step };
        let frame_index = batch * time + frame;
        let gate_start = frame_index * gate_width + lane;
        let mut reset = 0.0_f32;
        let mut update = 0.0_f32;
        let mut candidate = 0.0_f32;

        for column in 0..hidden_size {
            let val = state[column];
            let weight = column * gate_width + lane;
            reset += val * weights[weight];
            update += val * weights[weight + hidden_size];
            candidate += val * weights[weight + 2 * hidden_size];
        }
        reset += bias[lane];
        update += bias[lane + hidden_size];
        candidate += bias[lane + 2 * hidden_size];
        let r = 1.0_f32 / (1.0_f32 + (0.0_f32 - (input[gate_start] + reset)).exp());
        let z = 1.0_f32 / (1.0_f32 + (0.0_f32 - (input[gate_start + hidden_size] + update)).exp());
        let n = (input[gate_start + 2 * hidden_size] + r * candidate).tanh();
        let next = (1.0_f32 - z) * n + z * state[lane];
        sync_cube();
        state[lane] = next;
        output[frame_index * hidden_size + lane] = next;
        sync_cube();
    }
}

pub(super) fn forward(
    gates: Tensor<3>,
    weights: Tensor<2>,
    bias: Tensor<1>,
    reverse: bool,
) -> Option<Tensor<3>> {
    let [batch, time, gate_width] = gates.dims();
    let hidden = weights.dims()[0];
    if batch == 0
        || time == 0
        || batch > 65_535
        || !(1..=256).contains(&hidden)
        || gate_width != hidden * 3
        || weights.dims()[1] != gate_width
        || bias.dims() != [gate_width]
        || [gates.dtype(), weights.dtype(), bias.dtype()] != [DType::F32; 3]
    {
        return None;
    }

    let gates = gates.try_into_primitive::<Cube>().ok()?;
    let weights = weights.try_into_primitive::<Cube>().ok()?;
    let bias = bias.try_into_primitive::<Cube>().ok()?;
    let client = gates.client.clone();
    if !matches!(
        client.device(),
        cubecl::Device::Cuda(_) | cubecl::Device::Wgpu(_)
    ) {
        return None;
    }

    let output = TensorIr::uninit(
        client.create_empty_handle(),
        [batch, time, hidden].into(),
        DType::F32,
    );
    let desc = CustomOpIr::new(
        if reverse {
            "dettato_gru_reverse"
        } else {
            "dettato_gru_forward"
        },
        &[gates.into_ir(), weights.into_ir(), bias.into_ir()],
        &[output],
    );
    let exec = desc.clone();
    let op = OperationFn(
        move |handles: &mut burn_fusion::custom::HandleContainer<_>| {
            let ([gates, weights, bias], [output]) = exec.as_fixed::<3, 1>();
            let input = into_contiguous(handles.get_float_tensor::<CubeBackend>(gates));
            let weights = into_contiguous(handles.get_float_tensor::<CubeBackend>(weights));
            let bias = into_contiguous(handles.get_float_tensor::<CubeBackend>(bias));
            let bytes = batch
                .checked_mul(time)
                .and_then(|n| n.checked_mul(hidden * 4))
                .ok_or_else(|| {
                    ExecutionError::generic("Dettato GRU output size overflow".to_owned())
                })?;
            let result = input.client.empty(bytes);
            // The validated shapes and contiguous packing make every read and
            // write in the launch fit the corresponding buffer.
            unsafe {
                lane_gru_kernel::launch_unchecked(
                    &input.client,
                    CubeCount::Static(batch as u32, 1, 1),
                    CubeDim::new_1d(hidden as u32),
                    BufferArg::from_raw_parts(input.handle.clone(), batch * time * gate_width),
                    BufferArg::from_raw_parts(weights.handle.clone(), hidden * gate_width),
                    BufferArg::from_raw_parts(bias.handle.clone(), gate_width),
                    BufferArg::from_raw_parts(result.clone(), bytes / 4),
                    time,
                    hidden,
                    reverse,
                );
            }
            let result = CubeTensor::new_contiguous(
                input.client,
                input.device,
                output.shape.clone(),
                result,
                DType::F32,
            );
            handles.register_float_tensor::<CubeBackend>(&output.id, result);
            Ok(())
        },
    );
    let tensor = client
        .register(StreamId::current(), OperationIr::Custom(desc), op)
        .output();
    Some(Tensor::from_primitive::<Cube>(tensor))
}
