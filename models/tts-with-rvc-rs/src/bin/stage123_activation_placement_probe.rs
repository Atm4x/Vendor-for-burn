use burn::tensor::{Device, Tensor, TensorData};
use burn_cubecl::{
    Cube,
    kernel::conv::{
        conv1d_preact_residual_temporal_activation_plan,
        prepare_resblock_input_ntc,
    },
};

fn evaluate(device: &Device, time: usize, dilation: usize, move_activation: bool) -> Result<Vec<f32>, String> {
    let channels = 128;
    let kernel = 11;
    let pad_first = 5 * dilation;
    let x = Tensor::<3>::from_data(
        TensorData::new(
            (0..channels * time).map(|i| ((i * 31 % 89) as f32 - 44.0) * 0.006).collect::<Vec<_>>(),
            [1, channels, time],
        ),
        device,
    );
    let w1 = Tensor::<3>::from_data(
        TensorData::new(
            (0..kernel * channels * channels)
                .map(|i| ((i * 7 % 47) as f32 - 23.0) * 0.009)
                .collect::<Vec<_>>(),
            [kernel, channels, channels],
        ),
        device,
    ).permute([2, 1, 0]);
    let w2 = Tensor::<3>::from_data(
        TensorData::new(
            (0..kernel * channels * channels)
                .map(|i| ((i * 13 % 53) as f32 - 26.0) * 0.007)
                .collect::<Vec<_>>(),
            [kernel, channels, channels],
        ),
        device,
    ).permute([2, 1, 0]);
    let b1 = Tensor::<1>::from_data(
        TensorData::new((0..channels).map(|i| (i as f32 - 63.0) * 0.001).collect::<Vec<_>>(), [channels]),
        device,
    );
    let b2 = Tensor::<1>::from_data(
        TensorData::new((0..channels).map(|i| (i as f32 - 54.0) * 0.002).collect::<Vec<_>>(), [channels]),
        device,
    );
    let raw_x = x.try_into_primitive::<Cube>().map_err(|e| format!("input primitive: {e:?}"))?;
    let raw_w1 = w1.try_into_primitive::<Cube>().map_err(|e| format!("first weight: {e:?}"))?;
    let raw_w2 = w2.try_into_primitive::<Cube>().map_err(|e| format!("second weight: {e:?}"))?;
    let raw_b1 = b1.try_into_primitive::<Cube>().map_err(|e| format!("first bias: {e:?}"))?;
    let raw_b2 = b2.try_into_primitive::<Cube>().map_err(|e| format!("second bias: {e:?}"))?;
    let input_ntc = prepare_resblock_input_ntc(raw_x.clone());
    unsafe { std::env::set_var("RVC_NSF_RESBLOCK_TEMPORAL_TILE", "4") };
    unsafe { std::env::remove_var("RVC_NSF_RESBLOCK_PLANE_BROADCAST") };
    let first = conv1d_preact_residual_temporal_activation_plan(
        input_ntc, raw_w1, Some(raw_b1), None, dilation, pad_first, false, move_activation,
    ).map_err(|e| format!("first CubeCL convolution setup: {e:?}; T={time}, D={dilation}"))?;
    let second = conv1d_preact_residual_temporal_activation_plan(
        first, raw_w2, Some(raw_b2), Some(raw_x), 1, 5, move_activation, false,
    ).map_err(|e| format!("second CubeCL convolution setup: {e:?}; T={time}, D={dilation}"))?;
    let output = Tensor::<3>::from_primitive::<Cube>(second)
        .into_data().try_to_vec::<f32>()
        .map_err(|e| format!("result convert: {e:?}"))?;
    device.sync().map_err(|e| format!("synchronization: {e:?}"))?;
    Ok(output)
}

fn main() -> Result<(), String> {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "cpu".to_owned());
    let device = match backend.as_str() {
        "cpu" => Device::cpu(),
        "cuda" => {
            #[cfg(feature = "cuda")]
            { Device::cuda(0) }
            #[cfg(not(feature = "cuda"))]
            { return Err("Stage123 probe was built without CUDA feature".to_owned()); }
        }
        _ => return Err("Usage: stage123_activation_placement_probe.exe [cpu|cuda]".to_owned()),
    };
    for dilation in [1usize, 3, 5] {
        for time in [3usize, 17, 71] {
            let control = evaluate(&device, time, dilation, false)?;
            let placement = evaluate(&device, time, dilation, true)?;
            if control.len() != placement.len() {
                return Err(format!("Shape mismatch: T={time}, D={dilation}, control={}, placement={}", control.len(), placement.len()));
            }
            for (i, (&a, &b)) in control.iter().zip(placement.iter()).enumerate() {
                if a.to_bits() != b.to_bits() {
                    return Err(format!("Stage123 REJECT: backend={backend} T={time} D={dilation} sample={i} control={a:?} placement={b:?}"));
                }
            }
            println!("Stage123 {backend} PASS: C=128 K=11 T={time} D={dilation} samples={}", control.len());
        }
    }
    Ok(())
}
