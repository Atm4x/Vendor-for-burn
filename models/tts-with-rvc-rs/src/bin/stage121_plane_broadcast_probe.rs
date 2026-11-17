use burn::tensor::{Device, Tensor, TensorData};
use burn_cubecl::{
    Cube,
    kernel::conv::{conv1d_preact_residual_temporal, prepare_resblock_input_ntc},
};

fn evaluate(device: &Device, time: usize, with_broadcast: bool) -> Result<Vec<f32>, String> {
    let cin = 128;
    let cout = 128;
    let kernel = 11;
    let dilation = 5;
    let padding = 25;
    let input = Tensor::<3>::from_data(
        TensorData::new(
            (0..cin * time)
                .map(|i| ((i * 31 % 89) as f32 - 44.0) * 0.006)
                .collect::<Vec<_>>(),
            [1, cin, time],
        ),
        device,
    );
    let packed_weight = Tensor::<3>::from_data(
        TensorData::new(
            (0..kernel * cin * cout)
                .map(|i| ((i * 7 % 47) as f32 - 23.0) * 0.009)
                .collect::<Vec<_>>(),
            [kernel, cin, cout],
        ),
        device,
    )
    .permute([2, 1, 0]);
    let bias = Tensor::<1>::from_data(
        TensorData::new((0..cout).map(|i| (i as f32 - 63.0) * 0.001).collect::<Vec<_>>(), [cout]),
        device,
    );
    let residual = Tensor::<3>::from_data(
        TensorData::new(
            (0..cout * time)
                .map(|i| ((i * 11 % 61) as f32 - 30.0) * 0.004)
                .collect::<Vec<_>>(),
            [1, cout, time],
        ),
        device,
    );
    let raw_input = input.try_into_primitive::<Cube>().map_err(|e| format!("{e:?}"))?;
    let raw_weight = packed_weight.try_into_primitive::<Cube>().map_err(|e| format!("{e:?}"))?;
    let raw_bias = bias.try_into_primitive::<Cube>().map_err(|e| format!("{e:?}"))?;
    let raw_residual = residual.try_into_primitive::<Cube>().map_err(|e| format!("{e:?}"))?;
    let prepared = prepare_resblock_input_ntc(raw_input);
    if with_broadcast {
        unsafe { std::env::set_var("RVC_NSF_RESBLOCK_PLANE_BROADCAST", "1") };
    } else {
        unsafe { std::env::remove_var("RVC_NSF_RESBLOCK_PLANE_BROADCAST") };
    }
    unsafe { std::env::set_var("RVC_NSF_RESBLOCK_TEMPORAL_TILE", "4") };
    let result = conv1d_preact_residual_temporal(
        prepared, raw_weight, Some(raw_bias), Some(raw_residual), dilation, padding,
    )
    .map_err(|e| format!("resblock CubeCL setup: {e:?}"))?;
    let output = Tensor::<3>::from_primitive::<Cube>(result)
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| format!("convert floats: {e:?}"))?;
    device.sync().map_err(|e| format!("{e:?}"))?;
    Ok(output)
}

fn main() -> Result<(), String> {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "cpu".into());
    let device = match backend.as_str() {
        "cpu" => Device::cpu(),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::cuda(0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err("Stage121 probe compiled without CUDA feature".into());
            }
        }
        _ => return Err("Usage: stage121_plane_broadcast_probe [cpu|cuda]".into()),
    };
    for &time in &[3usize, 17, 71] {
        let base = evaluate(&device, time, false)?;
        let broadcast = evaluate(&device, time, true)?;
        if base.len() != broadcast.len() {
            return Err(format!("Output shape mismatch: {} != {}", base.len(), broadcast.len()));
        }
        for (i, (&a, &b)) in base.iter().zip(broadcast.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!("CubeCL bytewise mismatch: T={time}, sample={i}, baseline={a}, broadcast={b}"));
            }
        }
        println!("stage121 CubeCL {backend} PASS: time={time} samples={}", base.len());
    }
    Ok(())
}
