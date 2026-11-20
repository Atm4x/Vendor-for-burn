use burn::tensor::{DType, Device, Tensor, TensorData, module, ops::ConvTransposeOptions};
use burn_cubecl::{
    Cube,
    kernel::conv::{
        conv_transpose1d_phase_gemm_packed, conv_transpose1d_phase_gemm_packed_compensated,
        prepack_conv_transpose1d_phase_gemm_weight,
        prepack_conv_transpose1d_phase_gemm_weight_compensated,
    },
};
use serde_json::json;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    batch: usize,
    ci: usize,
    co: usize,
    t: usize,
    stride: usize,
    kernel: usize,
    padding: usize,
}

const CASES: &[Case] = &[
    Case {
        name: "small",
        batch: 1,
        ci: 3,
        co: 5,
        t: 25,
        stride: 2,
        kernel: 6,
        padding: 2,
    },
    Case {
        name: "upsample8",
        batch: 2,
        ci: 64,
        co: 32,
        t: 512,
        stride: 8,
        kernel: 16,
        padding: 4,
    },
    Case {
        name: "upsample10",
        batch: 2,
        ci: 128,
        co: 64,
        t: 256,
        stride: 10,
        kernel: 20,
        padding: 5,
    },
    Case {
        name: "upsample4",
        batch: 2,
        ci: 32,
        co: 16,
        t: 1024,
        stride: 4,
        kernel: 8,
        padding: 2,
    },
];

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let backend = args.next().unwrap_or_else(|| "cuda".into());
    if backend == "nvtx-status" {
        println!("nvtx_available={}", tts_with_rvc_rs::nvtx::available());
        return Ok(());
    }
    let case = args.next().unwrap_or_else(|| "upsample8".into());
    let mode = args.next().unwrap_or_else(|| "phase_f16".into());
    let repetitions: usize = args
        .next()
        .unwrap_or_else(|| "12".into())
        .parse()
        .map_err(|_| "invalid repetitions")?;
    if !(3..=40).contains(&repetitions)
        || !matches!(mode.as_str(), "direct" | "phase_f32" | "phase_f16")
    {
        return Err("usage: stage111_phase_gemm_probe [cuda|cpu] [small|upsample8|upsample10|upsample4] [direct|phase_f32|phase_f16] [repetitions]".into());
    }
    let c = CASES
        .iter()
        .find(|c| c.name == case)
        .ok_or("unsupported case")?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu(),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::cuda(0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err("CUDA feature not enabled".into());
            }
        }
        _ => return Err("backend must be cpu or cuda".into()),
    };
    let x = Tensor::<3>::from_data(
        TensorData::new(
            (0..c.batch * c.ci * c.t)
                .map(|i| (i as f32 * 0.031).sin() * 0.2)
                .collect::<Vec<_>>(),
            [c.batch, c.ci, c.t],
        ),
        &device,
    );
    let w = Tensor::<3>::from_data(
        TensorData::new(
            (0..c.ci * c.co * c.kernel)
                .map(|i| (i as f32 * 0.027).cos() * 0.1)
                .collect::<Vec<_>>(),
            [c.ci, c.co, c.kernel],
        ),
        &device,
    );
    let b = Tensor::<1>::from_data(
        TensorData::new(
            (0..c.co).map(|i| i as f32 * 0.001).collect::<Vec<_>>(),
            [c.co],
        ),
        &device,
    );
    let opts = ConvTransposeOptions::new([c.stride], [c.padding], [0], [1], 1);
    let half = mode == "phase_f16";
    let (packed, packed_low) = if mode == "phase_f16" {
        let raw = w.clone().cast(DType::F16)
            .try_into_primitive::<Cube>().expect("CubeCL half weight");
        (Some(Tensor::<4>::from_primitive::<Cube>(
            prepack_conv_transpose1d_phase_gemm_weight(raw, c.stride, c.padding, 1)
                .expect("phase matrix half prepack"),
        )), None)
    } else if mode == "phase_f32" {
        let raw = w.clone().try_into_primitive::<Cube>().expect("CubeCL FP32 weight");
        let (high, low) = prepack_conv_transpose1d_phase_gemm_weight_compensated(
            raw, c.stride, c.padding, 1,
        ).expect("phase matrix compensated FP32 prepack");
        (Some(Tensor::<4>::from_primitive::<Cube>(high)),
         Some(Tensor::<4>::from_primitive::<Cube>(low)))
    } else {
        (None, None)
    };
    device.sync().map_err(|e| e.to_string())?;
    let execute = || -> Tensor<3> {
        if let Some(weight) = &packed {
            let x = if half {
                x.clone().cast(DType::F16)
            } else {
                x.clone()
            };
            let b = if half {
                b.clone().cast(DType::F16)
            } else {
                b.clone()
            };
            let raw_x = x.try_into_primitive::<Cube>().expect("CubeCL input");
            let raw_weight = weight
                .clone()
                .try_into_primitive::<Cube>()
                .expect("CubeCL weight");
            let raw_bias = b.try_into_primitive::<Cube>().expect("CubeCL bias");
            let out = Tensor::<3>::from_primitive::<Cube>(
                if let Some(low) = &packed_low {
                    let raw_low = low.clone().try_into_primitive::<Cube>()
                        .expect("CubeCL FP32 low weight");
                    conv_transpose1d_phase_gemm_packed_compensated(
                        raw_x, raw_weight, raw_low, Some(raw_bias),
                        c.stride, c.padding, c.kernel, 1, 0, 1,
                    )
                } else {
                    conv_transpose1d_phase_gemm_packed(
                        raw_x, raw_weight, Some(raw_bias),
                        c.stride, c.padding, c.kernel, 1, 0, 1,
                    )
                }.expect("CubeCL phase matmul"),
            );
            if half { out.cast(DType::F32) } else { out }
        } else {
            module::conv_transpose1d(x.clone(), w.clone(), Some(b.clone()), opts.clone())
        }
    };
    let mut measurements = Vec::new();
    let mut output = None;
    for i in 0..repetitions {
        device.sync().map_err(|e| e.to_string())?;
        let label = format!("stage111_{case}_{mode}");
        let range = (i > 0).then(|| tts_with_rvc_rs::nvtx::Range::new(&label));
        let start = Instant::now();
        let result = execute();
        device.sync().map_err(|e| e.to_string())?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        drop(range);
        if i > 0 {
            measurements.push(elapsed);
        }
        output = Some(result);
    }
    measurements.sort_by(f64::total_cmp);
    let actual = output
        .unwrap()
        .cast(DType::F32)
        .into_data()
        .try_into_vec::<f32>()
        .map_err(|e| e.to_string())?;
    let expected = module::conv_transpose1d(x, w, Some(b), opts)
        .into_data()
        .try_into_vec::<f32>()
        .map_err(|e| e.to_string())?;
    if actual.len() != expected.len() {
        return Err("output shape mismatch".into());
    }
    let mut max_abs = 0f32;
    let mut mse = 0f64;
    let mut signal = 0f64;
    for (a, e) in actual.iter().zip(&expected) {
        let diff = (*a - *e).abs();
        max_abs = max_abs.max(diff);
        mse += (diff as f64).powi(2);
        signal += (*e as f64).powi(2);
    }
    let snr = if mse > 0.0 {
        Some(10.0 * (signal / mse).log10())
    } else {
        None
    };
    let limit = if half { 5e-3 } else { 2e-5 };
    if !max_abs.is_finite()
        || max_abs > limit
        || (half && snr.is_none_or(|v| v < 50.0))
        || (mode == "phase_f32" && snr.is_none_or(|v| v < 90.0))
    {
        return Err(format!(
            "full-output parity failure: {case} {mode}, max_abs={max_abs}, snr_db={snr:?}"
        ));
    }
    println!(
        "{}",
        json!({
            "stage":111,"backend":backend,"case":case,"mode":mode,"full_output_parity":true,
            "max_abs":max_abs,"snr_db":snr,"median_ms":measurements[measurements.len()/2],
            "weight_packed_once":true,"input_shape":[c.batch,c.ci,c.t],
            "weight_shape":[c.ci,c.co,c.kernel],"stride":c.stride,"padding":c.padding,
            "dtype":if half { "f16" } else { "f32" },"repetitions":repetitions,
        })
    );
    Ok(())
}
