use burn::tensor::{Device, Tensor, TensorData, module, ops::ConvTransposeOptions};
use burn_cubecl::{
    Cube,
    kernel::conv::{
        conv_transpose1d_polyphase_packed, conv_transpose1d_polyphase_single_packed,
        conv_transpose1d_prepack_weight, conv_transpose1d_prepack_weight_kic,
        conv_transpose1d_polyphase_single_packed_kic,
    },
};
use serde_json::json;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    n: usize,
    ci: usize,
    co: usize,
    t: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

const CASES: [Case; 6] = [
    Case {
        name: "small",
        n: 1,
        ci: 3,
        co: 5,
        t: 25,
        k: 6,
        stride: 2,
        pad: 2,
    },
    Case {
        name: "upsample8",
        n: 2,
        ci: 64,
        co: 32,
        t: 512,
        k: 16,
        stride: 8,
        pad: 4,
    },
    Case {
        name: "upsample10",
        n: 2,
        ci: 128,
        co: 64,
        t: 256,
        k: 20,
        stride: 10,
        pad: 5,
    },
    Case {
        name: "upsample4",
        n: 2,
        ci: 32,
        co: 16,
        t: 1024,
        k: 8,
        stride: 4,
        pad: 2,
    },
    Case {
        name: "upsample2",
        n: 2,
        ci: 16,
        co: 8,
        t: 4096,
        k: 4,
        stride: 2,
        pad: 1,
    },
    Case {
        name: "kernel3",
        n: 1,
        ci: 16,
        co: 16,
        t: 128,
        k: 3,
        stride: 1,
        pad: 1,
    },
];

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let backend = args.next().unwrap_or_else(|| "cuda".into());
    if backend == "nvtx-status" {
        println!("nvtx_available={}", tts_with_rvc_rs::nvtx::available());
        return Ok(());
    }
    let case_name = args.next().unwrap_or_else(|| "upsample8".into());
    let mode = args.next().unwrap_or_else(|| "polyphase".into());
    let repetitions: usize = args
        .next()
        .unwrap_or_else(|| "8".into())
        .parse()
        .map_err(|_| "repetitions must be integer")?;
    if !(2..=40).contains(&repetitions)
        || !matches!(mode.as_str(), "direct" | "polyphase" | "single" | "kic" | "kic_ntc")
    {
        return Err(
            "usage: stage106_polyphase_probe [cuda|cpu] [case] [direct|polyphase|single|kic|kic_ntc] [repetitions=8]"
                .into(),
        );
    }
    let c = CASES.iter().find(|c| c.name == case_name).ok_or_else(|| {
        format!(
            "unknown case {case_name}: use {}",
            CASES.iter().map(|c| c.name).collect::<Vec<_>>().join(",")
        )
    })?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu(),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::cuda(0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err("binary built without CUDA feature".into());
            }
        }
        _ => return Err("backend must be cpu or cuda".into()),
    };
    let x = Tensor::<3>::from_data(
        TensorData::new(
            (0..c.n * c.ci * c.t)
                .map(|i| (i as f32 * 0.031).sin() * 0.2)
                .collect::<Vec<_>>(),
            [c.n, c.ci, c.t],
        ),
        &device,
    );
    let w = Tensor::<3>::from_data(
        TensorData::new(
            (0..c.ci * c.co * c.k)
                .map(|i| (i as f32 * 0.027).cos() * 0.1)
                .collect::<Vec<_>>(),
            [c.ci, c.co, c.k],
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
    let opts = ConvTransposeOptions::new([c.stride], [c.pad], [0], [1], 1);
    let packed_weight = conv_transpose1d_prepack_weight(
        w.clone()
            .try_into_primitive::<Cube>()
            .expect("CubeCL weight"),
    );
    let packed_weight = Tensor::<3>::from_primitive::<Cube>(packed_weight);
    let packed_kic = Tensor::<3>::from_primitive::<Cube>(conv_transpose1d_prepack_weight_kic(
        w.clone().try_into_primitive::<Cube>().expect("CubeCL weight"),
    ));
    device.sync().map_err(|e| e.to_string())?;

    let execute = || -> Tensor<3> {
        if mode == "direct" {
            module::conv_transpose1d(x.clone(), w.clone(), Some(b.clone()), opts.clone())
        } else {
            let x = x
                .clone()
                .try_into_primitive::<Cube>()
                .expect("CubeCL input");
            let pw = if matches!(mode.as_str(), "kic" | "kic_ntc") {
                packed_kic.clone().try_into_primitive::<Cube>()
            } else {
                packed_weight.clone().try_into_primitive::<Cube>()
            }.expect("CubeCL packed weight");
            let bias = b.clone().try_into_primitive::<Cube>().expect("CubeCL bias");
            let forward = if matches!(mode.as_str(), "kic" | "kic_ntc") {
                conv_transpose1d_polyphase_single_packed_kic(
                    x, pw, Some(bias), c.stride, c.pad, 1, 0, 1, mode == "kic_ntc",
                )
            } else if mode == "single" {
                conv_transpose1d_polyphase_single_packed(
                    x,
                    pw,
                    Some(bias),
                    c.stride,
                    c.pad,
                    1,
                    0,
                    1,
                )
            } else {
                conv_transpose1d_polyphase_packed(x, pw, Some(bias), c.stride, c.pad, 1, 0, 1)
            };
            Tensor::from_primitive::<Cube>(forward.expect("CubeCL polyphase kernel"))
        }
    };
    let mut times = Vec::new();
    let mut result = None;
    for rep in 0..repetitions {
        device.sync().map_err(|e| e.to_string())?;
        let label = format!("stage106_{}_{}", c.name, mode);
        let _range = (rep > 0).then(|| tts_with_rvc_rs::nvtx::Range::new(&label));
        let start = Instant::now();
        let output = execute();
        device.sync().map_err(|e| e.to_string())?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        drop(_range);
        if rep > 0 {
            times.push(elapsed);
        }
        result = Some(output);
    }
    times.sort_by(f64::total_cmp);
    let actual: Vec<f32> = result
        .unwrap()
        .into_data()
        .try_into_vec::<f32>()
        .map_err(|e| e.to_string())?;
    let reference: Vec<f32> = module::conv_transpose1d(x, w, Some(b), opts)
        .into_data()
        .try_into_vec::<f32>()
        .map_err(|e| e.to_string())?;
    let mut max_abs = 0.0f32;
    let mut mse = 0.0f64;
    let mut signal = 0.0f64;
    for (a, expected) in actual.iter().zip(&reference) {
        let delta = (*a - expected).abs();
        max_abs = max_abs.max(delta);
        mse += (delta as f64).powi(2);
        signal += (*expected as f64).powi(2);
    }
    let snr_db = if mse > 0.0 {
        Some(10.0 * (signal / mse).log10())
    } else {
        None
    };
    if !max_abs.is_finite() || max_abs > 1e-4 || actual.len() != reference.len() {
        return Err(format!(
            "full-tensor parity failure in {case_name}/{mode}: max_abs={max_abs}"
        ));
    }
    println!(
        "{}",
        json!({
            "stage":106,"backend":backend,"case":c.name,"mode":mode,"repetitions":repetitions,
            "input_shape":[c.n,c.ci,c.t],"weight_shape":[c.ci,c.co,c.k],
            "stride":c.stride,"padding":c.pad,"output_elements":actual.len(),
            "median_ms":times[times.len()/2],"max_abs":max_abs,
            "snr_db":snr_db,"full_output_parity":true,
        })
    );
    Ok(())
}
