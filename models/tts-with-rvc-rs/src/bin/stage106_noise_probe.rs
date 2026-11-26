use burn::tensor::{Device, Tensor, TensorData, module, ops::ConvOptions};
use burn_cubecl::{
    Cube,
    kernel::conv::{
        conv1d_single_input_packed, conv1d_single_input_packed_interior,
        conv1d_single_input_prepack_weight,
    },
};
use serde_json::json;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    n: usize,
    cout: usize,
    t: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    pad: usize,
}

const CASES: [Case; 5] = [
    Case {
        name: "small",
        n: 1,
        cout: 5,
        t: 37,
        k: 8,
        stride: 4,
        dilation: 1,
        pad: 3,
    },
    Case {
        name: "noise2",
        n: 2,
        cout: 16,
        t: 8192,
        k: 4,
        stride: 2,
        dilation: 1,
        pad: 2,
    },
    Case {
        name: "noise8",
        n: 2,
        cout: 32,
        t: 32768,
        k: 16,
        stride: 8,
        dilation: 1,
        pad: 8,
    },
    Case {
        name: "noise40",
        n: 2,
        cout: 256,
        t: 25600,
        k: 80,
        stride: 40,
        dilation: 1,
        pad: 40,
    },
    Case {
        name: "noise100",
        n: 2,
        cout: 128,
        t: 32000,
        k: 200,
        stride: 100,
        dilation: 1,
        pad: 100,
    },
];

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let backend = args.next().unwrap_or_else(|| "cuda".into());
    if backend == "nvtx-status" {
        println!("nvtx_available={}", tts_with_rvc_rs::nvtx::available());
        return Ok(());
    }
    let case_name = args.next().unwrap_or_else(|| "noise8".into());
    let mode = args.next().unwrap_or_else(|| "tiled".into());
    let repetitions: usize = args
        .next()
        .unwrap_or_else(|| "8".into())
        .parse()
        .map_err(|_| "repetitions must be integer")?;
    if !(2..=40).contains(&repetitions) || !matches!(mode.as_str(), "direct" | "tiled" | "interior")
    {
        return Err(
            "usage: stage106_noise_probe [cuda|cpu] [case] [direct|tiled|interior] [repetitions=8]"
                .into(),
        );
    }
    let c = CASES
        .iter()
        .find(|c| c.name == case_name)
        .ok_or_else(|| format!("unknown case {case_name}"))?;
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
        _ => return Err("backend must be cuda or cpu".into()),
    };
    let x = Tensor::<3>::from_data(
        TensorData::new(
            (0..c.n * c.t)
                .map(|i| (i as f32 * 0.013).sin() * 0.19)
                .collect::<Vec<_>>(),
            [c.n, 1, c.t],
        ),
        &device,
    );
    let w = Tensor::<3>::from_data(
        TensorData::new(
            (0..c.cout * c.k)
                .map(|i| (i as f32 * 0.021).cos() * 0.12)
                .collect::<Vec<_>>(),
            [c.cout, 1, c.k],
        ),
        &device,
    );
    let b = Tensor::<1>::from_data(
        TensorData::new(
            (0..c.cout).map(|i| i as f32 * 0.001).collect::<Vec<_>>(),
            [c.cout],
        ),
        &device,
    );
    let opts = ConvOptions::new_with_padding([c.stride], [(c.pad, c.pad)], [c.dilation], 1);
    let packed_weight = Tensor::<3>::from_primitive::<Cube>(conv1d_single_input_prepack_weight(
        w.clone()
            .try_into_primitive::<Cube>()
            .expect("CubeCL weight"),
    ));
    device.sync().map_err(|e| e.to_string())?;
    let execute = || -> Tensor<3> {
        if mode == "direct" {
            module::conv1d(x.clone(), w.clone(), Some(b.clone()), opts.clone())
        } else {
            let raw_x = x
                .clone()
                .try_into_primitive::<Cube>()
                .expect("CubeCL input");
            let raw_w = packed_weight
                .clone()
                .try_into_primitive::<Cube>()
                .expect("CubeCL packed weight");
            let raw_b = b.clone().try_into_primitive::<Cube>().expect("CubeCL bias");
            let forward = if mode == "interior" {
                conv1d_single_input_packed_interior(raw_x, raw_w, Some(raw_b), opts.clone())
            } else {
                conv1d_single_input_packed(raw_x, raw_w, Some(raw_b), opts.clone())
            };
            Tensor::from_primitive::<Cube>(forward.expect("CubeCL tiled noise convolution"))
        }
    };
    let mut times = Vec::new();
    let mut last = None;
    for repeat in 0..repetitions {
        device.sync().map_err(|e| e.to_string())?;
        let label = format!("stage106_noise_{}_{}", c.name, mode);
        let _range = (repeat > 0).then(|| tts_with_rvc_rs::nvtx::Range::new(&label));
        let start = Instant::now();
        let output = execute();
        device.sync().map_err(|e| e.to_string())?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        drop(_range);
        if repeat > 0 {
            times.push(ms);
        }
        last = Some(output);
    }
    times.sort_by(f64::total_cmp);
    let actual: Vec<f32> = last
        .unwrap()
        .into_data()
        .try_into_vec::<f32>()
        .map_err(|e| e.to_string())?;
    let reference: Vec<f32> = module::conv1d(x, w, Some(b), opts)
        .into_data()
        .try_into_vec::<f32>()
        .map_err(|e| e.to_string())?;
    if actual.len() != reference.len() {
        return Err("shape mismatch".into());
    }
    let mut max_abs = 0f32;
    let mut mse = 0f64;
    let mut signal = 0f64;
    for (a, e) in actual.iter().zip(reference.iter()) {
        let d = (*a - *e).abs();
        max_abs = max_abs.max(d);
        mse += (d as f64) * (d as f64);
        signal += (*e as f64) * (*e as f64);
    }
    if !max_abs.is_finite() || max_abs > 1e-4 {
        return Err(format!("full-output parity failed: {max_abs}"));
    }
    println!(
        "{}",
        json!({
            "stage":106,"operator":"single_input_conv1d","backend":backend,
            "case":c.name,"mode":mode,"repetitions":repetitions,
            "shape":[c.n,1,c.t],"weight_shape":[c.cout,1,c.k],
            "stride":c.stride,"dilation":c.dilation,"padding":c.pad,
            "median_ms":times[times.len()/2],"output_elements":actual.len(),
            "max_abs":max_abs,"snr_db":if mse>0.0 {Some(10.0*(signal/mse).log10())}else{None::<f64>},
            "full_output_parity":true,
        })
    );
    Ok(())
}
