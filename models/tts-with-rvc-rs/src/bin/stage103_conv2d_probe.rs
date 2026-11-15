use burn::tensor::{module, ops::ConvOptions, Device, Tensor, TensorData};
use burn_cubecl::{Cube, kernel::conv::{conv_forward_spatial_reuse, conv_forward_spatial_pair}};
use serde_json::json;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str, batch: usize, cin: usize, cout: usize,
    height: usize, width: usize, kernel: usize, padding: usize,
    stride: usize, dilation: usize, groups: usize,
}

const CASES: [Case; 4] = [
    Case { name: "dense3x3", batch: 2, cin: 32, cout: 32, height: 64, width: 64, kernel: 3, padding: 1, stride: 1, dilation: 1, groups: 1 },
    Case { name: "dense3x3_wide", batch: 1, cin: 64, cout: 64, height: 48, width: 48, kernel: 3, padding: 1, stride: 1, dilation: 1, groups: 1 },
    Case { name: "pointwise1x1", batch: 2, cin: 16, cout: 32, height: 80, width: 64, kernel: 1, padding: 0, stride: 1, dilation: 1, groups: 1 },
    Case { name: "depthwise3x3", batch: 1, cin: 64, cout: 64, height: 40, width: 48, kernel: 3, padding: 1, stride: 1, dilation: 1, groups: 64 },
];

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let backend = args.next().unwrap_or_else(|| "cuda".to_owned());
    if backend == "nvtx-status" {
        println!("nvtx_available={}", tts_with_rvc_rs::nvtx::available());
        return Ok(());
    }
    let repeats: usize = args.next().unwrap_or_else(|| "8".to_owned()).parse().map_err(|_| "repeats must be an integer")?;
    if !(2..=64).contains(&repeats) { return Err("repeats must be 2..64".into()); }
    let selected_case = args.next().unwrap_or_else(|| "all".to_owned());
    if selected_case != "all" && !CASES.iter().any(|c| c.name == selected_case) {
        return Err(format!("unknown case: {selected_case}"));
    }
    if args.next().is_some() { return Err("expected [backend] [repeats] [case|all]".into()); }
    let device = match backend.as_str() {
        "cuda" => {
            #[cfg(feature = "cuda")]
            { Device::cuda(0) }
            #[cfg(not(feature = "cuda"))]
            { return Err("CUDA feature was not enabled at build time".into()); }
        },
        "cpu" => Device::cpu(),
        _ => return Err("backend must be cpu or cuda".into()),
    };
    let strategy = std::env::var("RVC_STAGE103_STRATEGY")
        .unwrap_or_else(|_| "direct".to_string());
    if strategy != "direct" && strategy != "reuse" && strategy != "pair" {
        return Err("RVC_STAGE103_STRATEGY must be direct, reuse, or pair".into());
    }
    let mut cases = Vec::new();
    for c in CASES {
        if selected_case != "all" && c.name != selected_case { continue; }
        let n_input = c.batch * c.cin * c.height * c.width;
        let n_weight = c.cout * (c.cin / c.groups) * c.kernel * c.kernel;
        let x: Vec<f32> = (0..n_input).map(|i| ((i * 37 % 149) as f32 - 70.0) * 0.005).collect();
        let w: Vec<f32> = (0..n_weight).map(|i| ((i * 53 % 211) as f32 - 90.0) * 0.001).collect();
        let input: Tensor<4> = Tensor::from_data(TensorData::new(x, [c.batch,c.cin,c.height,c.width]), &device);
        let weight: Tensor<4> = Tensor::from_data(TensorData::new(w, [c.cout,c.cin/c.groups,c.kernel,c.kernel]), &device);
        let opts = ConvOptions::new([c.stride,c.stride], [c.padding,c.padding], [c.dilation,c.dilation], c.groups);
        let mut timings = Vec::new();
        let mut last = None;
        for iteration in 0..repeats {
            device.sync().map_err(|e| e.to_string())?;
            let nvtx_label = if selected_case == "all" {
                "stage102_conv2d_matched".to_owned()
            } else {
                format!("stage104_conv2d_{}", c.name)
            };
            let _nvtx = (iteration > 0).then(|| tts_with_rvc_rs::nvtx::Range::new(&nvtx_label));
            let started = Instant::now();
            let output = if strategy == "pair" && c.groups == 1 && c.kernel == 3 && c.padding == 1 {
                let input = input.clone().try_into_primitive::<Cube>().expect("CubeCL input");
                let weight = weight.clone().try_into_primitive::<Cube>().expect("CubeCL weight");
                Tensor::from_primitive::<Cube>(
                    conv_forward_spatial_pair(input, weight, None, opts.clone())
                        .expect("Stage105 spatial pair kernel"))
            } else if strategy == "reuse" && c.groups == 1 && c.kernel == 3 && c.padding == 1 {
                let input = input.clone().try_into_primitive::<Cube>()
                    .expect("CubeCL input");
                let weight = weight.clone().try_into_primitive::<Cube>()
                    .expect("CubeCL weight");
                Tensor::from_primitive::<Cube>(
                    conv_forward_spatial_reuse(input, weight, None, opts.clone())
                        .expect("Stage103 spatial reuse kernel"))
            } else {
                module::conv2d(input.clone(),weight.clone(),None,opts.clone())
            };
            device.sync().map_err(|e| e.to_string())?;
            let elapsed = started.elapsed().as_secs_f64()*1000.0;
            drop(_nvtx);
            if iteration > 0 { timings.push(elapsed); }
            last = Some(output);
        }
        timings.sort_by(f64::total_cmp);
        let median = timings[timings.len()/2];
        let values: Vec<f32> = last.unwrap().into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let samples = [0, values.len()/3, values.len()/2, values.len()-1].map(|i| values[i]);
        let full_parity = if (strategy == "reuse" || strategy == "pair") && c.groups == 1 && c.kernel == 3 {
            let reference = module::conv2d(input.clone(),weight.clone(),None,opts.clone());
            let expected: Vec<f32> = reference.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
            let mut max_abs = 0.0f32;
            let mut mse = 0.0f64;
            let mut signal_energy = 0.0f64;
            for (&actual,&reference) in values.iter().zip(&expected) {
                let diff = (actual-reference).abs();
                max_abs = max_abs.max(diff);
                mse += (diff as f64).powi(2);
                signal_energy += (reference as f64).powi(2);
            }
            mse /= values.len() as f64;
            signal_energy /= values.len() as f64;
            let snr = if mse > 0.0 { 10.0 * (signal_energy / mse).log10() } else { f64::INFINITY };
            if max_abs > 1e-4 || !max_abs.is_finite() {
                return Err(format!("Stage103 reuse full parity failed on {}: max_abs={max_abs}", c.name));
            }
            Some(json!({"elements":values.len(),"max_abs":max_abs,"rmse":mse.sqrt(),"snr_db":if snr.is_finite(){Some(snr)}else{None::<f64>},"passed":true}))
        } else { None };

        cases.push(json!({
            "name":c.name,"shape":[c.batch,c.cin,c.height,c.width],
            "weight_shape":[c.cout,c.cin/c.groups,c.kernel,c.kernel],
            "padding":c.padding,"stride":c.stride,"dilation":c.dilation,"groups":c.groups,
            "output_elements":values.len(),"sample_values":samples,
            "mean":values.iter().map(|&v|v as f64).sum::<f64>()/values.len() as f64,
            "median_ms":median,"repeats":repeats,"dtype":"fp32", "layout":"nchw",
            "full_parity":full_parity
        }));
    }
    println!("{}", serde_json::to_string_pretty(&json!({"stage":103,"strategy":strategy,"engine":"burn","backend":backend,"cases":cases})).map_err(|e|e.to_string())?);
    Ok(())
}
