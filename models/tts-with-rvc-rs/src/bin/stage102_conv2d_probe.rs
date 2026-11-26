use burn::tensor::{module, ops::ConvOptions, Device, Tensor, TensorData};
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
    if !(2..=64).contains(&repeats) || args.next().is_some() { return Err("repeats must be 2..64".into()); }
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
    let mut cases = Vec::new();
    for c in CASES {
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
            let _nvtx = (iteration > 0).then(|| tts_with_rvc_rs::nvtx::Range::new("stage102_conv2d_matched"));
            let started = Instant::now();
            let output = module::conv2d(input.clone(),weight.clone(),None,opts.clone());
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
        cases.push(json!({
            "name":c.name,"shape":[c.batch,c.cin,c.height,c.width],
            "weight_shape":[c.cout,c.cin/c.groups,c.kernel,c.kernel],
            "padding":c.padding,"stride":c.stride,"dilation":c.dilation,"groups":c.groups,
            "output_elements":values.len(),"sample_values":samples,
            "mean":values.iter().map(|&v|v as f64).sum::<f64>()/values.len() as f64,
            "median_ms":median,"repeats":repeats,"dtype":"fp32", "layout":"nchw"
        }));
    }
    println!("{}", serde_json::to_string_pretty(&json!({"stage":102,"engine":"burn","backend":backend,"cases":cases})).map_err(|e|e.to_string())?);
    Ok(())
}
