use burn::tensor::{Device, Tensor, TensorData, module, ops::ConvOptions};
use std::time::Instant;

fn median(mut values: Vec<f64>) -> (f64, f64, f64) {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    (values[n / 2], values[n / 10], values[n * 9 / 10])
}
fn main() -> Result<(), String> {
    let d = Device::flex();
    for (cin, cout, size, batch, groups) in [
        (16, 32, 128, 1, 1),
        (32, 64, 64, 1, 1),
        (64, 128, 32, 1, 1),
        (128, 128, 16, 1, 1),
        (256, 256, 8, 1, 1),
        (512, 512, 4, 1, 1),
        (8, 12, 11, 2, 2),
        (16, 16, 16, 1, 16),
    ] {
        let x = (0..batch * cin * size * size)
            .map(|i| ((i * 17 % 911) as f32) * 0.0001 - 0.04)
            .collect::<Vec<_>>();
        let w = (0..cout * (cin / groups) * 9)
            .map(|i| ((i * 13 % 567) as f32) * 0.0002 - 0.05)
            .collect::<Vec<_>>();
        let x4: Tensor<4> =
            Tensor::from_data(TensorData::new(x.clone(), [batch, cin, size, size]), &d);
        let w4: Tensor<4> =
            Tensor::from_data(TensorData::new(w.clone(), [cout, cin / groups, 3, 3]), &d);
        let x5: Tensor<5> = Tensor::from_data(TensorData::new(x, [batch, cin, 1, size, size]), &d);
        let w5: Tensor<5> =
            Tensor::from_data(TensorData::new(w, [cout, cin / groups, 1, 3, 3]), &d);
        let new = || -> Result<(f64, Vec<f32>), String> {
            let now = Instant::now();
            let out = module::conv2d(
                x4.clone(),
                w4.clone(),
                None,
                ConvOptions::new([1, 1], [1, 1], [1, 1], groups),
            )
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|e| e.to_string())?;
            Ok((now.elapsed().as_secs_f64() * 1000.0, out))
        };
        let old = || -> Result<(f64, Vec<f32>), String> {
            let now = Instant::now();
            let out = module::conv3d(
                x5.clone(),
                w5.clone(),
                None,
                ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], groups),
            )
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|e| e.to_string())?;
            Ok((now.elapsed().as_secs_f64() * 1000.0, out))
        };
        let mut old_times = Vec::new();
        let mut new_times = Vec::new();
        let mut max_abs = 0f32;
        for i in 0..26 {
            let (time_a, out_a);
            let (time_b, out_b);
            if i % 2 == 0 {
                (time_a, out_a) = old()?;
                (time_b, out_b) = new()?;
            } else {
                (time_b, out_b) = new()?;
                (time_a, out_a) = old()?;
            }
            if out_a.len() != out_b.len() {
                return Err("length mismatch".into());
            }
            for (&a, &b) in out_a.iter().zip(out_b.iter()) {
                max_abs = max_abs.max((a - b).abs());
            }
            if i >= 6 {
                old_times.push(time_a);
                new_times.push(time_b)
            }
        }
        let a = median(old_times);
        let b = median(new_times);
        println!(
            "ci={cin} co={cout} size={size} batch={batch} groups={groups} conv3d_old_ms={:.5} new_universal_ms={:.5} speedup={:.3} parity_max={max_abs:.8} old_p10={:.4} old_p90={:.4} new_p10={:.4} new_p90={:.4}",
            a.0,
            b.0,
            a.0 / b.0,
            a.1,
            a.2,
            b.1,
            b.2
        );
    }
    Ok(())
}
