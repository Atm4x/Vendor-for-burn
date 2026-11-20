use burn::tensor::{TensorData, ops::ConvOptions};
use burn_flex::{
    FlexTensor,
    ops::conv::{conv1d_f32, conv3d_f32},
};
use std::{hint::black_box, time::Instant};

fn median(mut items: Vec<f64>) -> f64 {
    items.sort_by(|a, b| a.total_cmp(b));
    items[items.len() / 2]
}

fn run(
    name: &str,
    batch: usize,
    ci: usize,
    co: usize,
    n: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
) {
    let input: Vec<f32> = (0..batch * ci * n)
        .map(|j| (j as f32 * 0.173).sin() * 0.1)
        .collect();
    let weights: Vec<f32> = (0..co * ci / groups * k)
        .map(|j| (j as f32 * 0.113).cos() * 0.01)
        .collect();
    let x1 = FlexTensor::from_data(TensorData::new(input.clone(), vec![batch, ci, n]));
    let w1 = FlexTensor::from_data(TensorData::new(weights.clone(), vec![co, ci / groups, k]));
    let x3 = FlexTensor::from_data(TensorData::new(input, vec![batch, ci, 1, 1, n]));
    let w3 = FlexTensor::from_data(TensorData::new(weights, vec![co, ci / groups, 1, 1, k]));
    let o1 = ConvOptions::new([stride], [pad], [dilation], groups);
    let o3 = ConvOptions::new([1, 1, stride], [0, 0, pad], [1, 1, dilation], groups);
    let first: Vec<f32> = conv1d_f32(x1.clone(), w1.clone(), None, &o1)
        .into_data()
        .try_into_vec()
        .unwrap();
    let second: Vec<f32> = conv3d_f32(x3.clone(), w3.clone(), None, &o3)
        .into_data()
        .try_into_vec()
        .unwrap();
    assert_eq!(first.len(), second.len());
    let max_abs = first
        .iter()
        .zip(second.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    for (a, b) in first.iter().zip(second.iter()) {
        assert!(
            (a - b).abs() <= 0.0005,
            "{name}: max_abs={max_abs} mismatch {a} {b}"
        );
    }
    let mut native = Vec::new();
    let mut legacy = Vec::new();
    for i in 0..8 {
        let t = Instant::now();
        let data: Vec<f32> = black_box(conv1d_f32(x1.clone(), w1.clone(), None, &o1))
            .into_data()
            .try_into_vec()
            .unwrap();
        black_box(data);
        if i >= 2 {
            native.push(t.elapsed().as_secs_f64() * 1000.);
        }
        let t = Instant::now();
        let data: Vec<f32> = black_box(conv3d_f32(x3.clone(), w3.clone(), None, &o3))
            .into_data()
            .try_into_vec()
            .unwrap();
        black_box(data);
        if i >= 2 {
            legacy.push(t.elapsed().as_secs_f64() * 1000.);
        }
    }
    let native_ms = median(native);
    let legacy_ms = median(legacy);
    println!(
        "{name}: native_ms={native_ms:.4} legacy_ms={legacy_ms:.4} legacy/new={:.3}x max_abs={max_abs:.7}",
        legacy_ms / native_ms
    );
}

fn main() {
    run("tcn_dilated", 8, 64, 64, 160, 5, 1, 4, 2, 1);
    run("audio_padded", 8, 64, 64, 500, 5, 1, 2, 1, 1);
    run("rvc_short", 1, 256, 256, 24, 3, 1, 1, 1, 1);
    run("rvc_long", 1, 256, 256, 128, 3, 1, 1, 1, 1);
    run("wav2vec_stride", 1, 64, 32, 100, 3, 2, 0, 1, 1);
    run("depthwise", 2, 64, 64, 128, 7, 1, 6, 2, 64);
}
