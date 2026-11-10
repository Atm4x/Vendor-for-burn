#[cfg(feature = "cpu")]
use burn::tensor::{Device, Tensor, TensorData};
#[cfg(feature = "cpu")]
use std::{hint::black_box, time::Instant};
#[cfg(feature = "cpu")]
use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, decoder::NsfDecoderFixture};

#[cfg(feature = "cpu")]
fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

#[cfg(feature = "cpu")]
fn main() {
    let fixture: NsfDecoderFixture =
        serde_json::from_str(include_str!("../../tests/fixtures/nsf_decoder.json")).unwrap();
    let checkpoint = RvcCheckpoint::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/synthetic_nsf_decoder.pth"
    ))
    .unwrap();
    let device = Device::cpu();
    let decoder = fixture.load(&checkpoint, &device).unwrap();
    let iterations = std::env::args()
        .nth(1)
        .map(|x| x.parse().unwrap())
        .unwrap_or(12);
    let warmup = 4;
    for frames in [4usize, 64, 256] {
        let [batch, channels, original] = fixture.input_shape;
        let upsample = fixture.source_shape[2] / original;
        assert_eq!(batch, 1);
        let mut values = Vec::with_capacity(channels * frames);
        for c in 0..channels {
            for t in 0..frames {
                values.push(fixture.input[c * original + t % original]);
            }
        }
        let harmonic: Vec<f32> = (0..frames * upsample)
            .map(|t| fixture.harmonic_source[t % fixture.source_shape[2]])
            .collect();
        let input = Tensor::from_data(TensorData::new(values, [1, channels, frames]), &device);
        let source = Tensor::from_data(
            TensorData::new(harmonic, [1, 1, frames * upsample]),
            &device,
        );
        let conditioning = fixture.condition.as_ref().map(|v| {
            Tensor::from_data(
                TensorData::new(v.clone(), fixture.condition_shape.unwrap()),
                &device,
            )
        });
        let mut samples = Vec::new();
        let mut control: Option<(f64, usize)> = None;
        for i in 0..(warmup + iterations) {
            let start = Instant::now();
            let result = decoder.forward(input.clone(), source.clone(), conditioning.clone());
            let output = black_box(result.into_data().try_into_vec::<f32>().unwrap());
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            if i >= warmup {
                samples.push(elapsed_ms);
            }
            let s: f64 = output.iter().map(|v| *v as f64).sum();
            if let Some((sum, len)) = control {
                assert!(
                    (s - sum).abs() < 1e-4 && output.len() == len,
                    "nondeterministic NSF result"
                );
            } else {
                control = Some((s, output.len()));
            }
        }
        let (sum, len) = control.unwrap();
        println!(
            "{{\"frames\":{frames},\"median_ms\":{},\"iterations\":{iterations},\"sum\":{sum},\"samples\":{len}}}",
            median(samples)
        );
    }
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("Stage43 CPU benchmark requires --no-default-features --features cpu");
    std::process::exit(2);
}
