#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
use burn::tensor::{Device, Tensor, TensorData};
#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
use serde::Serialize;
#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
use std::time::Instant;
#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
use tts_with_rvc_rs::{
    checkpoint::RvcCheckpoint,
    decoder::{NsfDecoderFixture, ResidualExecution},
};

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Clone, Copy)]
enum ExecutionMode {
    Parallel,
    Sequential,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
impl ExecutionMode {
    fn name(self) -> &'static str {
        match self {
            Self::Parallel => "parallel",
            Self::Sequential => "sequential",
        }
    }

    fn execution(self) -> ResidualExecution {
        match self {
            Self::Parallel => ResidualExecution::Parallel,
            Self::Sequential => ResidualExecution::Sequential,
        }
    }
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Clone, Copy, Serialize)]
struct TimingSample {
    submit_ms: f64,
    readback_ms: f64,
    total_ms: f64,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct VariantResult {
    mode: &'static str,
    submit_median_ms: f64,
    readback_median_ms: f64,
    total_median_ms: f64,
    total_min_ms: f64,
    total_p90_ms: f64,
    samples: Vec<TimingSample>,
    output_len: usize,
    output_sum: f64,
    repeated_max_abs_error: f32,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct CaseResult {
    frames: usize,
    variants: Vec<VariantResult>,
    cross_mode_full_tensor_max_abs_error: Option<f32>,
    parallel_over_sequential_speedup: Option<f64>,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct Report {
    experiment: &'static str,
    backend: String,
    iterations: usize,
    warmup: usize,
    mode: String,
    timing_semantics: &'static str,
    parity_semantics: &'static str,
    cases: Vec<CaseResult>,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn quantile(values: &[f64], percentile: f64) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn max_abs_error(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn make_input(
    fixture: &NsfDecoderFixture,
    frames: usize,
    device: &Device,
) -> (Tensor<3>, Tensor<3>, Option<Tensor<3>>) {
    let [batch, channels, original] = fixture.input_shape;
    assert_eq!(batch, 1);
    let upsample = fixture.source_shape[2] / original;
    let mut values = Vec::with_capacity(channels * frames);
    for channel in 0..channels {
        for time in 0..frames {
            values.push(fixture.input[channel * original + time % original]);
        }
    }
    let harmonic: Vec<f32> = (0..frames * upsample)
        .map(|time| fixture.harmonic_source[time % fixture.source_shape[2]])
        .collect();
    let input = Tensor::from_data(TensorData::new(values, [1, channels, frames]), device);
    let source = Tensor::from_data(TensorData::new(harmonic, [1, 1, frames * upsample]), device);
    let condition = fixture.condition.as_ref().map(|values| {
        Tensor::from_data(
            TensorData::new(values.clone(), fixture.condition_shape.unwrap()),
            device,
        )
    });
    (input, source, condition)
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations: usize = option(&args, "--iterations")
        .unwrap_or("20")
        .parse()
        .expect("invalid iterations");
    let warmup: usize = option(&args, "--warmup")
        .unwrap_or("5")
        .parse()
        .expect("invalid warmup");
    let frames: Vec<usize> = option(&args, "--frames")
        .unwrap_or("64,256,1024")
        .split(',')
        .map(|item| item.parse().expect("invalid frame count"))
        .collect();
    assert!(iterations > 0 && frames.iter().all(|count| *count > 0));
    let requested_mode = option(&args, "--mode").unwrap_or("both");
    let modes: Vec<ExecutionMode> = match requested_mode {
        "both" => vec![ExecutionMode::Parallel, ExecutionMode::Sequential],
        "parallel" => vec![ExecutionMode::Parallel],
        "sequential" => vec![ExecutionMode::Sequential],
        _ => panic!("--mode must be both, parallel, or sequential"),
    };
    let backend = option(&args, "--backend").unwrap_or("cuda");
    let device = match backend {
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        #[cfg(feature = "cpu")]
        "cpu" => Device::cpu(),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(Default::default()),
        _ => panic!("backend {backend} was not compiled into this executable"),
    };
    let fixture: NsfDecoderFixture =
        serde_json::from_str(include_str!("../../tests/fixtures/nsf_decoder.json")).unwrap();
    let checkpoint_path = option(&args, "--checkpoint")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .join("synthetic_nsf_decoder.pth")
        });
    let checkpoint = RvcCheckpoint::open(&checkpoint_path).unwrap();
    let decoder = fixture.load(&checkpoint, &device).unwrap();
    let mut cases = Vec::new();

    for frame_count in frames {
        let (input, source, condition) = make_input(&fixture, frame_count, &device);
        let mut timings: Vec<Vec<TimingSample>> = vec![Vec::new(); modes.len()];
        let mut last_outputs: Vec<Option<Vec<f32>>> = vec![None; modes.len()];
        let mut repeated_errors = vec![0.0_f32; modes.len()];

        for round in 0..(warmup + iterations) {
            for offset in 0..modes.len() {
                let index = (round + offset) % modes.len();
                let mode = modes[index];
                let started = Instant::now();
                let tensor = decoder.forward_with_execution(
                    input.clone(),
                    source.clone(),
                    condition.clone(),
                    mode.execution(),
                );
                let submitted = Instant::now();
                let output = tensor.into_data().try_into_vec::<f32>().unwrap();
                let finished = Instant::now();
                if round >= warmup {
                    timings[index].push(TimingSample {
                        submit_ms: (submitted - started).as_secs_f64() * 1000.0,
                        readback_ms: (finished - submitted).as_secs_f64() * 1000.0,
                        total_ms: (finished - started).as_secs_f64() * 1000.0,
                    });
                    if let Some(previous) = &last_outputs[index] {
                        repeated_errors[index] =
                            repeated_errors[index].max(max_abs_error(previous, &output));
                    }
                    last_outputs[index] = Some(output);
                }
            }
        }

        let mut variants = Vec::new();
        for (index, mode) in modes.iter().copied().enumerate() {
            let samples = std::mem::take(&mut timings[index]);
            let output = last_outputs[index].as_ref().unwrap();
            let submit: Vec<f64> = samples.iter().map(|sample| sample.submit_ms).collect();
            let readback: Vec<f64> = samples.iter().map(|sample| sample.readback_ms).collect();
            let total: Vec<f64> = samples.iter().map(|sample| sample.total_ms).collect();
            assert!(repeated_errors[index] <= 1e-5, "nondeterministic output");
            variants.push(VariantResult {
                mode: mode.name(),
                submit_median_ms: quantile(&submit, 0.5),
                readback_median_ms: quantile(&readback, 0.5),
                total_median_ms: quantile(&total, 0.5),
                total_min_ms: quantile(&total, 0.0),
                total_p90_ms: quantile(&total, 0.9),
                samples,
                output_len: output.len(),
                output_sum: output.iter().map(|value| *value as f64).sum(),
                repeated_max_abs_error: repeated_errors[index],
            });
        }

        let parity = if modes.len() == 2 {
            let error = max_abs_error(
                last_outputs[0].as_ref().unwrap(),
                last_outputs[1].as_ref().unwrap(),
            );
            assert!(error <= 1e-5, "sequential/parallel parity failed: {error}");
            Some(error)
        } else {
            None
        };
        let speedup = if modes.len() == 2 {
            Some(variants[0].total_median_ms / variants[1].total_median_ms)
        } else {
            None
        };
        eprintln!(
            "NSF {frame_count}: {}",
            variants
                .iter()
                .map(|variant| {
                    format!(
                        "{} {:.4}ms (submit {:.4}, readback {:.4})",
                        variant.mode,
                        variant.total_median_ms,
                        variant.submit_median_ms,
                        variant.readback_median_ms
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        );
        cases.push(CaseResult {
            frames: frame_count,
            variants,
            cross_mode_full_tensor_max_abs_error: parity,
            parallel_over_sequential_speedup: speedup,
        });
    }

    let report = Report {
        experiment: "NSF residual execution topology A/B",
        backend: backend.to_owned(),
        iterations,
        warmup,
        mode: requested_mode.to_owned(),
        timing_semantics: "warmed host wall time with blocking CPU readback; submit includes Burn/CubeCL launch scheduling, readback includes device completion; neither is a pure GPU kernel-time measurement",
        parity_semantics: "full FP32 output comparison, across modes when both selected; repeated output checks within each mode",
        cases,
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

#[cfg(not(any(feature = "cpu", feature = "cuda", feature = "wgpu")))]
fn main() {
    eprintln!("Build with --no-default-features --features cpu|cuda|wgpu");
    std::process::exit(2);
}