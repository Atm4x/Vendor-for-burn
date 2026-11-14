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
#[derive(Serialize)]
struct Sample {
    submit_ms: f64,
    readback_ms: f64,
    total_ms: f64,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct Case {
    frames: usize,
    median_total_ms: f64,
    median_submit_ms: f64,
    median_readback_ms: f64,
    repeated_max_abs_error: f32,
    output_sum: f64,
    output_len: usize,
    samples: Vec<Sample>,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct Report {
    stage: u32,
    backend: String,
    preparation: &'static str,
    execution: &'static str,
    load_ms: f64,
    warmup: usize,
    iterations: usize,
    cases: Vec<Case>,
    semantics: &'static str,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
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
    let backend = option(&args, "--backend").unwrap_or("cuda");
    let device = match backend {
        #[cfg(feature = "cpu")]
        "cpu" => Device::cpu(),
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(Default::default()),
        _ => panic!("backend {backend} not included in this executable"),
    };
    let requested_execution = option(&args, "--execution").unwrap_or("sequential");
    let (execution, execution_name) = match requested_execution {
        "sequential" => (ResidualExecution::Sequential, "sequential"),
        "parallel" => (ResidualExecution::Parallel, "parallel"),
        _ => panic!("--execution must be sequential or parallel"),
    };
    let iterations: usize = option(&args, "--iterations")
        .unwrap_or("20")
        .parse()
        .expect("invalid --iterations");
    let warmup: usize = option(&args, "--warmup")
        .unwrap_or("5")
        .parse()
        .expect("invalid --warmup");
    let frames: Vec<usize> = option(&args, "--frames")
        .unwrap_or("64,256,1024")
        .split(',')
        .map(|value| value.parse().expect("invalid --frames"))
        .collect();
    assert!(iterations > 0 && frames.iter().all(|frames| *frames > 0));
    let dump_prefix = option(&args, "--dump-prefix");
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
    let prepacked = std::env::var_os("RVC_CONV_WEIGHT_PREPACK").is_some();
    let load_started = Instant::now();
    let decoder = fixture.load(&checkpoint, &device).unwrap();
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let mut cases = Vec::new();
    for frame_count in frames {
        let (input, source, condition) = make_input(&fixture, frame_count, &device);
        let mut samples = Vec::new();
        let mut previous_output: Option<Vec<f32>> = None;
        let mut repeat_error = 0.0_f32;
        for iteration in 0..(warmup + iterations) {
            let begin = Instant::now();
            let output_tensor = decoder.forward_with_execution(
                input.clone(),
                source.clone(),
                condition.clone(),
                execution,
            );
            let submitted = Instant::now();
            let output = output_tensor.into_data().try_into_vec::<f32>().unwrap();
            let completed = Instant::now();
            if iteration >= warmup {
                if let Some(previous) = &previous_output {
                    repeat_error = repeat_error.max(max_abs_error(previous, &output));
                }
                previous_output = Some(output);
                samples.push(Sample {
                    submit_ms: (submitted - begin).as_secs_f64() * 1000.0,
                    readback_ms: (completed - submitted).as_secs_f64() * 1000.0,
                    total_ms: (completed - begin).as_secs_f64() * 1000.0,
                });
            }
        }
        assert!(
            repeat_error <= 1e-5,
            "repeated NSF output changed: {repeat_error}"
        );
        let output = previous_output.expect("no measured output");
        if let Some(prefix) = dump_prefix {
            let bytes: Vec<u8> = output
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(format!("{prefix}-{frame_count}.f32le"), bytes).unwrap();
        }
        let total: Vec<f64> = samples.iter().map(|value| value.total_ms).collect();
        let submit: Vec<f64> = samples.iter().map(|value| value.submit_ms).collect();
        let readback: Vec<f64> = samples.iter().map(|value| value.readback_ms).collect();
        cases.push(Case {
            frames: frame_count,
            median_total_ms: median(&total),
            median_submit_ms: median(&submit),
            median_readback_ms: median(&readback),
            repeated_max_abs_error: repeat_error,
            output_sum: output.iter().map(|value| *value as f64).sum(),
            output_len: output.len(),
            samples,
        });
    }
    let report = Report {
        stage: 49,
        backend: backend.to_owned(),
        preparation: if prepacked {
            "cached_weight_layout"
        } else {
            "baseline"
        },
        execution: execution_name,
        load_ms,
        warmup,
        iterations,
        cases,
        semantics: "warmed full NSF forward with blocking CPU output readback; model initialization measured separately, excluding checkpoint file load and device creation",
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

#[cfg(not(any(feature = "cpu", feature = "cuda", feature = "wgpu")))]
fn main() {
    eprintln!("NSF prepack profiler requires --features cpu|cuda|wgpu");
    std::process::exit(2);
}
