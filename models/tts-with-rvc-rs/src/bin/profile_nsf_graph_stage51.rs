#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
use burn::tensor::{Device, Tensor, TensorData, capture};
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
#[derive(Clone, Serialize)]
struct Sample {
    dispatch_ms: f64,
    readback_ms: f64,
    total_ms: f64,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct Case {
    frames: usize,
    hardware_graph: bool,
    capture_setup_ms: f64,
    ordinary_median_ms: f64,
    replay_median_ms: f64,
    ordinary_dispatch_median_ms: f64,
    replay_dispatch_median_ms: f64,
    ordinary_readback_median_ms: f64,
    replay_readback_median_ms: f64,
    ordinary_over_replay: f64,
    max_abs_error: f32,
    exact_fp32_equal: bool,
    repeated_graph_max_abs_error: f32,
    output_len: usize,
    ordinary: Vec<Sample>,
    replay: Vec<Sample>,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct Report {
    stage: u32,
    backend: String,
    kernel: String,
    execution: String,
    prepack: bool,
    frames: Vec<usize>,
    iterations: usize,
    warmup: usize,
    load_ms: f64,
    basic_graph_hardware: bool,
    cases: Vec<Case>,
    semantics: &'static str,
    warning: &'static str,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn argument<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|v| v == flag)
        .and_then(|p| args.get(p + 1))
        .map(String::as_str)
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn sample_median(samples: &[Sample], accessor: impl Fn(&Sample) -> f64) -> f64 {
    median(&samples.iter().map(accessor).collect::<Vec<_>>())
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn max_error(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn make_inputs(
    fixture: &NsfDecoderFixture,
    frames: usize,
    device: &Device,
) -> (Tensor<3>, Tensor<3>, Option<Tensor<3>>) {
    let [batch, channels, original] = fixture.input_shape;
    assert_eq!(batch, 1);
    assert_eq!(fixture.source_shape[2] % original, 0);
    let upsample = fixture.source_shape[2] / original;
    let mut input_data = Vec::with_capacity(channels * frames);
    for channel in 0..channels {
        for index in 0..frames {
            input_data.push(fixture.input[channel * original + index % original]);
        }
    }
    let harmonic = (0..frames * upsample)
        .map(|index| fixture.harmonic_source[index % fixture.source_shape[2]])
        .collect::<Vec<f32>>();
    let input = Tensor::from_data(TensorData::new(input_data, [1, channels, frames]), device);
    let source = Tensor::from_data(TensorData::new(harmonic, [1, 1, frames * upsample]), device);
    let condition = fixture.condition.as_ref().map(|data| {
        Tensor::from_data(
            TensorData::new(data.clone(), fixture.condition_shape.unwrap()),
            device,
        )
    });
    (input, source, condition)
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn measure_tensor(result: Tensor<3>, started: Instant) -> (Sample, Vec<f32>) {
    let launched = Instant::now();
    let output = result.into_data().try_into_vec::<f32>().unwrap();
    let completed = Instant::now();
    (
        Sample {
            dispatch_ms: (launched - started).as_secs_f64() * 1000.0,
            readback_ms: (completed - launched).as_secs_f64() * 1000.0,
            total_ms: (completed - started).as_secs_f64() * 1000.0,
        },
        output,
    )
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let backend = argument(&args, "--backend").unwrap_or("cuda");
    let device = match backend {
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        #[cfg(feature = "cpu")]
        "cpu" => Device::cpu(),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(Default::default()),
        _ => panic!("backend {backend} was not compiled into the profiler"),
    };
    let execution_name = argument(&args, "--execution").unwrap_or("sequential");
    let execution = match execution_name {
        "sequential" => ResidualExecution::Sequential,
        "parallel" => ResidualExecution::Parallel,
        _ => panic!("invalid --execution"),
    };
    let iterations: usize = argument(&args, "--iterations")
        .unwrap_or("20")
        .parse()
        .expect("invalid iterations");
    let warmup: usize = argument(&args, "--warmup")
        .unwrap_or("5")
        .parse()
        .expect("invalid warmup");
    let frames = argument(&args, "--frames")
        .unwrap_or("64,256,1024")
        .split(',')
        .map(|s| s.parse::<usize>().expect("invalid frame count"))
        .collect::<Vec<_>>();
    assert!(iterations > 0 && frames.iter().all(|&v| v > 0));

    let fixture: NsfDecoderFixture =
        serde_json::from_str(include_str!("../../tests/fixtures/nsf_decoder.json")).unwrap();
    let checkpoint_path = argument(&args, "--checkpoint")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .join("synthetic_nsf_decoder.pth")
        });
    let checkpoint = RvcCheckpoint::open(&checkpoint_path).unwrap();
    let loaded = Instant::now();
    let decoder = fixture.load(&checkpoint, &device).unwrap();
    let load_ms = loaded.elapsed().as_secs_f64() * 1000.0;

    let probe_input: Tensor<3> = Tensor::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 1, 4]),
        &device,
    );
    let mut probe = capture(&device, || probe_input.clone() + 1.0);
    let basic_graph_hardware = probe.is_hardware();
    let probe_output = unsafe { probe.replay() }
        .clone()
        .into_data()
        .try_into_vec::<f32>()
        .expect("basic graph probe output");
    assert_eq!(probe_output, vec![2.0, 3.0, 4.0, 5.0]);
    drop(probe);
    drop(probe_input);

    let mut cases = Vec::new();
    for &frame_count in &frames {
        let (input, source, condition) = make_inputs(&fixture, frame_count, &device);
        let run = || {
            decoder.forward_with_execution(
                input.clone(),
                source.clone(),
                condition.clone(),
                execution,
            )
        };
        let reference_started = Instant::now();
        let (reference_timing, reference) = measure_tensor(run(), reference_started);
        assert!(!reference.is_empty());
        let capture_started = Instant::now();
        let mut graph = capture(&device, run);
        let capture_setup_ms = capture_started.elapsed().as_secs_f64() * 1000.0;
        let hardware_graph = graph.is_hardware();
        let mut ordinary = Vec::with_capacity(iterations);
        let mut replay = Vec::with_capacity(iterations);
        let mut repeated_error = 0.0f32;
        let mut complete_equal = true;
        let mut max_abs = 0.0f32;
        let mut previous_replay: Option<Vec<f32>> = None;
        for i in 0..(warmup + iterations) {
            let reverse = i % 2 == 1;
            let mut ordinary_result = None;
            let mut replay_result = None;
            for graph_first in [reverse, !reverse] {
                if graph_first {
                    let started = Instant::now();
                    let tensor = unsafe { graph.replay() }.clone();
                    replay_result = Some(measure_tensor(tensor, started));
                } else {
                    let started = Instant::now();
                    ordinary_result = Some(measure_tensor(
                        decoder.forward_with_execution(
                            input.clone(),
                            source.clone(),
                            condition.clone(),
                            execution,
                        ),
                        started,
                    ));
                }
            }
            let (ordinary_sample, normal_out) = ordinary_result.unwrap();
            let (replay_sample, graph_out) = replay_result.unwrap();
            assert!(graph_out.iter().all(|v| v.is_finite()));
            assert!(normal_out.iter().all(|v| v.is_finite()));
            if i >= warmup {
                let err = max_error(&normal_out, &graph_out);
                max_abs = max_abs.max(err);
                complete_equal &= normal_out
                    .iter()
                    .zip(&graph_out)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                if let Some(previous) = previous_replay.as_ref() {
                    repeated_error = repeated_error.max(max_error(previous, &graph_out));
                }
                previous_replay = Some(graph_out);
                ordinary.push(ordinary_sample);
                replay.push(replay_sample);
            }
        }
        assert!(
            max_abs <= 1.0e-5,
            "graph output FP32 parity failure at {frame_count}: {max_abs}"
        );
        assert!(
            repeated_error <= 1.0e-5,
            "graph output not stable at {frame_count}"
        );
        let ordinary_median_ms = sample_median(&ordinary, |s| s.total_ms);
        let replay_median_ms = sample_median(&replay, |s| s.total_ms);
        eprintln!(
            "NSF {frame_count}: graph_hardware={hardware_graph}, normal={ordinary_median_ms:.4}ms, replay={replay_median_ms:.4}ms, max_abs={max_abs:.2e}, capture={capture_setup_ms:.2}ms, initial_normal={:.2}ms",
            reference_timing.total_ms,
        );
        cases.push(Case {
            frames: frame_count,
            hardware_graph,
            capture_setup_ms,
            ordinary_median_ms,
            replay_median_ms,
            ordinary_dispatch_median_ms: sample_median(&ordinary, |s| s.dispatch_ms),
            replay_dispatch_median_ms: sample_median(&replay, |s| s.dispatch_ms),
            ordinary_readback_median_ms: sample_median(&ordinary, |s| s.readback_ms),
            replay_readback_median_ms: sample_median(&replay, |s| s.readback_ms),
            ordinary_over_replay: ordinary_median_ms / replay_median_ms,
            max_abs_error: max_abs,
            exact_fp32_equal: complete_equal,
            repeated_graph_max_abs_error: repeated_error,
            output_len: reference.len(),
            ordinary,
            replay,
        });
    }
    let report = Report {
        stage: 51,
        backend: backend.to_string(),
        kernel: std::env::var("RVC_NSF_CONV_KERNEL").unwrap_or_default(),
        execution: execution_name.to_owned(),
        prepack: std::env::var_os("RVC_CONV_WEIGHT_PREPACK").is_some(),
        frames,
        warmup,
        iterations,
        load_ms,
        basic_graph_hardware,
        cases,
        semantics: "Full synthetic NSF on fixed, stable input buffers; alternating ordinary forward and graph replay in one process; every iteration includes blocking output readback. Graph capture setup excluded from replay timing.",
        warning: "Static-input replay only. GPU graphs hold raw addresses and are not yet validated for changing inference inputs. A non-hardware graph is software closure replay, not CUDA Graph acceleration.",
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

#[cfg(not(any(feature = "cpu", feature = "cuda", feature = "wgpu")))]
fn main() {
    eprintln!("Stage51 profiler requires --features cpu|cuda|wgpu");
    std::process::exit(2);
}
