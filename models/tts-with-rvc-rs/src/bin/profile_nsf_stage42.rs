#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
use burn::tensor::{Device, Tensor, TensorData};
#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
use serde::Serialize;
#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
use std::time::Instant;
#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
use tts_with_rvc_rs::{
    checkpoint::RvcCheckpoint,
    decoder::{
        NsfDecoderFixture, NsfProfileStep, NsfScheduling, ResidualAggregation, ResidualExecution,
    },
};

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct CaseResult {
    frames: usize,
    zero_seeded_ms: f64,
    first_branch_ms: f64,
    flush_per_branch_ms: f64,
    flush_per_stage_ms: f64,
    sync_per_branch_ms: f64,
    flush_per_pair_ms: f64,
    flush_per_conv_ms: f64,
    sync_per_pair_ms: f64,
    relative_speedup: f64,
    max_abs_error: f32,
    reference_max_error: Option<f32>,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct RunResult {
    kind: &'static str,
    backend: String,
    semantics: &'static str,
    iterations: usize,
    warmup: usize,
    cases: Vec<CaseResult>,
    profile: Vec<NsfProfileStep>,
    profiled_total_ms: f64,
    profiled_vs_baseline_max_abs_error: f32,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn make_input(
    fixture: &NsfDecoderFixture,
    frames: usize,
    device: &Device,
) -> (Tensor<3>, Tensor<3>, Option<Tensor<3>>) {
    let [batch, channels, original] = fixture.input_shape;
    let upsample = fixture.source_shape[2] / original;
    assert_eq!(batch, 1);
    let values: Vec<f32> = (0..channels)
        .flat_map(|c| (0..frames).map(move |t| fixture.input[c * original + t % original]))
        .collect();
    let harmonic: Vec<f32> = (0..frames * upsample)
        .map(|t| fixture.harmonic_source[t % fixture.source_shape[2]])
        .collect();
    let features = Tensor::from_data(TensorData::new(values, [1, channels, frames]), device);
    let source = Tensor::from_data(TensorData::new(harmonic, [1, 1, frames * upsample]), device);
    let conditioning = fixture.condition.as_ref().map(|v| {
        Tensor::from_data(
            TensorData::new(v.clone(), fixture.condition_shape.unwrap()),
            device,
        )
    });
    (features, source, conditioning)
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations: usize = args
        .iter()
        .position(|s| s == "--iterations")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse().unwrap())
        .unwrap_or(8);
    let warmup: usize = args
        .iter()
        .position(|s| s == "--warmup")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse().unwrap())
        .unwrap_or(3);
    let frames: Vec<usize> = args
        .iter()
        .position(|s| s == "--frames")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![4, 16, 64]);
    assert!(iterations > 0 && frames.iter().all(|x| *x > 0));
    let backend_name = args
        .iter()
        .position(|s| s == "--backend")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("cpu");
    let device = match backend_name {
        #[cfg(feature = "cpu")]
        "cpu" => Device::cpu(),
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(Default::default()),
        _ => panic!("backend {backend_name} is not compiled into this executable"),
    };
    let fixture: NsfDecoderFixture =
        serde_json::from_str(include_str!("../../tests/fixtures/nsf_decoder.json")).unwrap();
    let checkpoint_path = args
        .iter()
        .position(|s| s == "--checkpoint")
        .and_then(|i| args.get(i + 1))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let sibling = std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .join("synthetic_nsf_decoder.pth");
            if sibling.exists() {
                sibling
            } else {
                std::path::PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/synthetic_nsf_decoder.pth"
                ))
            }
        });
    let checkpoint = RvcCheckpoint::open(&checkpoint_path).unwrap();
    let decoder = fixture.load(&checkpoint, &device).unwrap();
    let mut cases = Vec::new();
    for size in frames {
        let (input, source, condition) = make_input(&fixture, size, &device);
        let mut samples: [Vec<f64>; 8] = std::array::from_fn(|_| Vec::new());
        let mut outputs: [Option<Vec<f32>>; 8] = std::array::from_fn(|_| None);
        for round in 0..(warmup + iterations) {
            for offset in 0..8 {
                let index = (offset + round) % 8;
                let (aggregation, scheduling) = match index {
                    0 => (ResidualAggregation::ZeroSeeded, NsfScheduling::Default),
                    1 => (ResidualAggregation::FirstBranch, NsfScheduling::Default),
                    2 => (
                        ResidualAggregation::ZeroSeeded,
                        NsfScheduling::FlushPerBranch,
                    ),
                    3 => (
                        ResidualAggregation::ZeroSeeded,
                        NsfScheduling::FlushPerStage,
                    ),
                    4 => (
                        ResidualAggregation::ZeroSeeded,
                        NsfScheduling::SyncPerBranch,
                    ),
                    5 => (ResidualAggregation::ZeroSeeded, NsfScheduling::FlushPerPair),
                    6 => (ResidualAggregation::ZeroSeeded, NsfScheduling::FlushPerConv),
                    7 => (ResidualAggregation::ZeroSeeded, NsfScheduling::SyncPerPair),
                    _ => unreachable!(),
                };
                let begin = Instant::now();
                let out = decoder
                    .forward_with_scheduling(
                        input.clone(),
                        source.clone(),
                        condition.clone(),
                        ResidualExecution::Sequential,
                        aggregation,
                        scheduling,
                    )
                    .into_data()
                    .try_into_vec::<f32>()
                    .unwrap();
                let elapsed = begin.elapsed().as_secs_f64() * 1000.0;
                if round >= warmup {
                    samples[index].push(elapsed);
                }
                outputs[index] = Some(out);
            }
        }
        let baseline = outputs[0].take().unwrap();
        let max_abs_error = outputs[1..]
            .iter()
            .flat_map(|v| v.as_ref().unwrap().iter().zip(&baseline))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_error <= 1e-5,
            "NSF scheduling discrepancy {max_abs_error}"
        );
        let reference_max_error = if size == fixture.input_shape[2] {
            Some(
                baseline
                    .iter()
                    .zip(&fixture.expected_waveform)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max),
            )
        } else {
            None
        };
        let m: Vec<f64> = samples.into_iter().map(median).collect();
        eprintln!(
            "NSF {size} frames: default {:.3} first {:.3} branch-flush {:.3} stage-flush {:.3} branch-sync {:.3} pair-flush {:.3} conv-flush {:.3} pair-sync {:.3} ms, error={max_abs_error:e}",
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]
        );
        cases.push(CaseResult {
            frames: size,
            zero_seeded_ms: m[0],
            first_branch_ms: m[1],
            flush_per_branch_ms: m[2],
            flush_per_stage_ms: m[3],
            sync_per_branch_ms: m[4],
            flush_per_pair_ms: m[5],
            flush_per_conv_ms: m[6],
            sync_per_pair_ms: m[7],
            relative_speedup: m[0] / m[1],
            max_abs_error,
            reference_max_error,
        });
    }
    let (input, source, condition) = make_input(&fixture, 64, &device);
    let reference = decoder
        .forward_with_aggregation(
            input.clone(),
            source.clone(),
            condition.clone(),
            ResidualExecution::Sequential,
            ResidualAggregation::ZeroSeeded,
        )
        .into_data()
        .try_into_vec::<f32>()
        .unwrap();
    let _ = decoder.profile_cpu(
        input.clone(),
        source.clone(),
        condition.clone(),
        ResidualAggregation::ZeroSeeded,
    );
    let begin = Instant::now();
    let (profiled, profile) =
        decoder.profile_cpu(input, source, condition, ResidualAggregation::ZeroSeeded);
    let profiled = profiled.into_data().try_into_vec::<f32>().unwrap();
    let profiled_total_ms = begin.elapsed().as_secs_f64() * 1000.0;
    let profiled_vs_baseline_max_abs_error = reference
        .iter()
        .zip(&profiled)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(profiled_vs_baseline_max_abs_error < 1e-5);
    let result = RunResult {
        kind: "NSF real synthetic checkpoint CPU execution",
        backend: backend_name.to_owned(),
        semantics: "end-to-end NSF forward including eager dispatch and blocking output readback; detailed profile separately inserts a blocking readback at each operator boundary",
        warmup,
        iterations,
        cases,
        profile,
        profiled_total_ms,
        profiled_vs_baseline_max_abs_error,
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

#[cfg(not(any(feature = "cpu", feature = "cuda", feature = "wgpu")))]
fn main() {
    eprintln!("This profiler requires --no-default-features --features cpu|cuda|wgpu");
    std::process::exit(2);
}
