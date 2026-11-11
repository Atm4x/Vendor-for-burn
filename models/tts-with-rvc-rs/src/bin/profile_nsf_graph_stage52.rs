#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
use burn::cubecl::bytes::Bytes;
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

#[cfg(feature = "cuda")]
type CubeBackend = burn::backend::Cuda;
#[cfg(all(not(feature = "cuda"), feature = "cpu"))]
type CubeBackend = burn::backend::Cpu;
#[cfg(all(not(feature = "cuda"), not(feature = "cpu"), feature = "wgpu"))]
type CubeBackend = burn::backend::Wgpu;

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Clone)]
struct Inputs {
    input: Vec<f32>,
    source: Vec<f32>,
    condition: Option<Vec<f32>>,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
impl Inputs {
    fn for_step(fixture: &NsfDecoderFixture, frames: usize, step: usize) -> Self {
        let [batch, channels, original] = fixture.input_shape;
        assert_eq!(batch, 1);
        assert_eq!(fixture.source_shape[2] % original, 0);
        let upsample = fixture.source_shape[2] / original;
        let gain = [1.0_f32, -0.75, 0.45, 1.35][step % 4];
        let source_gain = [1.0_f32, 0.6, -0.8, 1.2][step % 4];
        let input = (0..channels)
            .flat_map(|channel| {
                (0..frames).map(move |i| {
                    let x = fixture.input[channel * original + i % original];
                    gain * x + ((i + channel + step) % 9) as f32 * 0.0003
                })
            })
            .collect();
        let source = (0..frames * upsample)
            .map(|i| source_gain * fixture.harmonic_source[i % fixture.source_shape[2]])
            .collect();
        let condition = fixture.condition.as_ref().map(|values| {
            values
                .iter()
                .map(|&v| v * gain + step as f32 * 0.0001)
                .collect()
        });
        Self {
            input,
            source,
            condition,
        }
    }

    fn tensors(
        &self,
        device: &Device,
        fixture: &NsfDecoderFixture,
        frames: usize,
    ) -> (Tensor<3>, Tensor<3>, Option<Tensor<3>>) {
        let source_frames = frames * fixture.source_shape[2] / fixture.input_shape[2];
        let input = Tensor::from_data(
            TensorData::new(self.input.clone(), [1, fixture.input_shape[1], frames]),
            device,
        );
        let source = Tensor::from_data(
            TensorData::new(self.source.clone(), [1, 1, source_frames]),
            device,
        );
        let condition = self.condition.as_ref().map(|values| {
            Tensor::from_data(
                TensorData::new(values.clone(), fixture.condition_shape.unwrap()),
                device,
            )
        });
        (input, source, condition)
    }
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
struct PersistentInput {
    tensor: Tensor<3>,
    primitive: <CubeBackend as burn::backend::BackendTypes>::FloatTensorPrimitive,
    values_len: usize,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
impl PersistentInput {
    fn new(tensor: Tensor<3>) -> Self {
        let primitive = tensor
            .clone()
            .try_into_primitive::<CubeBackend>()
            .expect("CubeCL tensor primitive is required for stable memory refresh");
        assert_eq!(
            primitive.dtype,
            burn::tensor::DType::F32,
            "raw FP32 upload requires an FP32 allocation"
        );
        let dims = tensor.dims();
        assert_eq!(primitive.meta.shape().len(), 3, "unexpected storage layout");
        let mut expected_stride = 1;
        for axis in (0..3).rev() {
            assert_eq!(primitive.meta.shape()[axis], dims[axis], "tiled layout unsupported");
            if dims[axis] > 1 {
                assert_eq!(
                    primitive.meta.strides()[axis],
                    expected_stride,
                    "padded or strided input: unsafe to upload raw contiguous FP32 bytes"
                );
            }
            expected_stride *= dims[axis];
        }
        assert_eq!(
            primitive.handle.size_in_used(),
            (expected_stride * size_of::<f32>()) as u64,
            "storage byte size differs from logical FP32 input"
        );
        Self {
            tensor,
            primitive,
            values_len: expected_stride,
        }
    }

    fn refresh(&self, values: &[f32]) {
        assert_eq!(values.len(), self.values_len, "shape changed during graph replay");
        self.primitive
            .client
            .write(&self.primitive.handle, Bytes::from_elems(values.to_vec()));
    }
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn read(output: Tensor<3>) -> Vec<f32> {
    output.into_data().try_into_vec::<f32>().unwrap()
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct Measurement {
    input_step: usize,
    refresh_ms: f64,
    replay_and_readback_ms: f64,
    normal_and_readback_ms: f64,
    max_abs_error: f32,
    bitwise_equal: bool,
    changed_from_first_input: bool,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct Case {
    frames: usize,
    hardware_graph: bool,
    capture_setup_ms: f64,
    max_abs_error: f32,
    bitwise_equal: bool,
    distinct_outputs: usize,
    median_refresh_ms: f64,
    median_replay_ms: f64,
    median_normal_ms: f64,
    measurements: Vec<Measurement>,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
#[derive(Serialize)]
struct Report {
    stage: u32,
    backend: String,
    frames: Vec<usize>,
    iterations: usize,
    kernel: String,
    prepack: bool,
    basic_refresh_passed: bool,
    basic_graph_hardware: bool,
    cases: Vec<Case>,
    semantics: &'static str,
    limitation: &'static str,
}

#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
fn argument<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
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
        _ => panic!("backend {backend} was not compiled into profiler"),
    };
    let frames = argument(&args, "--frames")
        .unwrap_or("64,256,1024")
        .split(',')
        .map(|f| f.parse::<usize>().expect("invalid frame count"))
        .collect::<Vec<_>>();
    let iterations = argument(&args, "--iterations")
        .unwrap_or("16")
        .parse::<usize>()
        .expect("invalid iteration count");
    assert!(iterations >= 4 && frames.iter().all(|&f| f > 0));
    let execution = match argument(&args, "--execution").unwrap_or("sequential") {
        "sequential" => ResidualExecution::Sequential,
        "parallel" => ResidualExecution::Parallel,
        _ => panic!("invalid residual execution policy"),
    };
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
    let decoder = fixture
        .load(&RvcCheckpoint::open(&checkpoint_path).unwrap(), &device)
        .unwrap();

    let probe = Tensor::from_data(TensorData::new(vec![1_f32, 2.0, 3.0, 4.0], [1, 1, 4]), &device);
    let pinned_probe = PersistentInput::new(probe);
    let mut basic_graph = capture(&device, || pinned_probe.tensor.clone() + 1.0);
    let basic_graph_hardware = basic_graph.is_hardware();
    if backend == "cuda" {
        assert!(basic_graph_hardware, "hardware CUDA Graph unavailable for basic buffer refresh");
    }
    for (values, expected) in [([2.0_f32, 4.0, 6.0, 8.0], [3.0, 5.0, 7.0, 9.0]), ([10.0, -2.0, 0.0, 7.0], [11.0, -1.0, 1.0, 8.0])] {
        pinned_probe.refresh(&values);
        let actual = read(unsafe { basic_graph.replay() }.clone());
        assert_eq!(actual, expected, "persistent input refresh probe failed");
    }
    drop(basic_graph);
    drop(pinned_probe);

    let mut cases = Vec::new();
    for &frame_count in &frames {
        let initial = Inputs::for_step(&fixture, frame_count, 0);
        let (input, source, condition) = initial.tensors(&device, &fixture, frame_count);
        let pinned_input = PersistentInput::new(input);
        let pinned_source = PersistentInput::new(source);
        let pinned_condition = condition.map(PersistentInput::new);
        let graph_input = pinned_input.tensor.clone();
        let graph_source = pinned_source.tensor.clone();
        let graph_condition = pinned_condition.as_ref().map(|p| p.tensor.clone());
        let capture_start = Instant::now();
        let mut graph = capture(&device, || {
            decoder.forward_with_execution(
                graph_input.clone(),
                graph_source.clone(),
                graph_condition.clone(),
                execution,
            )
        });
        let capture_setup_ms = capture_start.elapsed().as_secs_f64() * 1000.0;
        let hardware_graph = graph.is_hardware();
        if backend == "cuda" {
            assert!(hardware_graph, "CUDA graph capture fell back: experiment invalid");
        }

        let mut results = Vec::with_capacity(iterations);
        let mut reference_first: Option<Vec<f32>> = None;
        let mut max_error = 0.0_f32;
        let mut exact = true;
        let mut unique_outputs = std::collections::HashSet::new();
        for i in 0..iterations {
            let values = Inputs::for_step(&fixture, frame_count, i);
            let (normal_input, normal_source, normal_condition) =
                values.tensors(&device, &fixture, frame_count);
            let t = Instant::now();
            pinned_input.refresh(&values.input);
            pinned_source.refresh(&values.source);
            if let (Some(pinned), Some(condition)) = (&pinned_condition, &values.condition) {
                pinned.refresh(condition);
            }
            let refresh_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let replay = read(unsafe { graph.replay() }.clone());
            let replay_and_readback_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let normal = read(decoder.forward_with_execution(
                normal_input,
                normal_source,
                normal_condition,
                execution,
            ));
            let normal_and_readback_ms = t.elapsed().as_secs_f64() * 1000.0;
            assert!(replay.iter().all(|v| v.is_finite()));
            assert!(normal.iter().all(|v| v.is_finite()));
            let error = max_abs(&normal, &replay);
            max_error = max_error.max(error);
            let bitwise_equal = normal
                .iter()
                .zip(&replay)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            exact &= bitwise_equal;
            assert!(bitwise_equal, "dynamic graph FP32 bitwise mismatch at frame {frame_count}, step {i}: max error {error}");
            let changed = reference_first
                .as_ref()
                .is_some_and(|first| max_abs(first, &normal) > 1e-6);
            if reference_first.is_none() {
                reference_first = Some(normal.clone());
            }
            unique_outputs.insert(normal.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            results.push(Measurement {
                input_step: i,
                refresh_ms,
                replay_and_readback_ms,
                normal_and_readback_ms,
                max_abs_error: error,
                bitwise_equal,
                changed_from_first_input: changed,
            });
        }
        assert!(max_error <= 1e-5, "dynamic graph FP32 parity failure: {max_error}");
        assert!(unique_outputs.len() >= 2, "test inputs failed to change NSF output");
        cases.push(Case {
            frames: frame_count,
            hardware_graph,
            capture_setup_ms,
            max_abs_error: max_error,
            bitwise_equal: exact,
            distinct_outputs: unique_outputs.len(),
            median_refresh_ms: median(&results.iter().map(|r| r.refresh_ms).collect::<Vec<_>>()),
            median_replay_ms: median(&results.iter().map(|r| r.replay_and_readback_ms).collect::<Vec<_>>()),
            median_normal_ms: median(&results.iter().map(|r| r.normal_and_readback_ms).collect::<Vec<_>>()),
            measurements: results,
        });
        drop(graph);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            stage: 52,
            backend: backend.to_owned(),
            frames,
            iterations,
            kernel: std::env::var("RVC_NSF_CONV_KERNEL").unwrap_or_default(),
            prepack: std::env::var_os("RVC_CONV_WEIGHT_PREPACK").is_some(),
            basic_refresh_passed: true,
            basic_graph_hardware,
            cases,
            semantics: "Persistent CubeCL buffer writes on capture stream, hardware graph replays with alternating input, source and conditioning; each replay is checked against a fresh ordinary Burn forward with full FP32 CPU readback.",
            limitation: "Synthetic fixed-shape NSF only; upload time measured separately; CPU fallback is not a hardware graph, and no end-to-end live RVC weights were validated.",
        })
        .unwrap()
    );
}

#[cfg(not(any(feature = "cpu", feature = "cuda", feature = "wgpu")))]
fn main() {
    eprintln!("Stage52 requires --features cpu|cuda|wgpu");
    std::process::exit(2);
}
