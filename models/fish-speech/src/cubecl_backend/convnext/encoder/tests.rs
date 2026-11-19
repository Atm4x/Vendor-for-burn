use super::*;
use cubecl::Device;
use fish_speech_core::firefly_cpu::layer_norm_channels_first;

fn checkpoint_for_prefix(graph: &FireflyGraph, stop: &str) -> BTreeMap<String, Vec<f32>> {
    let stop = graph
        .nodes
        .iter()
        .position(|node| node.name == stop)
        .unwrap();
    graph.nodes[..=stop]
        .iter()
        .flat_map(|node| node.checkpoint_keys.iter())
        .map(|key| {
            let shape = &graph.weights[key].shape;
            let n = shape.iter().product::<usize>();
            let values = if key.ends_with("norm.weight")
                || key.ends_with("downsample_layers.0.1.weight")
                || key.ends_with(".0.weight") && key.contains("backbone.downsample_layers.1")
            {
                vec![1.0; n]
            } else if key.ends_with("gamma") {
                vec![0.1; n]
            } else if key.ends_with("bias") || key.ends_with("norm.bias") {
                vec![0.0; n]
            } else {
                (0..n)
                    .map(|i| ((i * 17 + 3) % 53) as f32 / 275.0 - 0.095)
                    .collect()
            };
            (key.clone(), values)
        })
        .collect()
}

fn conv_from_graph(
    graph: &FireflyGraph,
    weights: &BTreeMap<String, Vec<f32>>,
    name: &str,
) -> Conv1dKernel {
    let node = graph.nodes.iter().find(|node| node.name == name).unwrap();
    let before = graph.nodes[node.inputs[0]].output;
    let (kernel, stride, dilation, groups) = match node.operation {
        FireflyOp::FishConv {
            kernel,
            stride,
            dilation,
            groups,
        } => (kernel, stride, dilation, groups),
        FireflyOp::Conv1x1 => (1, 1, 1, 1),
        _ => panic!("not a convolution"),
    };
    Conv1dKernel {
        in_channels: before.channels,
        out_channels: node.output.channels,
        kernel_size: kernel,
        stride,
        dilation,
        groups,
        weights: weights[&node.checkpoint_keys[0]].clone(),
        bias: Some(weights[&node.checkpoint_keys[1]].clone()),
    }
}

fn convnext_from_graph(
    graph: &FireflyGraph,
    weights: &BTreeMap<String, Vec<f32>>,
    prefix: &str,
) -> ConvNextBlock {
    let p = |suffix: &str| -> Vec<f32> { weights[&format!("{prefix}.{suffix}")].clone() };
    ConvNextBlock {
        depthwise: conv_from_graph(graph, weights, &format!("{prefix}.dwconv")),
        norm_weight: p("norm.weight"),
        norm_bias: p("norm.bias"),
        pointwise1_weight: p("pwconv1.weight"),
        pointwise1_bias: p("pwconv1.bias"),
        pointwise2_weight: p("pwconv2.weight"),
        pointwise2_bias: p("pwconv2.bias"),
        gamma: Some(p("gamma")),
        epsilon: 1e-6,
    }
}

#[test]
fn actual_graph_ir_encodes_mel_through_downsample_norm_and_residual() {
    let graph = FireflyGraph::reference_encode(1, 3073).unwrap();
    let mel_shape = graph
        .nodes
        .iter()
        .find(|node| node.operation == FireflyOp::LogMel)
        .unwrap()
        .output;
    let mel = AudioTensor::new(
        mel_shape.batch,
        mel_shape.channels,
        mel_shape.frames,
        (0..tensor_len(mel_shape).unwrap())
            .map(|i| (i % 37) as f32 / 59.0 - 0.3)
            .collect(),
    )
    .unwrap();
    let stop = "backbone.stages.0.0.residual";
    let checkpoint = checkpoint_for_prefix(&graph, stop);
    let conv = conv_from_graph(&graph, &checkpoint, "backbone.downsample_layers.0.0");
    let stage = conv.forward(&mel).unwrap();
    let expected_norm = layer_norm_channels_first(
        &stage,
        &checkpoint["backbone.downsample_layers.0.1.weight"],
        &checkpoint["backbone.downsample_layers.0.1.bias"],
        1e-6,
    )
    .unwrap();
    let expected = convnext_from_graph(&graph, &checkpoint, "backbone.stages.0.0")
        .forward(&expected_norm)
        .unwrap();
    let gpu = CubeClConvNextExecutor::new(Device::Cpu(Default::default()).client());
    for (point, reference) in [
        ("backbone.downsample_layers.0.0", &stage),
        ("backbone.downsample_layers.0.1", &expected_norm),
        (stop, &expected),
    ] {
        let actual = gpu
            .reference_features_until(&graph, &checkpoint, &mel, point)
            .unwrap();
        assert_eq!(
            (actual.batch, actual.channels, actual.frames),
            (reference.batch, reference.channels, reference.frames)
        );
        for (i, (&x, &y)) in actual.data.iter().zip(reference.data.iter()).enumerate() {
            assert!(
                (x - y).abs() <= 3e-3 + y.abs() * 4e-4,
                "{point} at {i}: {x} vs {y}"
            );
        }
    }
}

#[test]
fn graph_ir_preflight_rejects_missing_invalid_and_unsupported_boundaries() {
    let graph = FireflyGraph::reference_encode(2, 4096).unwrap();
    let mel_shape = graph
        .nodes
        .iter()
        .find(|node| node.operation == FireflyOp::LogMel)
        .unwrap()
        .output;
    let mel = AudioTensor::new(
        mel_shape.batch,
        mel_shape.channels,
        mel_shape.frames,
        vec![0.0; tensor_len(mel_shape).unwrap()],
    )
    .unwrap();
    let gpu = CubeClConvNextExecutor::new(Device::Cpu(Default::default()).client());
    let stop = "backbone.stages.0.0.residual";
    let mut checkpoint = checkpoint_for_prefix(&graph, stop);
    assert!(
        gpu.reference_features_until(&graph, &checkpoint, &mel, "quantizer.fsq_encode")
            .is_err()
    );
    checkpoint.remove("backbone.stages.0.0.gamma");
    assert!(
        gpu.reference_features_until(&graph, &checkpoint, &mel, stop)
            .is_err()
    );
    checkpoint.insert("backbone.stages.0.0.gamma".into(), vec![f32::NAN; 128]);
    assert!(
        gpu.reference_features_until(&graph, &checkpoint, &mel, stop)
            .is_err()
    );
    checkpoint.insert("backbone.stages.0.0.gamma".into(), vec![0.0; 128]);
    checkpoint
        .get_mut("backbone.stages.0.0.pwconv2.weight")
        .unwrap()
        .pop();
    assert!(
        gpu.reference_features_until(&graph, &checkpoint, &mel, stop)
            .is_err()
    );
    assert!(
        gpu.reference_features_until(&graph, &checkpoint, &mel, "spec_transform.logmel")
            .is_err()
    );
    let wrong_mel = AudioTensor::new(
        1,
        mel.channels,
        mel.frames,
        vec![0.0; mel.channels * mel.frames],
    )
    .unwrap();
    assert!(
        gpu.reference_features_until(&graph, &checkpoint, &wrong_mel, stop)
            .is_err()
    );
}

#[test]
#[ignore = "full 156-node CubeCL CPU encoder parity requires JIT-compiling a large graph"]
fn complete_reference_graph_encodes_without_cpu_feature_readback() {
    use fish_speech_core::{
        audio::FIREFLY_LEVELS,
        fsq_cpu::{FsqCodebook, FsqGroupProjection, GroupedResidualFsqEncoder},
    };
    let graph = FireflyGraph::reference_encode(1, 3073).unwrap();
    let mel_shape = graph
        .nodes
        .iter()
        .find(|node| node.operation == FireflyOp::LogMel)
        .unwrap()
        .output;
    let mel = AudioTensor::new(
        mel_shape.batch,
        mel_shape.channels,
        mel_shape.frames,
        (0..tensor_len(mel_shape).unwrap())
            .map(|i| (i % 43) as f32 / 67.0 - 0.3)
            .collect(),
    )
    .unwrap();
    let feature_id = graph.nodes[graph.output].inputs[0];
    let feature_name = graph.nodes[feature_id].name.clone();
    let checkpoint = checkpoint_for_prefix(&graph, "quantizer.fsq_encode");
    let gpu = CubeClConvNextExecutor::new(Device::Cpu(Default::default()).client());
    let features = gpu
        .reference_features_until(&graph, &checkpoint, &mel, &feature_name)
        .unwrap();
    let fsq = &graph.nodes[graph.output];
    let encoder = GroupedResidualFsqEncoder {
        codebook: FsqCodebook::new(&FIREFLY_LEVELS).unwrap(),
        num_quantizers: 1,
        projections: (0..8)
            .map(|group| FsqGroupProjection {
                weights: checkpoint[&fsq.checkpoint_keys[group * 2]].clone(),
                bias: checkpoint[&fsq.checkpoint_keys[group * 2 + 1]].clone(),
            })
            .collect(),
    };
    let expected = encoder.encode(&features).unwrap();
    let actual = gpu
        .reference_encode_from_mel(&graph, &checkpoint, &mel)
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(
        actual.len(),
        graph.nodes[graph.output].output.batch * 8 * graph.nodes[graph.output].output.frames
    );
}

#[test]
#[ignore = "full two-batch 156-node CubeCL CPU encoder JIT parity"]
fn cpu_logmel_boundary_feeds_batched_device_encoder() {
    use fish_speech_core::reference_audio::ReferenceMelPreprocessor;
    let first = (0..4096)
        .map(|i| (i as f32 / 27.0).sin() * 0.25)
        .collect::<Vec<_>>();
    let second = (0..4096)
        .map(|i| (i as f32 / 61.0).cos() * 0.35)
        .collect::<Vec<_>>();
    let pcm = &[first.as_slice(), second.as_slice()];
    let prepared = ReferenceMelPreprocessor::new()
        .prepare_mono(pcm, 44_100)
        .unwrap();
    let graph = prepared.reference_graph().unwrap();
    let checkpoint = checkpoint_for_prefix(&graph, "quantizer.fsq_encode");
    let gpu = CubeClConvNextExecutor::new(Device::Cpu(Default::default()).client());
    let expected = gpu
        .reference_encode_from_mel(&graph, &checkpoint, &prepared.mels)
        .unwrap();
    let actual = gpu
        .reference_encode_with_cpu_logmel(&checkpoint, pcm, 44_100)
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(
        actual.len(),
        2 * 8 * graph.nodes[graph.output].output.frames
    );
    assert!(
        gpu.reference_encode_with_cpu_logmel(&checkpoint, pcm, 16_000)
            .is_err()
    );
    assert!(
        gpu.reference_encode_with_cpu_logmel(&checkpoint, &[&first, &second[..4000]], 44_100)
            .is_err()
    );
}

#[test]
fn cubecl_pcm_logmel_feeds_reference_encoder_graph_without_host_mel() {
    use fish_speech_core::spectrogram_cpu::FishLogMel;
    use super::super::super::logmel::CubeClLogMel;
    let pcm: Vec<f32> = (0..2048)
        .map(|i| ((i as f64 * std::f64::consts::TAU * 431.0 / 44_100.0).sin() * 0.25) as f32)
        .collect();
    let graph = FireflyGraph::reference_encode(1, pcm.len()).unwrap();
    let stop = "backbone.downsample_layers.0.1";
    let checkpoint = checkpoint_for_prefix(&graph, stop);
    let client = Device::Cpu(Default::default()).client();
    let gpu = CubeClConvNextExecutor::new(client.clone());
    let frontend = CubeClLogMel::new(client);
    let actual = gpu
        .reference_features_until_from_cubecl_pcm(&frontend, &graph, &checkpoint, &pcm, stop)
        .unwrap();
    let mel = FishLogMel::new().transform_mono(&pcm).unwrap();
    let expected = gpu.reference_features_until(&graph, &checkpoint, &mel, stop).unwrap();
    assert_eq!(actual.data.len(), expected.data.len());
    for (i, (&a, &b)) in actual.data.iter().zip(expected.data.iter()).enumerate() {
        assert!((a - b).abs() <= 0.015, "prefix at {i}: {a} vs {b}");
    }
}

#[test]
#[ignore = "full 156-node CubeCL CPU encoder+FSQ with FFT JIT requires an explicit run"]
fn complete_native_pcm_to_fsq_agrees_with_reference_logmel_path() {
    use super::super::super::logmel::CubeClLogMel;
    use fish_speech_core::spectrogram_cpu::FishLogMel;
    let pcm: Vec<f32> = (0..3073)
        .map(|i| ((i as f64 * std::f64::consts::TAU * 350.0 / 44_100.0).sin() * 0.2) as f32)
        .collect();
    let graph = FireflyGraph::reference_encode(1, pcm.len()).unwrap();
    let checkpoint = checkpoint_for_prefix(&graph, "quantizer.fsq_encode");
    let client = Device::Cpu(Default::default()).client();
    let gpu = CubeClConvNextExecutor::new(client.clone());
    let front = CubeClLogMel::new(client);
    let actual = gpu.reference_encode_from_cubecl_pcm(&front, &checkpoint, &pcm).unwrap();
    let expected = gpu.reference_encode_from_mel(
        &graph, &checkpoint, &FishLogMel::new().transform_mono(&pcm).unwrap(),
    ).unwrap();
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "full 156-node prepared Firefly model must compile full CPU JIT explicitly"]
fn prepared_firefly_model_reuses_weights_across_pcm_requests() {
    use super::super::super::logmel::CubeClLogMel;
    let pcm: Vec<f32> = (0..3073)
        .map(|i| ((i as f64 * std::f64::consts::TAU * 270.0 / 44_100.0).sin() * 0.18) as f32)
        .collect();
    let graph = FireflyGraph::reference_encode(1, pcm.len()).unwrap();
    let checkpoint = checkpoint_for_prefix(&graph, "quantizer.fsq_encode");
    let client = Device::Cpu(Default::default()).client();
    let normal = CubeClConvNextExecutor::new(client.clone());
    let frontend = CubeClLogMel::new(client.clone());
    let expected = normal.reference_encode_from_cubecl_pcm(&frontend, &checkpoint, &pcm).unwrap();
    let prepared = CubeClPreparedFireflyEncoder::new(client, graph, checkpoint).unwrap();
    assert!(prepared.prepared_weight_count() > 100);
    assert_eq!(prepared.encode_mono_pcm(&pcm).unwrap(), expected);
    assert_eq!(prepared.encode_mono_pcm(&pcm).unwrap(), expected);
    let mut invalid = pcm.clone();
    invalid[101] = f32::NAN;
    assert!(prepared.encode_mono_pcm(&invalid).is_err());
}
