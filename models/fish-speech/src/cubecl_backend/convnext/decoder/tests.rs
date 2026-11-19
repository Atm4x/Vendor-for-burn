use super::*;
use cubecl::Device;
use fish_speech_core::synthesis::AcousticDecoder;
use fish_speech_core::{
    firefly_ir::{FireflyNode, FireflyStage, FireflyTensor},
    firefly_weighted::WeightedFireflyExecutor,
    graph_ir::{Layout, ScalarType, TensorType},
};

fn data(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 17 + salt * 11) % 61) as f32 / 150.0 - 0.18)
        .collect()
}

#[test]
fn native_convtranspose_matches_cpu_reference_across_shapes() {
    let gpu = CubeClFireflyDecoder::new(Device::Cpu(Default::default()).client());
    for (batch, ci, co, frames, kernel, stride) in [
        (1, 2, 3, 2, 4, 2),
        (2, 4, 2, 5, 6, 3),
        (1, 8, 5, 3, 8, 4),
        (2, 3, 1, 7, 3, 1),
    ] {
        let conv = TransConv1dKernel {
            in_channels: ci,
            out_channels: co,
            kernel_size: kernel,
            stride,
            dilation: 1,
            weights: data(ci * co * kernel, 7),
            bias: Some(data(co, 9)),
        };
        let input = AudioTensor::new(batch, ci, frames, data(batch * ci * frames, 13)).unwrap();
        let reference = conv.forward(&input).unwrap();
        let actual = gpu.transconv1d(&conv, &input).unwrap();
        assert_eq!(
            (actual.batch, actual.channels, actual.frames),
            (reference.batch, reference.channels, reference.frames)
        );
        for (i, (a, b)) in actual.data.iter().zip(&reference.data).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "ConvTranspose ({ci},{co},{kernel},{stride}) index {i}: {a} vs {b}"
            );
        }
    }
}

fn node(
    name: &str,
    op: FireflyOp,
    inputs: &[usize],
    channels: usize,
    frames: usize,
    dtype: ScalarType,
    keys: &[&str],
) -> FireflyNode {
    FireflyNode {
        name: name.into(),
        operation: op,
        inputs: inputs.to_vec(),
        output: FireflyTensor {
            batch: 1,
            channels,
            frames,
            dtype,
        },
        checkpoint_keys: keys.iter().map(|s| (*s).into()).collect(),
    }
}

fn short_decoder() -> (FireflyGraph, BTreeMap<String, Vec<f32>>) {
    let mut graph = FireflyGraph {
        stage: FireflyStage::WaveformDecode,
        nodes: vec![
            node(
                "codes",
                FireflyOp::CodeInput,
                &[],
                2,
                2,
                ScalarType::I32,
                &[],
            ),
            node(
                "fsq",
                FireflyOp::FsqDecode {
                    groups: 2,
                    quantizers: 1,
                },
                &[0],
                4,
                2,
                ScalarType::F32,
                &["g0w", "g0b", "g1w", "g1b"],
            ),
            node(
                "upsample",
                FireflyOp::FishTransConv {
                    kernel: 4,
                    stride: 2,
                },
                &[1],
                3,
                4,
                ScalarType::F32,
                &["upw", "upb"],
            ),
            node(
                "activation",
                FireflyOp::SiLu,
                &[2],
                3,
                4,
                ScalarType::F32,
                &[],
            ),
            node(
                "res_a",
                FireflyOp::FishConv {
                    kernel: 3,
                    stride: 1,
                    dilation: 2,
                    groups: 1,
                },
                &[3],
                3,
                4,
                ScalarType::F32,
                &["aw", "ab"],
            ),
            node(
                "residual",
                FireflyOp::Add,
                &[3, 4],
                3,
                4,
                ScalarType::F32,
                &[],
            ),
            node(
                "res_b",
                FireflyOp::FishConv {
                    kernel: 3,
                    stride: 1,
                    dilation: 1,
                    groups: 3,
                },
                &[3],
                3,
                4,
                ScalarType::F32,
                &["bw", "bb"],
            ),
            node(
                "parallel",
                FireflyOp::ParallelMean,
                &[5, 6],
                3,
                4,
                ScalarType::F32,
                &[],
            ),
            node(
                "post",
                FireflyOp::FishConv {
                    kernel: 3,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                &[7],
                1,
                4,
                ScalarType::F32,
                &["pw", "pb"],
            ),
            node("tanh", FireflyOp::Tanh, &[8], 1, 4, ScalarType::F32, &[]),
            node(
                "waveform.audio_mask",
                FireflyOp::Mask,
                &[9],
                1,
                4,
                ScalarType::F32,
                &[],
            ),
        ],
        weights: BTreeMap::new(),
        input: 0,
        output: 10,
        valid_frames: 3,
    };
    let mut weights = BTreeMap::new();
    for (name, shape, seed) in [
        ("g0w", vec![2, 4], 1),
        ("g0b", vec![2], 2),
        ("g1w", vec![2, 4], 3),
        ("g1b", vec![2], 4),
        ("upw", vec![4, 3, 4], 5),
        ("upb", vec![3], 6),
        ("aw", vec![3, 3, 3], 7),
        ("ab", vec![3], 8),
        ("bw", vec![3, 1, 3], 9),
        ("bb", vec![3], 10),
        ("pw", vec![1, 3, 3], 11),
        ("pb", vec![1], 12),
    ] {
        let n = shape.iter().product::<usize>();
        graph.weights.insert(
            name.into(),
            TensorType::new(shape, ScalarType::F32, Layout::Plain),
        );
        weights.insert(name.into(), data(n, seed));
    }
    graph.validate().unwrap();
    (graph, weights)
}

#[test]
fn complete_device_fsq_transconv_residual_mean_to_pcm_matches_cpu() {
    let (graph, weights) = short_decoder();
    let codes = [0i32, 17, 999, -1];
    let reference = WeightedFireflyExecutor::new(graph.clone(), weights.clone())
        .unwrap()
        .decode(&codes)
        .unwrap();
    let gpu = CubeClFireflyDecoder::new(Device::Cpu(Default::default()).client());
    let actual = gpu.decode(&graph, &weights, &codes).unwrap();
    assert_eq!(actual.len(), 3);
    assert_eq!(actual.len(), reference.len());
    for (i, (a, b)) in actual.iter().zip(&reference).enumerate() {
        assert!((a - b).abs() < 6e-4, "PCM at {i}: {a} vs {b}");
    }
}

#[test]
fn decoder_preflight_rejects_invalid_codes_and_weights() {
    let (graph, mut weights) = short_decoder();
    let gpu = CubeClFireflyDecoder::new(Device::Cpu(Default::default()).client());
    assert!(gpu.decode(&graph, &weights, &[1000, 0, 0, 0]).is_err());
    assert!(gpu.decode(&graph, &weights, &[-2, 0, 0, 0]).is_err());
    assert!(gpu.decode(&graph, &weights, &[0, 0, 0]).is_err());
    weights.get_mut("upw").unwrap().pop();
    assert!(gpu.decode(&graph, &weights, &[0, 0, 0, 0]).is_err());
}

#[test]
fn full_hifigan_graph_preflight_requires_checkpoint_keys() {
    let graph = FireflyGraph::waveform_decode(1, 1, 1).unwrap();
    let gpu = CubeClFireflyDecoder::new(Device::Cpu(Default::default()).client());
    assert!(graph.nodes.len() > 240);
    assert!(gpu.decode(&graph, &BTreeMap::new(), &[0; 8]).is_err());
}

#[test]
#[ignore = "full-size 240+ node Firefly HiFiGAN graph; slow on Tracel LLVM CPU"]
fn full_waveform_graph_native_device_pcm_matches_weighted_cpu() {
    let graph = FireflyGraph::waveform_decode(1, 1, 1).unwrap();
    let checkpoint = graph
        .weights
        .iter()
        .map(|(key, tensor)| {
            let n = tensor.shape.iter().product::<usize>();
            let values = if key.contains("norm.weight") {
                vec![1.0; n]
            } else if key.ends_with("gamma") {
                vec![0.1; n]
            } else if key.ends_with("bias") {
                vec![0.0; n]
            } else {
                (0..n)
                    .map(|i| ((i * 17 + 11) % 101) as f32 / 33000.0 - 0.0015)
                    .collect()
            };
            (key.clone(), values)
        })
        .collect::<BTreeMap<_, _>>();
    let codes = (0..8)
        .map(|i| ((i * 127 + 73) % 1000) as i32)
        .collect::<Vec<_>>();
    let reference = WeightedFireflyExecutor::new(graph.clone(), checkpoint.clone())
        .unwrap()
        .decode(&codes)
        .unwrap();
    let gpu = CubeClFireflyDecoder::new(Device::Cpu(Default::default()).client());
    let actual = gpu.decode(&graph, &checkpoint, &codes).unwrap();
    assert_eq!(actual.len(), 2048);
    assert_eq!(actual.len(), reference.len());
    for (i, (a, b)) in actual.iter().zip(&reference).enumerate() {
        assert!(
            (a - b).abs() < 2e-3 + 1e-3 * b.abs(),
            "full audio {i}: {a} vs {b}"
        );
    }
}

#[test]
#[ignore = "full native reference encoder plus full native HiFiGAN; CPU reference is expensive"]
fn full_native_reference_encoder_to_pcm_matches_cpu_fish_speech() {
    use fish_speech_core::firefly_cpu::AudioTensor;
    let pcm = (0..4096)
        .map(|i| (i as f32 * 0.019).sin() * 0.24)
        .collect::<Vec<_>>();
    let encoder_graph = FireflyGraph::reference_encode(1, pcm.len()).unwrap();
    let decoder_graph = FireflyGraph::waveform_decode(1, 2, 2).unwrap();
    let weight_values = |key: &str, shape: &[usize]| {
        let n = shape.iter().product::<usize>();
        if key.contains("norm.weight") {
            vec![1.0f32; n]
        } else if key.ends_with("gamma") {
            vec![0.1; n]
        } else if key.ends_with("bias") {
            vec![0.0; n]
        } else {
            (0..n)
                .map(|i| ((i * 17 + 11) % 101) as f32 / 33000.0 - 0.0015)
                .collect()
        }
    };
    let mut all_weights = BTreeMap::new();
    for (key, ty) in encoder_graph.weights.iter().chain(&decoder_graph.weights) {
        let values = weight_values(key, &ty.shape);
        if let Some(old) = all_weights.insert(key.clone(), values.clone()) {
            assert_eq!(old, values);
        }
    }
    let encoder_weights: BTreeMap<String, Vec<f32>> = encoder_graph
        .weights
        .keys()
        .map(|k| (k.clone(), all_weights[k].clone()))
        .collect();
    let decoder_weights: BTreeMap<String, Vec<f32>> = decoder_graph
        .weights
        .keys()
        .map(|k| (k.clone(), all_weights[k].clone()))
        .collect();
    let client = Device::Cpu(Default::default()).client();
    let encoder = CubeClPreparedFireflyEncoder::new(
        client.clone(),
        encoder_graph.clone(),
        encoder_weights.clone(),
    )
    .unwrap();
    let indices = encoder.encode_mono_pcm(&pcm).unwrap();
    let expected_codes = WeightedFireflyExecutor::new(encoder_graph, encoder_weights)
        .unwrap()
        .encode(AudioTensor::new(1, 1, pcm.len(), pcm).unwrap())
        .unwrap()
        .0;
    assert_eq!(indices, expected_codes);
    let native = CubeClFireflyDecoder::new(client)
        .decode(&decoder_graph, &decoder_weights, &indices)
        .unwrap();
    let expected = WeightedFireflyExecutor::new(decoder_graph, decoder_weights)
        .unwrap()
        .decode(&indices)
        .unwrap();
    assert_eq!(native.len(), 4096);
    for (i, (a, b)) in native.iter().zip(&expected).enumerate() {
        assert!(
            (a - b).abs() < 2e-3 + 1e-3 * b.abs(),
            "codec PCM at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn acoustic_decoder_adapter_rejects_invalid_codebooks() {
    let client = Device::Cpu(Default::default()).client();
    let mut decoder = CubeClFireflyAcousticDecoder::new(client, BTreeMap::new());
    assert!(decoder.decode(&vec![vec![1]; 7]).is_err());
    assert!(
        decoder
            .decode(
                &vec![vec![1]; 8]
                    .into_iter()
                    .enumerate()
                    .map(|(i, mut x)| {
                        if i == 2 {
                            x.clear()
                        }
                        x
                    })
                    .collect::<Vec<_>>()
            )
            .is_err()
    );
    assert!(decoder.decode(&vec![vec![1000]; 8]).is_err());
}
