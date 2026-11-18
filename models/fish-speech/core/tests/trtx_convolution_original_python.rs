use std::collections::BTreeMap;

use fish_speech_core::firefly_cpu::{AudioTensor, Conv1dKernel, TransConv1dKernel};
use fish_speech_core::firefly_ir::{
    FireflyGraph, FireflyNode, FireflyOp, FireflyStage, FireflyTensor,
};
use fish_speech_core::firefly_trtx_conv::{Conv2dKind, FireflyConv2dPlan};
use fish_speech_core::graph_ir::{Layout, ScalarType, TensorType};

fn floats(line: &str, label: &str) -> Vec<f32> {
    let (name, values) = line.split_once(' ').expect("nonempty fixture tensor");
    assert_eq!(name, label);
    values
        .split_whitespace()
        .map(|v| v.parse::<f32>().unwrap())
        .collect()
}

#[test]
fn fish_convolutions_match_original_pytorch_fixtures_and_trt_geometry() {
    let mut lines = include_str!("../fixtures/trtx_convolution_original_python.txt").lines();
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let parts = header.split_whitespace().collect::<Vec<_>>();
        assert_eq!(parts.len(), 11);
        let name = parts[0];
        let dims = parts[2..]
            .iter()
            .map(|v| v.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        let [batch, cin, cout, frames, kernel, stride, dilation, groups, expected_frames] =
            <[usize; 9]>::try_from(dims).unwrap();
        let kind = match parts[1] {
            "conv" | "linear" => Conv2dKind::Convolution,
            "transconv" => Conv2dKind::Deconvolution,
            _ => panic!("unknown fixture kind"),
        };
        let input = floats(lines.next().unwrap(), "input");
        let weights = floats(lines.next().unwrap(), "weight");
        let bias = floats(lines.next().unwrap(), "bias");
        let expected = floats(lines.next().unwrap(), "output");
        let input_tensor = AudioTensor::new(batch, cin, frames, input).unwrap();
        let result = match kind {
            Conv2dKind::Convolution => Conv1dKernel {
                in_channels: cin,
                out_channels: cout,
                kernel_size: kernel,
                stride,
                dilation,
                groups,
                weights: weights.clone(),
                bias: Some(bias.clone()),
            }
            .forward(&input_tensor)
            .unwrap(),
            Conv2dKind::Deconvolution => TransConv1dKernel {
                in_channels: cin,
                out_channels: cout,
                kernel_size: kernel,
                stride,
                dilation,
                weights: weights.clone(),
                bias: Some(bias.clone()),
            }
            .forward(&input_tensor)
            .unwrap(),
        };
        assert_eq!(result.frames, expected_frames, "{name}");
        assert_eq!(result.data.len(), expected.len(), "{name}");
        let worst = result
            .data
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 2e-5,
            "{name}: maximum PyTorch FP32 difference={worst}"
        );

        let weight_shape = if parts[1] == "linear" {
            vec![cout, cin]
        } else if kind == Conv2dKind::Convolution {
            vec![cout, cin / groups, kernel]
        } else {
            vec![cin, cout, kernel]
        };
        let mut requirements = BTreeMap::new();
        requirements.insert(
            "test.weight".to_owned(),
            TensorType::new(weight_shape, ScalarType::F32, Layout::Plain),
        );
        requirements.insert(
            "test.bias".to_owned(),
            TensorType::new(vec![cout], ScalarType::F32, Layout::Plain),
        );
        let graph = FireflyGraph {
            stage: FireflyStage::WaveformDecode,
            nodes: vec![
                FireflyNode {
                    name: "audio".into(),
                    operation: FireflyOp::AudioInput,
                    inputs: vec![],
                    output: FireflyTensor {
                        batch,
                        channels: cin,
                        frames,
                        dtype: ScalarType::F32,
                    },
                    checkpoint_keys: vec![],
                },
                FireflyNode {
                    name: name.into(),
                    operation: if parts[1] == "linear" {
                        FireflyOp::LinearChannelsLast
                    } else if kind == Conv2dKind::Convolution {
                        FireflyOp::FishConv {
                            kernel,
                            stride,
                            dilation,
                            groups,
                        }
                    } else {
                        FireflyOp::FishTransConv { kernel, stride }
                    },
                    inputs: vec![0],
                    output: FireflyTensor {
                        batch,
                        channels: cout,
                        frames: expected_frames,
                        dtype: ScalarType::F32,
                    },
                    checkpoint_keys: vec!["test.weight".into(), "test.bias".into()],
                },
            ],
            weights: requirements,
            input: 0,
            output: 1,
            valid_frames: expected_frames,
        };
        let plan = FireflyConv2dPlan::from_node(&graph, 1).unwrap().unwrap();
        assert_eq!(plan.kind, kind);
        assert_eq!(plan.kernel_nchw[3], kernel);
        assert_eq!(plan.input_nchw, [batch, cin, 1, frames]);
        assert_eq!(plan.output_nchw, [batch, cout, 1, expected_frames]);
        assert_eq!(plan.groups, groups);
        if parts[1] == "linear" {
            assert_eq!(plan.source_weight_shape, vec![cout, cin]);
            assert_eq!(plan.kernel_nchw, [cout, cin, 1, 1]);
        }
        plan.validate_runtime_weights(weights.len() * 4, bias.len() * 4)
            .unwrap();
        if kind == Conv2dKind::Deconvolution {
            assert_eq!(plan.trim_after_hw, [0, kernel - stride]);
            assert_eq!(plan.pad_before_hw, [0, 0]);
        }
        cases += 1;
    }
    assert_eq!(cases, 6);
}
