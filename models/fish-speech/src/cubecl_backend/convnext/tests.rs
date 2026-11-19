use super::*;
use cubecl::Device;
use fish_speech_core::firefly_cpu::Conv1dKernel;

fn weights(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 17 + seed * 7) % 47) as f32 / 90.0 - 0.25)
        .collect()
}

fn block(channels: usize, kernel: usize, seed: usize, gamma: bool) -> ConvNextBlock {
    let wide = channels * 4;
    ConvNextBlock {
        depthwise: Conv1dKernel {
            in_channels: channels,
            out_channels: channels,
            kernel_size: kernel,
            stride: 1,
            dilation: 1,
            groups: channels,
            weights: weights(channels * kernel, seed),
            bias: Some(weights(channels, seed + 2)),
        },
        norm_weight: weights(channels, seed + 4),
        norm_bias: weights(channels, seed + 6),
        pointwise1_weight: weights(wide * channels, seed + 8),
        pointwise1_bias: weights(wide, seed + 10),
        pointwise2_weight: weights(channels * wide, seed + 12),
        pointwise2_bias: weights(channels, seed + 14),
        gamma: gamma.then(|| weights(channels, seed + 16)),
        epsilon: 1e-6,
    }
}

#[test]
fn complete_convnext_block_matches_cpu_reference() {
    let gpu = CubeClConvNextExecutor::new(Device::Cpu(Default::default()).client());
    for (channels, frames, batch, kernel, gamma) in [
        (1, 1, 1, 7, true),
        (2, 5, 2, 7, false),
        (4, 9, 2, 7, true),
        (8, 3, 1, 3, true),
        (16, 11, 2, 7, false),
    ] {
        let b = block(channels, kernel, frames, gamma);
        let input = AudioTensor::new(
            batch,
            channels,
            frames,
            weights(batch * channels * frames, kernel),
        )
        .unwrap();
        let expected = b.forward(&input).unwrap();
        let actual = gpu.forward(&b, &input).unwrap();
        for (i, (&x, &y)) in actual.data.iter().zip(&expected.data).enumerate() {
            assert!(
                (x - y).abs() <= 3e-4 + 1e-4 * y.abs(),
                "case ({channels},{frames},{batch},{kernel}) at {i}: {x} != {y}"
            );
        }
    }
}

#[test]
fn multiple_blocks_remain_device_resident_until_final_readback() {
    let gpu = CubeClConvNextExecutor::new(Device::Cpu(Default::default()).client());
    let input = AudioTensor::new(2, 4, 11, weights(88, 31)).unwrap();
    let blocks = vec![
        block(4, 7, 3, true),
        block(4, 7, 5, false),
        block(4, 7, 7, true),
    ];
    let expected = blocks
        .iter()
        .try_fold(input.clone(), |value, block| block.forward(&value))
        .unwrap();
    let actual = gpu.forward_chain(&blocks, &input).unwrap();
    for (i, (&x, &y)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!((x - y).abs() <= 6e-4, "block chain at {i}: {x} vs {y}");
    }
}

#[test]
fn grouped_and_downsampling_conv_matches_reference() {
    let gpu = CubeClConvNextExecutor::new(Device::Cpu(Default::default()).client());
    for (batch, channels, output_channels, frames, kernel, stride, dilation, groups) in [
        (2, 8, 8, 9, 7, 1, 1, 8),
        (1, 4, 8, 13, 2, 2, 1, 1),
        (2, 4, 8, 11, 3, 2, 1, 2),
        (1, 8, 4, 7, 1, 1, 1, 1),
        (1, 4, 4, 12, 3, 1, 3, 1),
    ] {
        let conv = Conv1dKernel {
            in_channels: channels,
            out_channels: output_channels,
            kernel_size: kernel,
            stride,
            dilation,
            groups,
            weights: weights(output_channels * (channels / groups) * kernel, 5),
            bias: Some(weights(output_channels, 9)),
        };
        let input = AudioTensor::new(
            batch,
            channels,
            frames,
            weights(batch * channels * frames, 19),
        )
        .unwrap();
        let expected = conv.forward(&input).unwrap();
        let actual = gpu.conv1d(&conv, &input).unwrap();
        assert_eq!(
            (actual.batch, actual.channels, actual.frames),
            (expected.batch, expected.channels, expected.frames)
        );
        for (i, (&x, &y)) in actual.data.iter().zip(&expected.data).enumerate() {
            assert!(
                (x - y).abs() < 4e-5,
                "grouped Conv1d {channels}→{output_channels} at {i}: {x} vs {y}"
            );
        }
    }
}

#[test]
fn graph_ir_checkpoint_binding_matches_convnext_reference() {
    let graph = FireflyGraph::reference_encode(1, 4096).unwrap();
    let prefix = "backbone.stages.0.0";
    let node = graph
        .nodes
        .iter()
        .find(|n| n.name == format!("{prefix}.dwconv"))
        .unwrap();
    let shape = graph.nodes[node.inputs[0]].output;
    let b = block(shape.channels, 7, 42, true);
    let checkpoint = BTreeMap::from([
        (
            format!("{prefix}.dwconv.conv.weight"),
            b.depthwise.weights.clone(),
        ),
        (
            format!("{prefix}.dwconv.conv.bias"),
            b.depthwise.bias.clone().unwrap(),
        ),
        (format!("{prefix}.norm.weight"), b.norm_weight.clone()),
        (format!("{prefix}.norm.bias"), b.norm_bias.clone()),
        (
            format!("{prefix}.pwconv1.weight"),
            b.pointwise1_weight.clone(),
        ),
        (format!("{prefix}.pwconv1.bias"), b.pointwise1_bias.clone()),
        (
            format!("{prefix}.pwconv2.weight"),
            b.pointwise2_weight.clone(),
        ),
        (format!("{prefix}.pwconv2.bias"), b.pointwise2_bias.clone()),
        (format!("{prefix}.gamma"), b.gamma.clone().unwrap()),
    ]);
    let input = AudioTensor::new(
        shape.batch,
        shape.channels,
        shape.frames,
        weights(shape.batch * shape.channels * shape.frames, 87),
    )
    .unwrap();
    let expected = b.forward(&input).unwrap();
    let gpu = CubeClConvNextExecutor::new(Device::Cpu(Default::default()).client());
    let actual = gpu
        .forward_ir_block(&graph, &checkpoint, prefix, &input)
        .unwrap();
    for (i, (&x, &y)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (x - y).abs() <= 2e-3 + 2e-4 * y.abs(),
            "IR block at {i}: {x} vs {y}"
        );
    }
    let mut missing = checkpoint;
    missing.remove(&format!("{prefix}.gamma"));
    assert!(
        gpu.forward_ir_block(&graph, &missing, prefix, &input)
            .is_err()
    );
}

#[test]
fn invalid_block_is_rejected_before_dispatch() {
    let gpu = CubeClConvNextExecutor::new(Device::Cpu(Default::default()).client());
    let input = AudioTensor::new(1, 4, 3, vec![0.0; 12]).unwrap();
    let mut b = block(4, 7, 2, true);
    b.pointwise1_weight.pop();
    assert!(gpu.forward(&b, &input).is_err());
    b = block(4, 7, 2, true);
    b.norm_bias[0] = f32::NAN;
    assert!(gpu.forward(&b, &input).is_err());
}
