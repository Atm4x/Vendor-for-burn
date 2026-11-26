//! Strict Graph IR preflight, executed before any decoder dispatch.

use super::*;
use fish_speech_core::{
    audio::{FIREFLY_LEVELS, fish_conv_output_len},
    firefly_ir::FireflyStage,
    graph_ir::ScalarType,
};

pub(super) fn validate_decoder(
    graph: &FireflyGraph,
    checkpoint: &BTreeMap<String, Vec<f32>>,
    indices: &[i32],
) -> Result<()> {
    graph.validate()?;
    if graph.stage != FireflyStage::WaveformDecode {
        return Err(FishError::InvalidShape(
            "expected Firefly waveform decoder graph",
        ));
    }
    let input = graph.nodes[graph.input].output;
    if input.dtype != ScalarType::I32
        || indices.len() != extent(&[input.batch, input.channels, input.frames])?
        || indices.iter().any(|&v| v < -1 || v >= 1000)
    {
        return Err(FishError::InvalidShape(
            "invalid CubeCL acoustic code payload",
        ));
    }
    for (id, node) in graph.nodes.iter().enumerate() {
        if id == graph.input {
            continue;
        }
        let expected = match node.operation {
            FireflyOp::FsqDecode { groups, .. } => groups * 2,
            FireflyOp::FishConv { .. }
            | FireflyOp::Conv1x1
            | FireflyOp::FishTransConv { .. }
            | FireflyOp::LayerNorm
            | FireflyOp::LinearChannelsLast => 2,
            FireflyOp::LayerScale => 1,
            FireflyOp::SiLu
            | FireflyOp::Gelu
            | FireflyOp::Tanh
            | FireflyOp::Add
            | FireflyOp::ParallelMean
            | FireflyOp::Mask => 0,
            _ => return Err(FishError::InvalidShape("unsupported decoder Graph IR op")),
        };
        if node.checkpoint_keys.len() != expected || node.inputs.is_empty() {
            return Err(FishError::InvalidShape("invalid decoder operation arity"));
        }
        let input_shape = graph.nodes[node.inputs[0]].output;
        let output = node.output;
        extent(&[output.batch, output.channels, output.frames])?;
        if output.batch != input_shape.batch || output.dtype != ScalarType::F32 {
            return Err(FishError::InvalidShape(
                "decoder batch or output dtype mismatch",
            ));
        }
        let shape_of = |index: usize| -> Result<&[usize]> {
            let key = node
                .checkpoint_keys
                .get(index)
                .ok_or(FishError::InvalidShape("missing decoder parameter"))?;
            Ok(graph
                .weights
                .get(key)
                .ok_or(FishError::InvalidShape(
                    "missing declared decoder parameter",
                ))?
                .shape
                .as_slice())
        };
        let unchanged = input_shape == output;
        match node.operation {
            FireflyOp::FsqDecode { groups, quantizers } => {
                if groups == 0
                    || quantizers == 0
                    || input_shape.dtype != ScalarType::I32
                    || input_shape.channels != groups * quantizers
                    || output.channels % groups != 0
                    || output.frames != input_shape.frames
                {
                    return Err(FishError::InvalidShape("invalid decoder FSQ profile"));
                }
                let width = output.channels / groups;
                for group in 0..groups {
                    if shape_of(2 * group)? != [width, FIREFLY_LEVELS.len()]
                        || shape_of(2 * group + 1)? != [width]
                    {
                        return Err(FishError::InvalidShape("invalid decoder FSQ projection"));
                    }
                }
            }
            FireflyOp::FishConv {
                kernel,
                stride,
                dilation,
                groups,
            } => {
                if groups == 0
                    || input_shape.channels % groups != 0
                    || output.channels % groups != 0
                    || shape_of(0)? != [output.channels, input_shape.channels / groups, kernel]
                    || shape_of(1)? != [output.channels]
                    || output.frames
                        != fish_conv_output_len(input_shape.frames, kernel, stride, dilation)?
                {
                    return Err(FishError::InvalidShape(
                        "invalid decoder grouped convolution",
                    ));
                }
            }
            FireflyOp::Conv1x1 => {
                if shape_of(0)? != [output.channels, input_shape.channels, 1]
                    || shape_of(1)? != [output.channels]
                    || output.frames != input_shape.frames
                {
                    return Err(FishError::InvalidShape(
                        "invalid decoder pointwise convolution",
                    ));
                }
            }
            FireflyOp::FishTransConv { kernel, stride } => {
                if shape_of(0)? != [input_shape.channels, output.channels, kernel]
                    || shape_of(1)? != [output.channels]
                    || output.frames
                        != fish_transconv_output_len(input_shape.frames, kernel, stride, 1)?
                {
                    return Err(FishError::InvalidShape(
                        "invalid decoder transposed convolution",
                    ));
                }
            }
            FireflyOp::LayerNorm => {
                if !unchanged
                    || shape_of(0)? != [output.channels]
                    || shape_of(1)? != [output.channels]
                {
                    return Err(FishError::InvalidShape(
                        "invalid decoder layer normalization",
                    ));
                }
            }
            FireflyOp::LinearChannelsLast => {
                if shape_of(0)? != [output.channels, input_shape.channels]
                    || shape_of(1)? != [output.channels]
                    || output.frames != input_shape.frames
                {
                    return Err(FishError::InvalidShape("invalid decoder channel linear"));
                }
            }
            FireflyOp::LayerScale => {
                if !unchanged || shape_of(0)? != [output.channels] {
                    return Err(FishError::InvalidShape("invalid decoder layer scaling"));
                }
            }
            FireflyOp::Add | FireflyOp::ParallelMean => {
                if (node.operation == FireflyOp::Add && node.inputs.len() != 2)
                    || node.inputs.is_empty()
                    || !node
                        .inputs
                        .iter()
                        .all(|&idx| graph.nodes[idx].output == output)
                {
                    return Err(FishError::InvalidShape(
                        "invalid decoder residual or parallel mean",
                    ));
                }
            }
            FireflyOp::SiLu | FireflyOp::Gelu | FireflyOp::Tanh | FireflyOp::Mask => {
                if !unchanged || node.inputs.len() != 1 {
                    return Err(FishError::InvalidShape(
                        "invalid decoder activation or mask",
                    ));
                }
            }
            _ => unreachable!(),
        }
        if node.operation == FireflyOp::Mask {
            FireflyMaskPlan::from_node(graph, id)?;
        }
        for key in &node.checkpoint_keys {
            let shape = &graph.weights[key].shape;
            let expected_len = shape
                .iter()
                .try_fold(1usize, |v, &x| v.checked_mul(x))
                .ok_or(FishError::InvalidShape(
                    "decoder checkpoint extent overflow",
                ))?;
            let values = checkpoint
                .get(key)
                .ok_or(FishError::InvalidShape("missing decoder weight"))?;
            if values.len() != expected_len || values.iter().any(|x| !x.is_finite()) {
                return Err(FishError::InvalidShape("invalid decoder weight"));
            }
        }
    }
    if graph.nodes[graph.output].output.channels != 1 || graph.nodes[graph.output].output.batch != 1
    {
        return Err(FishError::InvalidShape("decoder output is not mono"));
    }
    Ok(())
}
