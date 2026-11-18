use crate::audio::{fish_conv_padding, fish_transconv_output_len};
use crate::error::{FishError, Result};
use crate::firefly_ir::{FireflyGraph, FireflyOp};
use crate::graph_ir::ScalarType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv2dKind {
    Convolution,
    Deconvolution,
}

/// NCT -> NCHW with height=1. A TensorRT 2D convolution represents Fish's 1D operation;
/// the last dimension is time. PyTorch OIK/IOK weights become OI1K/IO1K without reordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FireflyConv2dPlan {
    pub name: String,
    pub kind: Conv2dKind,
    pub input_nchw: [usize; 4],
    pub output_nchw: [usize; 4],
    pub kernel_nchw: [usize; 4],
    pub source_weight_shape: Vec<usize>,
    pub kernel_key: String,
    pub bias_key: String,
    pub stride_hw: [usize; 2],
    pub dilation_hw: [usize; 2],
    pub groups: usize,
    pub pad_before_hw: [usize; 2],
    pub pad_after_hw: [usize; 2],
    pub trim_before_hw: [usize; 2],
    pub trim_after_hw: [usize; 2],
}

impl FireflyConv2dPlan {
    pub fn from_node(graph: &FireflyGraph, index: usize) -> Result<Option<Self>> {
        graph.validate()?;
        let node = graph
            .nodes
            .get(index)
            .ok_or(FishError::InvalidShape("Firefly node outside graph"))?;
        let (kind, kernel, stride, dilation, groups) = match node.operation {
            FireflyOp::FishConv {
                kernel,
                stride,
                dilation,
                groups,
            } => (Conv2dKind::Convolution, kernel, stride, dilation, groups),
            FireflyOp::Conv1x1 | FireflyOp::LinearChannelsLast => {
                (Conv2dKind::Convolution, 1, 1, 1, 1)
            }
            FireflyOp::FishTransConv { kernel, stride } => {
                (Conv2dKind::Deconvolution, kernel, stride, 1, 1)
            }
            _ => return Ok(None),
        };
        if node.inputs.len() != 1 || node.checkpoint_keys.len() != 2 {
            return Err(FishError::InvalidShape("invalid Firefly conv DAG arity"));
        }
        let input = graph.nodes[node.inputs[0]].output;
        let output = node.output;
        if input.dtype != ScalarType::F32 || output.dtype != ScalarType::F32 {
            return Err(FishError::InvalidShape(
                "Firefly conv expects FP32 graph nodes",
            ));
        }
        if kernel == 0 || stride == 0 || dilation == 0 || groups == 0 {
            return Err(FishError::InvalidShape("invalid convolution geometry"));
        }
        let expected_weights = match kind {
            Conv2dKind::Convolution => {
                if input.channels % groups != 0 || output.channels % groups != 0 {
                    return Err(FishError::InvalidShape("invalid convolution groups"));
                }
                [output.channels, input.channels / groups, kernel]
            }
            Conv2dKind::Deconvolution => [input.channels, output.channels, kernel],
        };
        let source_weight_shape = if matches!(node.operation, FireflyOp::LinearChannelsLast) {
            vec![expected_weights[0], expected_weights[1]]
        } else {
            expected_weights.to_vec()
        };
        let weight = graph
            .weights
            .get(&node.checkpoint_keys[0])
            .ok_or(FishError::InvalidShape("missing convolution kernel"))?;
        let bias = graph
            .weights
            .get(&node.checkpoint_keys[1])
            .ok_or(FishError::InvalidShape("missing convolution bias"))?;
        if weight.shape != source_weight_shape || bias.shape != [output.channels] {
            return Err(FishError::InvalidShape(
                "Firefly convolution weights shape mismatch",
            ));
        }
        let (pad_left, pad_right, trim_right) = match kind {
            Conv2dKind::Convolution => {
                let (left, right) = fish_conv_padding(input.frames, kernel, stride, dilation)?;
                (left, right, 0)
            }
            Conv2dKind::Deconvolution => {
                if kernel < stride {
                    return Err(FishError::InvalidShape("invalid FishTransConv crop"));
                }
                let expected = fish_transconv_output_len(input.frames, kernel, stride, dilation)?;
                if output.frames != expected {
                    return Err(FishError::InvalidShape("transconv output length mismatch"));
                }
                (0, 0, kernel - stride)
            }
        };
        if kind == Conv2dKind::Convolution {
            let effective = (kernel - 1)
                .checked_mul(dilation)
                .and_then(|n| n.checked_add(1))
                .ok_or(FishError::InvalidShape(
                    "convolution effective kernel overflow",
                ))?;
            let extent = input
                .frames
                .checked_add(pad_left)
                .and_then(|n| n.checked_add(pad_right))
                .ok_or(FishError::InvalidShape(
                    "convolution padded extent overflow",
                ))?;
            if extent < effective || (extent - effective) / stride + 1 != output.frames {
                return Err(FishError::InvalidShape(
                    "convolution output length mismatch",
                ));
            }
        }
        Ok(Some(Self {
            name: node.name.clone(),
            kind,
            input_nchw: [input.batch, input.channels, 1, input.frames],
            output_nchw: [output.batch, output.channels, 1, output.frames],
            kernel_nchw: [expected_weights[0], expected_weights[1], 1, kernel],
            source_weight_shape,
            kernel_key: node.checkpoint_keys[0].clone(),
            bias_key: node.checkpoint_keys[1].clone(),
            stride_hw: [1, stride],
            dilation_hw: [1, dilation],
            groups,
            pad_before_hw: [0, pad_left],
            pad_after_hw: [0, pad_right],
            trim_before_hw: [0, 0],
            trim_after_hw: [0, trim_right],
        }))
    }

    pub fn validate_runtime_weights(&self, kernel_bytes: usize, bias_bytes: usize) -> Result<()> {
        let kernel_elements = self
            .kernel_nchw
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or(FishError::InvalidShape("kernel extent overflow"))?;
        let required_kernel = kernel_elements
            .checked_mul(4)
            .ok_or(FishError::InvalidShape("kernel bytes overflow"))?;
        let required_bias = self.output_nchw[1]
            .checked_mul(4)
            .ok_or(FishError::InvalidShape("bias bytes overflow"))?;
        if kernel_bytes != required_kernel || bias_bytes != required_bias {
            return Err(FishError::InvalidShape(
                "TensorRT convolution FP32 bytes mismatch",
            ));
        }
        Ok(())
    }
}

pub fn plan_firefly_convolutions(graph: &FireflyGraph) -> Result<Vec<FireflyConv2dPlan>> {
    graph.validate()?;
    graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            matches!(
                node.operation,
                FireflyOp::FishConv { .. }
                    | FireflyOp::Conv1x1
                    | FireflyOp::LinearChannelsLast
                    | FireflyOp::FishTransConv { .. }
            )
            .then(|| {
                FireflyConv2dPlan::from_node(graph, index).and_then(|v| {
                    v.ok_or(FishError::InvalidShape("missing Firefly convolution plan"))
                })
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::firefly_ir::FireflyGraph;

    #[test]
    fn reference_encoder_conv2d_plans_cover_direct_depthwise_and_downsample() {
        let graph = FireflyGraph::reference_encode(1, 12_345).unwrap();
        let plans = plan_firefly_convolutions(&graph).unwrap();
        assert!(plans.len() >= 25);
        let stem = plans
            .iter()
            .find(|v| v.name == "backbone.downsample_layers.0.0")
            .unwrap();
        assert_eq!(stem.kind, Conv2dKind::Convolution);
        assert_eq!(stem.kernel_nchw, [128, 160, 1, 7]);
        assert_eq!((stem.pad_before_hw, stem.pad_after_hw), ([0, 6], [0, 0]));
        let down = plans
            .iter()
            .find(|v| v.name == "quantizer.downsample.0.0")
            .unwrap();
        assert_eq!((down.stride_hw, down.pad_before_hw), ([1, 2], [0, 0]));
        let depthwise = plans
            .iter()
            .find(|v| v.name == "backbone.stages.0.0.dwconv")
            .unwrap();
        assert_eq!(depthwise.groups, 128);
        assert_eq!(depthwise.kernel_nchw, [128, 1, 1, 7]);
        let pointwise = plans
            .iter()
            .find(|v| v.name == "backbone.downsample_layers.1.1")
            .unwrap();
        assert_eq!(pointwise.kernel_nchw, [256, 128, 1, 1]);
        assert_eq!(pointwise.source_weight_shape, vec![256, 128, 1]);
        let linear = plans
            .iter()
            .find(|v| v.name == "backbone.stages.0.0.pwconv1")
            .unwrap();
        assert_eq!(linear.kernel_nchw, [512, 128, 1, 1]);
        assert_eq!(linear.source_weight_shape, vec![512, 128]);
    }

    #[test]
    fn waveform_decode_transpose_plans_have_exact_right_trim() {
        for frames in [1, 2, 3, 7, 19] {
            let graph = FireflyGraph::waveform_decode(1, frames, frames).unwrap();
            let plans = plan_firefly_convolutions(&graph).unwrap();
            let head = plans.iter().find(|v| v.name == "head.ups.0").unwrap();
            assert_eq!(head.kernel_nchw, [512, 256, 1, 16]);
            assert_eq!(head.stride_hw, [1, 8]);
            assert_eq!(head.trim_after_hw, [0, 8]);
            assert_eq!(head.trim_before_hw, [0, 0]);
            assert_eq!(head.output_nchw[3], frames * 4 * 8);
            head.validate_runtime_weights(512 * 256 * 16 * 4, 256 * 4)
                .unwrap();
            assert!(head.validate_runtime_weights(0, 256 * 4).is_err());
            let up = plans
                .iter()
                .find(|v| v.name == "quantizer.upsample.0.0")
                .unwrap();
            assert_eq!(up.trim_after_hw, [0, 0]);
            assert_eq!(up.stride_hw, [1, 2]);
        }
    }

    #[test]
    fn only_convolution_nodes_are_included_in_plan() {
        let graph = FireflyGraph::waveform_decode(2, 6, 4).unwrap();
        let plans = plan_firefly_convolutions(&graph).unwrap();
        assert!(plans.len() > 70);
        assert!(plans.len() < graph.nodes.len());
        assert_eq!(
            plans
                .iter()
                .filter(|p| p.kind == Conv2dKind::Deconvolution)
                .count(),
            7
        );
        assert!(plans
            .iter()
            .all(|p| p.input_nchw[0] == 2 && p.output_nchw[0] == 2));
    }
}
