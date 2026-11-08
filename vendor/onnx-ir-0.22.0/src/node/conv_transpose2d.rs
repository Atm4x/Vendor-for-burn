//! # ConvTranspose (2D)
//!
//! 2D transposed convolution (deconvolution) operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__ConvTranspose.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic transposed convolution support
//! - **Opset 11**: No changes to ConvTranspose operator itself (broader ONNX updates)
//!
//! ## Implementation Notes
//! - Weight tensor layout: Implementation expects [out_channels, in_channels, kernel_h, kernel_w]
//! - Padding order: ONNX `pads` format is [H_begin, W_begin, H_end, W_end]

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::node::padding::{AutoPad, conv_transpose_ints as ints};

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for ConvTranspose2d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ConvTranspose2dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ConvTranspose2dConfig,
}

/// Configuration for ConvTranspose2d operations.
#[derive(Debug, Clone, PartialEq, Eq, new)]
#[allow(clippy::too_many_arguments)]
pub struct ConvTranspose2dConfig {
    /// Size of the kernel.
    pub kernel_size: [usize; 2],
    /// Stride of the convolutional kernel.
    pub stride: [usize; 2],
    /// Dilation of the convolutional kernel.
    pub dilation: [usize; 2],
    /// Symmetric explicit `pads`. Only used when `auto_pad` is `NotSet` and `output_shape` is
    /// `None`.
    pub padding: [usize; 2],
    /// Output padding.
    pub padding_out: [usize; 2],
    /// Groups.
    pub groups: usize,
    /// ONNX `auto_pad`. `VALID` means zero pads, `SAME_UPPER`/`SAME_LOWER` derive them from the
    /// input size.
    pub auto_pad: AutoPad,
    /// ONNX `output_shape`, spatial dimensions only. When set, the pads derive from it and
    /// `padding` is ignored.
    pub output_shape: Option<[usize; 2]>,
}

pub(crate) struct Convtranspose2dProcessor;

impl NodeProcessor for Convtranspose2dProcessor {
    type Config = ConvTranspose2dConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Range(2, 3),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // Weight (input[1]) and optional bias (input[2]) go into the module only when
        // both are constants. Otherwise they stay graph values for the functional
        // conv_transpose, which takes both as ordinary inputs.
        crate::processor::lift_all_or_none(node, &[1, 2])
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let config = self.extract_config(node, opset)?;
        crate::node::padding::validate_conv_transpose_pads(
            node,
            &config.auto_pad,
            config.output_shape.is_some(),
        )?;

        crate::node::padding::conv_transpose_output_type(
            node,
            crate::node::padding::ConvTransposeDims {
                kernel: &config.kernel_size,
                stride: &config.stride,
                dilation: &config.dilation,
                padding: &config.padding,
                output_padding: &config.padding_out,
                groups: config.groups,
                auto_pad: &config.auto_pad,
                output_shape: config.output_shape.as_ref().map(|shape| shape.as_slice()),
            },
        )
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut stride = vec![1, 1]; // Default stride to 1
        let mut pads = vec![0, 0, 0, 0]; // Default padding to 0
        let mut dilations = vec![1, 1]; // Default dilation to 1
        let mut group: usize = 1; // Default group to 1
        let mut output_padding = vec![0, 0]; // Default output padding to 0
        let mut auto_pad = AutoPad::NotSet;
        let mut output_shape = None;

        // Extract attributes
        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "kernel_shape" => {
                    kernel_shape = ints("kernel_shape", &value.clone().into_i64s()?, 1)?
                }
                "strides" => stride = ints("strides", &value.clone().into_i64s()?, 1)?,
                "pads" => pads = value.clone().into_i64s()?,
                "dilations" => dilations = ints("dilations", &value.clone().into_i64s()?, 1)?,
                "group" => group = ints("group", &[value.clone().into_i64()?], 1)?[0],
                "output_padding" => {
                    output_padding = ints("output_padding", &value.clone().into_i64s()?, 0)?
                }
                "auto_pad" => auto_pad = AutoPad::parse(&value.clone().into_string()?)?,
                "output_shape" => {
                    output_shape = Some(crate::node::padding::conv_transpose_output_shape(
                        &value.clone().into_i64s()?,
                        2,
                    )?)
                }
                _ => {}
            }
        }

        crate::node::padding::check_attr_len("pads", &pads, 4)?;
        if !kernel_shape.is_empty() {
            crate::node::padding::check_attr_len("kernel_shape", &kernel_shape, 2)?;
        }
        crate::node::padding::check_attr_len("dilations", &dilations, 2)?;
        crate::node::padding::check_attr_len("output_padding", &output_padding, 2)?;

        // ONNX pads format: [H_begin, W_begin, H_end, W_end] = [top, left, bottom, right].
        // `auto_pad` and `output_shape` take the place of `pads`.
        if auto_pad == AutoPad::NotSet && output_shape.is_none() {
            let [top, left, bottom, right] = [pads[0], pads[1], pads[2], pads[3]];
            if left < 0 || top < 0 || right < 0 || bottom < 0 {
                return Err(ProcessError::Custom(
                    "Negative pad values are not supported".to_string(),
                ));
            } else if (left != right) || (top != bottom) {
                return Err(ProcessError::Custom(
                    "Asymmetric padding is not supported".to_string(),
                ));
            }
        }

        let kernel_size = if kernel_shape.is_empty() {
            let weight_shape = crate::node::padding::known_weight_shape(&node.inputs[1])
                .ok_or_else(|| {
                    ProcessError::Custom(
                    "ConvTranspose2d: kernel_shape is not set and the weight shape is not known"
                        .to_string(),
                )
                })?;

            if weight_shape.len() != 4 {
                return Err(ProcessError::Custom(format!(
                    "expected to infer kernel shape from a weight tensor of rank 4 but got shape {weight_shape:?}"
                )));
            }

            [weight_shape[2], weight_shape[3]]
        } else {
            [kernel_shape[0], kernel_shape[1]]
        };

        let config = ConvTranspose2dConfig::new(
            kernel_size,
            [stride[0], stride[1]],
            [dilations[0], dilations[1]],
            [pads[0] as usize, pads[1] as usize],
            [output_padding[0], output_padding[1]],
            group,
            auto_pad,
            output_shape.map(|shape| [shape[0], shape[1]]),
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(Node::ConvTranspose2d(ConvTranspose2dNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    #[allow(clippy::too_many_arguments)]
    fn create_test_node(
        kernel_shape: Vec<i64>,
        strides: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        output_padding: Vec<i64>,
        group: i64,
        has_bias: bool,
        auto_pad: Option<&str>,
    ) -> TestNodeBuilder {
        // Create weight tensor data
        // ONNX ConvTranspose weight: [in_channels, out_channels/groups, k_h, k_w]
        let weight_shape = vec![2, 4, 2, 2]; // [C=in_channels, M/groups=out_channels/groups, k_h, k_w]
        let weight_data = vec![0.0; 32]; // 2*4*2*2 = 32

        let has_kernel_shape = !kernel_shape.is_empty();

        // Start building the node with input and weight
        let mut builder = TestNodeBuilder::new(NodeType::ConvTranspose2d, "test_convtranspose2d")
            .input_tensor_f32("data", 4, None)
            .input_tensor_f32_data("weight", weight_data, weight_shape)
            .output_tensor_f32("output", 4, None);

        // Add bias if needed
        if has_bias {
            builder = builder.input_tensor_f32("bias", 1, None);
        }

        if has_kernel_shape {
            builder = builder.attr_ints("kernel_shape", kernel_shape);
        }

        // Add attributes
        builder = builder
            .attr_ints("strides", strides)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_ints("output_padding", output_padding)
            .attr_int("group", group);

        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }

        builder
    }

    #[test]
    fn test_conv_transpose2d_config_basic() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![0, 0],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [2, 2]);
        assert_eq!(config.stride, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert_eq!(config.padding, [0, 0]);
        assert_eq!(config.padding_out, [0, 0]);
        assert_eq!(config.groups, 1);
    }

    #[test]
    fn test_conv_transpose2d_config_with_padding() {
        let node = create_test_node(
            vec![3, 3],
            vec![2, 2],
            vec![1, 1, 1, 1],
            vec![1, 1],
            vec![0, 0],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.padding, [1, 1]);
        assert_eq!(config.stride, [2, 2]);
    }

    #[test]
    fn test_conv_transpose2d_config_with_output_padding() {
        let node = create_test_node(
            vec![2, 2],
            vec![2, 2],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![1, 1],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.padding_out, [1, 1]);
    }

    #[test]
    fn test_conv_transpose2d_config_with_groups() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![0, 0],
            2,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.groups, 2);
    }

    #[test]
    fn test_conv_transpose2d_config_with_asymmetric_padding() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![1, 1, 2, 2],
            vec![1, 1],
            vec![0, 0],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let processor = Convtranspose2dProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("Asymmetric padding is not supported"))
        );
    }

    #[test]
    fn test_conv_transpose2d_config_autopad_not_set() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![0, 0],
            1,
            false,
            Some("NOTSET"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [2, 2]);
        assert_eq!(config.stride, [1, 1]);
        assert_eq!(config.dilation, [1, 1]);
        assert_eq!(config.padding, [0, 0]);
        assert_eq!(config.padding_out, [0, 0]);
        assert_eq!(config.groups, 1);
    }

    #[test]
    fn test_conv_transpose2d_config_autopad_same_dynamic_input() {
        let node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![0, 0],
            1,
            false,
            Some("SAME_UPPER"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose2dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("dynamic")),
            "{result:?}"
        );
    }

    /// A node over a statically shaped `[1, 2, 3, 3]` input with a `[2, 4, 3, 3]` weight.
    fn create_static_node() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::ConvTranspose2d, "test_convtranspose2d")
            .input_tensor_f32("data", 4, Some(vec![1, 2, 3, 3]))
            .input_tensor_f32_data("weight", vec![0.0; 72], vec![2, 4, 3, 3])
            .output_tensor_f32("output", 4, None)
            .attr_ints("strides", vec![2, 2])
    }

    #[test]
    fn test_conv_transpose2d_config_autopad_same_static_input() {
        let mut node = create_static_node()
            .attr_string("auto_pad", "SAME_UPPER")
            .build_with_graph_data(16);
        let processor = Convtranspose2dProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.auto_pad, AutoPad::SameUpper);
        assert_eq!(config.output_shape, None);
    }

    #[test]
    fn test_conv_transpose2d_config_output_shape_ignores_pads() {
        let mut node = create_static_node()
            .attr_ints("output_shape", vec![7, 8])
            .attr_ints("pads", vec![0, 0, 1, 1])
            .build_with_graph_data(16);
        let processor = Convtranspose2dProcessor;
        processor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
        let config = processor.extract_config(&node, 16).unwrap();

        assert_eq!(config.output_shape, Some([7, 8]));
        assert_eq!(config.auto_pad, AutoPad::NotSet);
    }

    #[test]
    fn test_conv_transpose2d_config_output_shape_wrong_rank() {
        let node = create_static_node()
            .attr_ints("output_shape", vec![1, 4, 7, 8])
            .build_with_graph_data(16);
        let result = Convtranspose2dProcessor.extract_config(&node, 16);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidAttribute { ref name, .. }) if name == "output_shape"
        ));
    }

    #[test]
    fn test_conv_transpose2d_output_shape_dynamic_input() {
        let mut node = create_test_node(
            vec![2, 2],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![0, 0],
            1,
            false,
            None,
        )
        .attr_ints("output_shape", vec![4, 4])
        .build_with_graph_data(16);
        let result = Convtranspose2dProcessor.infer_types(&mut node, 16, &OutputPreferences::new());
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("output_shape")),
            "{result:?}"
        );
    }

    fn inferred_shape(builder: TestNodeBuilder) -> Vec<Option<usize>> {
        let mut node = builder.build_with_graph_data(16);
        Convtranspose2dProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => tensor.static_shape.clone().unwrap(),
            other => panic!("expected a tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_conv_transpose2d_output_shape_inference() {
        let expected = |h, w| vec![Some(1), Some(4), Some(h), Some(w)];

        // SAME: input * stride.
        let same = create_static_node().attr_string("auto_pad", "SAME_LOWER");
        assert_eq!(inferred_shape(same), expected(6, 6));

        // output_shape wins over everything else.
        let fixed = create_static_node()
            .attr_string("auto_pad", "SAME_UPPER")
            .attr_ints("output_shape", vec![7, 8]);
        assert_eq!(inferred_shape(fixed), expected(7, 8));

        // Explicit: 2 * (3 - 1) + 1 + (3 - 1) + 1 - 2.
        let explicit = create_static_node()
            .attr_ints("pads", vec![1, 1, 1, 1])
            .attr_ints("output_padding", vec![1, 1]);
        assert_eq!(inferred_shape(explicit), expected(6, 6));

        // VALID ignores pads.
        let valid = create_static_node()
            .attr_string("auto_pad", "VALID")
            .attr_ints("pads", vec![1, 1, 1, 1]);
        assert_eq!(inferred_shape(valid), expected(7, 7));
    }

    #[test]
    fn test_conv_transpose2d_negative_output_padding() {
        let node = create_static_node()
            .attr_ints("output_padding", vec![-1, 0])
            .build_with_graph_data(16);
        let result = Convtranspose2dProcessor.extract_config(&node, 16);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidAttribute { ref name, .. }) if name == "output_padding"
        ));
    }

    #[test]
    fn test_conv_transpose2d_autopad_same_empty_input() {
        let mut node = TestNodeBuilder::new(NodeType::ConvTranspose2d, "test_convtranspose2d")
            .input_tensor_f32("data", 4, Some(vec![1, 2, 0, 3]))
            .input_tensor_f32_data("weight", vec![0.0; 72], vec![2, 4, 3, 3])
            .output_tensor_f32("output", 4, None)
            .attr_string("auto_pad", "SAME_UPPER")
            .build_with_graph_data(16);
        let result = Convtranspose2dProcessor.infer_types(&mut node, 16, &OutputPreferences::new());
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("non-empty")),
            "{result:?}"
        );
    }

    #[test]
    fn test_conv_transpose2d_ignores_unknown_attribute() {
        let mut node = create_static_node()
            .attr_int("some_future_attribute", 1)
            .build_with_graph_data(16);
        Convtranspose2dProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
    }

    #[test]
    fn test_conv_transpose2d_config_kernel_shape_not_set() {
        let node = create_test_node(
            vec![],
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![0, 0],
            1,
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose2dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, [2, 2]); // Inferred via weight tensor shape
    }
}
