//! # ConvTranspose (1D)
//!
//! 1D transposed convolution (deconvolution) operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__ConvTranspose.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic transposed convolution support
//! - **Opset 11**: No changes to ConvTranspose operator itself (broader ONNX updates)
//!
//! ## Implementation Notes
//! - Weight tensor layout: Implementation expects [out_channels, in_channels, kernel_size]
//!   (see FIXME at line 185 regarding ONNX spec clarification)

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::node::padding::{AutoPad, conv_transpose_ints as ints};

use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for ConvTranspose1d operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ConvTranspose1dNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: ConvTranspose1dConfig,
}

/// Configuration for ConvTranspose1d operations extracted from ONNX nodes
#[derive(Debug, Clone, new)]
#[allow(clippy::too_many_arguments)]
pub struct ConvTranspose1dConfig {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Dilation
    pub dilation: usize,
    /// Number of groups
    pub groups: usize,
    /// Symmetric explicit `pads`. Only used when `auto_pad` is `NotSet` and `output_shape` is
    /// `None`.
    pub padding: usize,
    /// Output padding size
    pub padding_out: usize,
    /// ONNX `auto_pad`. `VALID` means zero pads, `SAME_UPPER`/`SAME_LOWER` derive them from the
    /// input size.
    pub auto_pad: AutoPad,
    /// ONNX `output_shape`, spatial dimensions only. When set, the pads derive from it and
    /// `padding` is ignored.
    pub output_shape: Option<usize>,
}

pub(crate) struct Convtranspose1dProcessor;

impl NodeProcessor for Convtranspose1dProcessor {
    type Config = ConvTranspose1dConfig;

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
                kernel: std::slice::from_ref(&config.kernel_size),
                stride: std::slice::from_ref(&config.stride),
                dilation: std::slice::from_ref(&config.dilation),
                padding: std::slice::from_ref(&config.padding),
                output_padding: std::slice::from_ref(&config.padding_out),
                groups: config.groups,
                auto_pad: &config.auto_pad,
                output_shape: config.output_shape.as_ref().map(std::slice::from_ref),
            },
        )
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut kernel_shape = Vec::new();
        let mut stride = vec![1]; // Default stride to 1
        let mut pads = vec![0, 0]; // Default padding to 0
        let mut dilations = vec![1]; // Default dilation to 1
        let mut group: usize = 1; // Default group to 1
        let mut output_padding = vec![0]; // Default output padding to 0
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
                        1,
                    )?)
                }
                _ => {}
            }
        }

        crate::node::padding::check_attr_len("pads", &pads, 2)?;
        if !kernel_shape.is_empty() {
            crate::node::padding::check_attr_len("kernel_shape", &kernel_shape, 1)?;
        }
        crate::node::padding::check_attr_len("dilations", &dilations, 1)?;
        crate::node::padding::check_attr_len("output_padding", &output_padding, 1)?;

        // Check the pads are symmetric. `auto_pad` and `output_shape` take the place of `pads`.
        if auto_pad == AutoPad::NotSet
            && output_shape.is_none()
            && (pads.len() != 2 || pads[0] != pads[1])
        {
            return Err(ProcessError::Custom(format!(
                "Asymmetric padding is not supported for ConvTranspose1d: {pads:?}"
            )));
        }

        let kernel_size = if kernel_shape.is_empty() {
            // https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
            // Spec says if kernel shape not present in attributes it should be inferred from the weight tensor
            let weight_shape = crate::node::padding::known_weight_shape(&node.inputs[1])
                .ok_or_else(|| {
                    ProcessError::Custom(
                    "ConvTranspose1d: kernel_shape is not set and the weight shape is not known"
                        .to_string(),
                )
                })?;
            if weight_shape.len() != 3 {
                return Err(ProcessError::Custom(format!(
                    "expected to infer kernel shape from a weight tensor of rank 3 but got shape {weight_shape:?}"
                )));
            }

            weight_shape[2]
        } else {
            // Was set explicitly via attributes- use that
            kernel_shape[0]
        };

        let config = ConvTranspose1dConfig::new(
            kernel_size,
            stride[0],
            dilations[0],
            group,
            pads[0] as usize,
            output_padding[0],
            auto_pad,
            output_shape.map(|shape| shape[0]),
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;

        Ok(Node::ConvTranspose1d(ConvTranspose1dNode {
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
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    #[allow(clippy::too_many_arguments)]
    fn create_test_node(
        kernel_shape: Vec<i64>,
        stride: Vec<i64>,
        pads: Vec<i64>,
        dilations: Vec<i64>,
        group: i64,
        output_padding: Vec<i64>,
        has_bias: bool,
        auto_pad: Option<&str>,
    ) -> TestNodeBuilder {
        // Create weight tensor data
        let weight_data = vec![0.1; 16];

        let has_kernel_shape = !kernel_shape.is_empty();

        // Start building the node with input and weight
        let mut builder = TestNodeBuilder::new(NodeType::ConvTranspose1d, "test_conv_transpose1d")
            .input_tensor_f32("data", 3, None)
            .input_tensor_f32_data(
                "weight",
                weight_data,
                vec![2, 2, 4], // [in_channels, out_channels/groups, kernel_size] per ONNX spec
            )
            .output_tensor_f32("output", 3, None);

        // Add bias if needed
        if has_bias {
            builder = builder.input_tensor_f32_data("bias", vec![0.1, 0.2], vec![2]);
        }

        // Add attributes
        builder = builder
            .attr_ints("strides", stride)
            .attr_ints("pads", pads)
            .attr_ints("dilations", dilations)
            .attr_int("group", group)
            .attr_ints("output_padding", output_padding);

        if let Some(auto_pad) = auto_pad {
            builder = builder.attr_string("auto_pad", auto_pad);
        }

        if has_kernel_shape {
            builder = builder.attr_ints("kernel_shape", kernel_shape);
        }

        builder
    }

    #[test]
    fn test_conv_transpose1d_config_basic() {
        let node = create_test_node(
            vec![4],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            vec![0],
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.padding, 0);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.padding_out, 0);
        assert_eq!(config.groups, 1);
    }

    #[test]
    fn test_conv_transpose1d_config_with_params() {
        let node = create_test_node(
            vec![4],
            vec![2],
            vec![1, 1],
            vec![2],
            2,
            vec![1],
            true,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 2);
        assert_eq!(config.padding, 1);
        assert_eq!(config.dilation, 2);
        assert_eq!(config.padding_out, 1);
        assert_eq!(config.groups, 2);
    }

    #[test]
    fn test_conv_transpose1d_config_asymmetric_padding() {
        let node = create_test_node(
            vec![4],
            vec![1],
            vec![1, 2],
            vec![1],
            1,
            vec![0],
            false,
            None,
        )
        .build_with_graph_data(16);
        let processor = Convtranspose1dProcessor;
        let result = processor.extract_config(&node, 16);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("Asymmetric padding is not supported"))
        );
    }

    #[test]
    fn test_conv_transpose1d_config_autopad_not_set() {
        let node = create_test_node(
            vec![4],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            vec![0],
            false,
            Some("NOTSET"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.stride, 1);
        assert_eq!(config.padding, 0);
        assert_eq!(config.dilation, 1);
        assert_eq!(config.padding_out, 0);
        assert_eq!(config.groups, 1);
    }

    #[test]
    fn test_conv_transpose1d_config_autopad_same_dynamic_input() {
        let node = create_test_node(
            vec![4],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            vec![0],
            false,
            Some("SAME_UPPER"),
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose1dProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(
            matches!(result, Err(ProcessError::Custom(ref msg)) if msg.contains("dynamic")),
            "{result:?}"
        );
    }

    #[test]
    fn test_conv_transpose1d_config_kernel_shape_not_set() {
        let node = create_test_node(
            vec![],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            vec![0],
            false,
            None,
        )
        .build_with_graph_data(16);
        let mut node = node;
        let processor = Convtranspose1dProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(config.kernel_size, 4); // Inferred via weight tensor shape
    }

    #[test]
    fn test_conv_transpose1d_ignores_unknown_attribute() {
        let mut node = create_test_node(
            vec![4],
            vec![1],
            vec![0, 0],
            vec![1],
            1,
            vec![0],
            false,
            None,
        )
        .attr_int("some_future_attribute", 1)
        .build_with_graph_data(16);
        Convtranspose1dProcessor
            .infer_types(&mut node, 16, &OutputPreferences::new())
            .unwrap();
    }
}
