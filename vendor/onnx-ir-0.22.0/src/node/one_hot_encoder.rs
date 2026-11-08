//! # OneHotEncoder
//!
//! Replace each input element with an array of ones and zeros, where a single
//! one is placed at the index of the category that was passed in.
//! The output has one more dimension than the input (categories appended as last dim).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx_aionnxml_OneHotEncoder.html>
//!
//! ## Type Constraints
//!
//! - T: tensor(float), tensor(double), tensor(int32), tensor(int64)
//!   (tensor(string) is not supported in Burn)
//!
//! ## Opset Versions
//!
//! - **Opset 1**: Initial version with cats_int64s, cats_strings, and zeros attributes

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, AttributeValue, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for OneHotEncoder operation
#[derive(Debug, Clone, Default, new)]
pub struct OneHotEncoderConfig {
    /// List of integer categories for lookup
    pub cats_int64s: Option<Vec<i64>>,
    /// List of string categories for lookup (not supported in Burn codegen)
    pub cats_strings: Option<Vec<String>>,
    /// If true (1, default) and category is not present, return all zeros;
    /// if false (0), the operator will fail on unknown categories.
    pub zeros: Option<i64>,
}

/// Node representation for OneHotEncoder operation
#[derive(Debug, Clone, new, NodeBuilder)]
pub struct OneHotEncoderNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: OneHotEncoderConfig,
}

pub(crate) struct OneHotEncoderProcessor;

impl NodeProcessor for OneHotEncoderProcessor {
    type Config = OneHotEncoderConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        let config = self.extract_config(node, opset)?;

        // Output Y is always tensor(float) with rank = input_rank + 1.
        // The extra dimension is the number of categories.
        let (input_rank, num_categories) = match &node.inputs[0].ty {
            ArgType::Tensor(t) => {
                match t.dtype {
                    DType::F32 | DType::F64 | DType::I32 | DType::I64 => {}
                    other => {
                        return Err(ProcessError::TypeMismatch {
                            expected: "tensor(float | double | int32 | int64)".to_string(),
                            actual: format!("tensor({other:?})"),
                        });
                    }
                }

                // Determine number of categories from validated config.
                let num_cats = self.get_num_categories(&config);

                (t.rank, num_cats)
            }
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "tensor(float | double | int32 | int64)".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        };

        let output_rank = input_rank + 1;

        // Build static shape: input dims + [num_categories]
        let static_shape = match &node.inputs[0].ty {
            ArgType::Tensor(t) => {
                if let Some(input_shape) = &t.static_shape {
                    let mut shape = input_shape.clone();
                    shape.push(Some(num_categories));
                    Some(shape)
                } else {
                    // We know the last dim is num_categories even if input shape is unknown
                    let mut shape: Vec<Option<usize>> = vec![None; input_rank];
                    shape.push(Some(num_categories));
                    Some(shape)
                }
            }
            _ => None,
        };

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::F32,
            rank: output_rank,
            static_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        let mut cats_int64s: Option<Vec<i64>> = None;
        let mut cats_strings: Option<Vec<String>> = None;
        let mut zeros: Option<i64> = None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "cats_int64s" => {
                    if let AttributeValue::Int64s(ints) = value {
                        cats_int64s = Some(ints.clone());
                    } else {
                        return Err(ProcessError::InvalidAttribute {
                            name: "cats_int64s".to_string(),
                            reason: format!("expected Int64s, got {value:?}"),
                        });
                    }
                }
                "cats_strings" => {
                    if let AttributeValue::Strings(strings) = value {
                        cats_strings = Some(strings.clone());
                    } else {
                        return Err(ProcessError::InvalidAttribute {
                            name: "cats_strings".to_string(),
                            reason: format!("expected Strings, got {value:?}"),
                        });
                    }
                }
                "zeros" => {
                    if let AttributeValue::Int64(i) = value {
                        zeros = Some(*i);
                    } else {
                        return Err(ProcessError::InvalidAttribute {
                            name: "zeros".to_string(),
                            reason: format!("expected Int64, got {value:?}"),
                        });
                    }
                }
                _ => {}
            }
        }

        let config = OneHotEncoderConfig::new(cats_int64s, cats_strings, zeros);
        self.validate_config(&config)?;

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Result<Node, ProcessError> {
        let config = self.extract_config(&builder, opset)?;
        Ok(Node::OneHotEncoder(OneHotEncoderNode::new(
            builder.name,
            builder.inputs,
            builder.outputs,
            config,
        )))
    }
}

impl OneHotEncoderProcessor {
    fn validate_config(&self, config: &OneHotEncoderConfig) -> Result<(), ProcessError> {
        let has_ints = config.cats_int64s.is_some();
        let has_strings = config.cats_strings.is_some();

        if has_ints == has_strings {
            return Err(ProcessError::InvalidAttribute {
                name: "cats_int64s/cats_strings".to_string(),
                reason: "exactly one of cats_int64s or cats_strings must be provided".to_string(),
            });
        }

        if has_strings {
            return Err(ProcessError::Custom(
                "OneHotEncoder with cats_strings is not supported in burn-onnx; \
                 only numeric categories (cats_int64s) are supported"
                    .to_string(),
            ));
        }

        if let Some(ints) = &config.cats_int64s
            && ints.is_empty()
        {
            return Err(ProcessError::InvalidAttribute {
                name: "cats_int64s".to_string(),
                reason: "must not be empty".to_string(),
            });
        }

        let zeros = config.zeros.unwrap_or(1);
        if zeros != 1 {
            return Err(ProcessError::Custom(
                "OneHotEncoder zeros=0 is not supported in burn-onnx because unknown \
                 categories must raise at runtime"
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Get the number of categories from validated config.
    fn get_num_categories(&self, config: &OneHotEncoderConfig) -> usize {
        // validate_config rejects cats_strings and requires a non-empty cats_int64s,
        // so this is the only category list that can reach codegen.
        config
            .cats_int64s
            .as_ref()
            .expect("OneHotEncoder config must contain cats_int64s")
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;
    use crate::processor::OutputPreferences;

    fn make_node_builder(
        input_dtype: DType,
        input_rank: usize,
        input_static_shape: Option<Vec<usize>>,
    ) -> TestNodeBuilder {
        let builder =
            TestNodeBuilder::new(NodeType::OneHotEncoder, "test_ohe").output_default("output");

        match input_dtype {
            DType::F32 => builder.input_tensor_f32("input", input_rank, input_static_shape),
            DType::F64 => builder.input_tensor_f64("input", input_rank, input_static_shape),
            DType::I32 => builder.input_tensor_i32("input", input_rank, input_static_shape),
            DType::I64 => builder.input_tensor_i64("input", input_rank, input_static_shape),
            DType::Bool(_) => builder.input_tensor_bool("input", input_rank, input_static_shape),
            _ => panic!("unsupported test dtype"),
        }
    }

    #[test]
    fn test_onehotencoder_config_int_categories() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2, 3]), None, Some(1));
        assert_eq!(config.cats_int64s.as_ref().unwrap().len(), 4);
        assert!(config.cats_strings.is_none());
        assert_eq!(config.zeros, Some(1));
    }

    #[test]
    fn test_onehotencoder_config_default() {
        let config = OneHotEncoderConfig::default();
        assert!(config.cats_int64s.is_none());
        assert!(config.cats_strings.is_none());
        assert!(config.zeros.is_none());
    }

    #[test]
    fn test_onehotencoder_node_builder() {
        let config = OneHotEncoderConfig::new(Some(vec![0, 1, 2]), None, None);
        let node = OneHotEncoderNode::new("test_ohe".to_string(), vec![], vec![], config);
        assert_eq!(node.name, "test_ohe");
        assert_eq!(node.config.cats_int64s.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_spec_is_opset1_single_input_single_output() {
        let processor = OneHotEncoderProcessor;
        let spec = processor.spec();

        assert_eq!(spec.min_opset, 1);
        assert_eq!(spec.max_opset, None);
        assert!(matches!(spec.inputs, InputSpec::Exact(1)));
        assert!(matches!(spec.outputs, OutputSpec::Exact(1)));
    }

    #[test]
    fn test_extract_config_rejects_both_category_attributes() {
        let node = make_node_builder(DType::I64, 1, Some(vec![4]))
            .attr_ints("cats_int64s", vec![0, 1, 2])
            .attr_strings("cats_strings", vec!["a".to_string(), "b".to_string()])
            .attr_int("zeros", 1)
            .build();

        let processor = OneHotEncoderProcessor;
        let err = processor.extract_config(&node, 1).unwrap_err();

        assert!(matches!(err, ProcessError::InvalidAttribute { .. }));
        assert!(
            err.to_string()
                .contains("exactly one of cats_int64s or cats_strings")
        );
    }

    #[test]
    fn test_extract_config_reads_single_category_attribute() {
        let node = make_node_builder(DType::I64, 1, Some(vec![4]))
            .attr_ints("cats_int64s", vec![0, 1, 2])
            .attr_int("zeros", 1)
            .build();

        let processor = OneHotEncoderProcessor;
        let config = processor.extract_config(&node, 1).unwrap();

        assert_eq!(config.cats_int64s, Some(vec![0, 1, 2]));
        assert!(config.cats_strings.is_none());
        assert_eq!(config.zeros, Some(1));
    }

    #[test]
    fn test_extract_config_rejects_zeros_zero() {
        let node = make_node_builder(DType::I64, 1, Some(vec![4]))
            .attr_ints("cats_int64s", vec![0, 1, 2])
            .attr_int("zeros", 0)
            .build();

        let processor = OneHotEncoderProcessor;
        let err = processor.extract_config(&node, 1).unwrap_err();

        assert!(matches!(err, ProcessError::Custom(_)));
        assert!(err.to_string().contains("zeros=0 is not supported"));
    }

    #[test]
    fn test_extract_config_rejects_empty_cats_int64s() {
        let node = make_node_builder(DType::I64, 1, Some(vec![4]))
            .attr_ints("cats_int64s", vec![])
            .build();

        let processor = OneHotEncoderProcessor;
        let err = processor.extract_config(&node, 1).unwrap_err();

        assert!(matches!(err, ProcessError::InvalidAttribute { .. }));
        assert!(err.to_string().contains("cats_int64s"));
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn test_extract_config_rejects_cats_strings() {
        let node = make_node_builder(DType::F32, 1, Some(vec![2]))
            .attr_strings("cats_strings", vec!["a".to_string(), "b".to_string()])
            .build();

        let processor = OneHotEncoderProcessor;
        let err = processor.extract_config(&node, 1).unwrap_err();

        assert!(matches!(err, ProcessError::Custom(_)));
        assert!(err.to_string().contains("cats_strings is not supported"));
    }

    #[test]
    fn test_infer_types_sets_rank_and_static_shape_from_int_categories() {
        let mut node = make_node_builder(DType::F32, 2, Some(vec![2, 3]))
            .attr_ints("cats_int64s", vec![10, 20, 30, 40])
            .build();

        let processor = OneHotEncoderProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 1, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32);
                assert_eq!(t.rank, 3);
                assert_eq!(t.static_shape, Some(vec![Some(2), Some(3), Some(4)]));
            }
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_infer_types_sets_partial_static_shape_when_input_shape_unknown() {
        let mut node = make_node_builder(DType::I64, 1, None)
            .attr_ints("cats_int64s", vec![0, 1, 2])
            .build();

        let processor = OneHotEncoderProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 1, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 2);
                assert_eq!(t.static_shape, Some(vec![None, Some(3)]));
            }
            other => panic!("expected tensor output, got {other:?}"),
        }
    }

    #[test]
    fn test_infer_types_rejects_unsupported_input_dtype() {
        let mut node =
            make_node_builder(DType::Bool(crate::ir::BoolStore::Native), 1, Some(vec![3]))
                .attr_ints("cats_int64s", vec![0, 1])
                .build();

        let processor = OneHotEncoderProcessor;
        let prefs = OutputPreferences::new();
        let err = processor.infer_types(&mut node, 1, &prefs).unwrap_err();

        assert!(matches!(err, ProcessError::TypeMismatch { .. }));
    }

    #[test]
    fn test_infer_types_requires_categories_attribute() {
        let mut node = make_node_builder(DType::F32, 1, Some(vec![3])).build();

        let processor = OneHotEncoderProcessor;
        let prefs = OutputPreferences::new();
        let err = processor.infer_types(&mut node, 1, &prefs).unwrap_err();

        assert!(matches!(err, ProcessError::InvalidAttribute { .. }));
        assert!(
            err.to_string()
                .contains("exactly one of cats_int64s or cats_strings")
        );
    }
}
