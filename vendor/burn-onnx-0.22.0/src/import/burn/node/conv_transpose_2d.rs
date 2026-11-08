use super::prelude::*;
use burn_pack::Tensor as PackTensor;

fn resolve_padding(
    node: &onnx_ir::node::conv_transpose2d::ConvTranspose2dNode,
) -> super::conv_helpers::TransposePadding {
    super::conv_helpers::transpose_padding(
        &node.inputs[0],
        super::conv_helpers::ConvTransposeGeometry {
            auto_pad: &node.config.auto_pad,
            output_shape: node
                .config
                .output_shape
                .as_ref()
                .map(|shape| shape.as_slice()),
            padding: &node.config.padding,
            padding_out: &node.config.padding_out,
            kernel: &node.config.kernel_size,
            stride: &node.config.stride,
            dilation: &node.config.dilation,
        },
    )
}

impl NodeCodegen for onnx_ir::node::conv_transpose2d::ConvTranspose2dNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if !self.inputs[1].is_static() {
            return None;
        }
        let name = Ident::new(&self.name, Span::call_site());
        let weight_shape = self.inputs[1]
            .ty
            .static_shape_known()
            .expect("ConvTranspose2d: weight tensor shape must be known at codegen time");
        let groups = self.config.groups;
        let channels = [weight_shape[0], weight_shape[1] * groups].to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = groups.to_tokens();
        let resolved = resolve_padding(self);
        let padding = resolved.padding.to_tokens();
        let padding_out = resolved.padding_out.to_tokens();
        let bias = self.inputs.get(2).is_some_and(|bias| !bias.is_optional());

        Some(Field::new(
            self.name.clone(),
            quote! {
                ConvTranspose2d
            },
            quote! {
                let #name = ConvTranspose2dConfig::new(#channels, #kernel_size)
                    .with_stride(#stride)
                    .with_padding(#padding)
                    .with_padding_out(#padding_out)
                    .with_dilation(#dilation)
                    .with_groups(#groups)
                    .with_bias(#bias)
                    .init(device);
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let resolved = resolve_padding(self);
        let crop = resolved.crop_tokens();

        // A runtime weight has no module to live in, so the functional op takes it.
        if !self.inputs[1].is_static() {
            let weight = scope.arg(&self.inputs[1]);
            let bias = super::conv_helpers::optional_input(scope, self.inputs.get(2));
            let stride = self.config.stride.to_tokens();
            let padding = resolved.padding.to_tokens();
            let padding_out = resolved.padding_out.to_tokens();
            let dilation = self.config.dilation.to_tokens();
            let groups = self.config.groups.to_tokens();
            return quote! {
                let #output = burn::tensor::module::conv_transpose2d(
                    #input,
                    #weight,
                    #bias,
                    burn::tensor::ops::ConvTransposeOptions::new(
                        #stride,
                        #padding,
                        #padding_out,
                        #dilation,
                        #groups,
                    ),
                )#crop;
            };
        }
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input)#crop;
        }
    }
    fn register_imports(&self, imports: &mut BurnImports) {
        if !self.inputs[1].is_static() {
            return;
        }
        imports.register("burn::nn::conv::ConvTranspose2d");
        imports.register("burn::nn::conv::ConvTranspose2dConfig");
    }

    fn collect_tensors(&self, field_name: &str) -> Vec<PackTensor> {
        if !self.inputs[1].is_static() {
            return vec![];
        }
        use crate::burn::node_traits::create_deferred_tensor;

        let mut tensors = vec![];

        // Weight tensor (input index 1)
        // ONNX ConvTranspose weight: [in_channels, out_channels/groups, kH, kW]
        // Burn ConvTranspose2d weight: [channels_in, channels_out/groups, kH, kW]
        // These layouts match! No transformation needed.
        if let Some(weight_input) = self.inputs.get(1) {
            let weight_path = format!("{}.weight", field_name);
            if let Some(tensor) = create_deferred_tensor(weight_input, &weight_path) {
                tensors.push(tensor);
            }
        }

        // Bias tensor (input index 2, optional)
        if self.inputs.len() > 2
            && let Some(bias_input) = self.inputs.get(2)
        {
            let bias_path = format!("{}.bias", field_name);
            if let Some(tensor) = create_deferred_tensor(bias_input, &bias_path) {
                tensors.push(tensor);
            }
        }

        tensors
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::conv_transpose2d::{
        ConvTranspose2dConfig, ConvTranspose2dNode, ConvTranspose2dNodeBuilder,
    };
    use onnx_ir::node::padding::AutoPad;

    fn create_conv_transpose_2d_node(name: &str) -> ConvTranspose2dNode {
        let config = ConvTranspose2dConfig::new(
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1],
            [0, 0],
            1,
            AutoPad::NotSet,
            None,
        );

        ConvTranspose2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("weight", vec![3, 64, 3, 3], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_conv_transpose_2d_forward() {
        let node = create_conv_transpose_2d_node("conv_transpose1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.conv_transpose1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_conv_transpose_2d_forward_with_clone() {
        let node = create_conv_transpose_2d_node("conv_transpose1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.conv_transpose1.forward(input.clone());
            output
        }
        ");
    }
    #[test]
    fn test_conv_transpose_2d_runtime_weight() {
        let node = {
            let config = ConvTranspose2dConfig::new(
                [3, 3],
                [1, 1],
                [1, 1],
                [1, 1],
                [0, 0],
                1,
                AutoPad::NotSet,
                None,
            );

            ConvTranspose2dNodeBuilder::new("conv1")
                .input_tensor("input", 4, DType::F32)
                .input_tensor("weight", 4, DType::F32)
                .input_tensor("bias", 1, DType::F32)
                .output_tensor("output", 4, DType::F32)
                .config(config)
                .build()
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<4>,
            weight: Tensor<4>,
            bias: Tensor<1>,
        ) -> Tensor<4> {
            let output = burn::tensor::module::conv_transpose2d(
                input,
                weight,
                Some(bias),
                burn::tensor::ops::ConvTransposeOptions::new([1, 1], [1, 1], [0, 0], [1, 1], 1),
            );
            output
        }
        ");
    }

    /// A `[1, 1, 3, 3]` input with a 3x3 kernel and no bias.
    fn create_derived_pad_node(
        stride: [usize; 2],
        auto_pad: AutoPad,
        output_shape: Option<[usize; 2]>,
        static_weight: bool,
    ) -> ConvTranspose2dNode {
        let config = ConvTranspose2dConfig::new(
            [3, 3],
            stride,
            [1, 1],
            [0, 0],
            [0, 0],
            1,
            auto_pad,
            output_shape,
        );
        let builder = ConvTranspose2dNodeBuilder::new("conv1").input_tensor_shape(
            "input",
            vec![1, 1, 3, 3],
            DType::F32,
        );
        let builder = if static_weight {
            builder.input_static_tensor_shape("weight", vec![1, 2, 3, 3], DType::F32)
        } else {
            builder.input_tensor_shape("weight", vec![1, 2, 3, 3], DType::F32)
        };
        builder
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_conv_transpose_2d_auto_pad_same_upper_crops_end() {
        // Full length 7, SAME wants 6: the odd pad goes at the end, past what burn can trim.
        let node = create_derived_pad_node([2, 2], AutoPad::SameUpper, None, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>, weight: Tensor<4>) -> Tensor<4> {
            let output = burn::tensor::module::conv_transpose2d(
                    input,
                    weight,
                    None,
                    burn::tensor::ops::ConvTransposeOptions::new(
                        [2, 2],
                        [0, 0],
                        [0, 0],
                        [1, 1],
                        1,
                    ),
                )
                .slice(s![.., .., 0..6, 0..6]);
            output
        }
        ");
    }

    #[test]
    fn test_conv_transpose_2d_auto_pad_same_lower() {
        // SAME_LOWER puts the odd pad at the start: burn trims it from both ends and
        // padding_out restores the end.
        let node = create_derived_pad_node([2, 2], AutoPad::SameLower, None, false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>, weight: Tensor<4>) -> Tensor<4> {
            let output = burn::tensor::module::conv_transpose2d(
                input,
                weight,
                None,
                burn::tensor::ops::ConvTransposeOptions::new([2, 2], [1, 1], [1, 1], [1, 1], 1),
            );
            output
        }
        ");
    }

    #[test]
    fn test_conv_transpose_2d_output_shape_field_init() {
        // Full size [9, 7], requested [10, 8]: the output grows by one at the end.
        let node = create_derived_pad_node([3, 2], AutoPad::NotSet, Some([10, 8]), true);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let conv1 = ConvTranspose2dConfig::new([1, 2], [3, 3])
            .with_stride([3, 2])
            .with_padding([0, 0])
            .with_padding_out([1, 1])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        ");
    }

    #[test]
    fn test_conv_transpose_2d_auto_pad_same_upper_module_crop() {
        let node = create_derived_pad_node([2, 2], AutoPad::SameUpper, None, true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.conv1.forward(input).slice(s![.., .., 0..6, 0..6]);
            output
        }
        ");
    }
}
