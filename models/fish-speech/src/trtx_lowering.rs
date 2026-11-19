#[cfg(feature = "tensorrt-api")]
use std::collections::BTreeMap;

#[cfg(feature = "tensorrt-api")]
use fish_speech_core::graph_ir::ScalarType;
use fish_speech_core::trtx_blueprint::{Binding, TrtBlueprint, TrtPrimitive};
#[cfg(feature = "tensorrt-api")]
use trtx::{
    ActivationType, Axes, DataType, ElementWiseOperation, MatrixOperation, NetworkDefinition,
    ReduceOperation, Tensor, UnaryOperation,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    InvalidBlueprint(String),
    DeferredOperations(Vec<String>),
    UnsupportedType(String),
    MissingOperand(Binding),
    TensorRt(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweringSummary {
    pub network_inputs: usize,
    pub unbound_weights: usize,
    pub layers: usize,
    pub network_outputs: usize,
}

pub fn preflight_blueprint(plan: &TrtBlueprint) -> Result<LoweringSummary, LoweringError> {
    plan.validate()
        .map_err(|e| LoweringError::InvalidBlueprint(format!("{e:?}")))?;
    let unsupported = plan
        .steps
        .iter()
        .filter_map(|step| match step.primitive {
            TrtPrimitive::Deferred { reason, .. } => Some(format!("{}: {reason}", step.name)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !unsupported.is_empty() {
        return Err(LoweringError::DeferredOperations(unsupported));
    }
    let mut summary = LoweringSummary {
        network_inputs: 0,
        unbound_weights: 0,
        layers: 0,
        network_outputs: plan.outputs.len(),
    };
    for step in &plan.steps {
        match step.primitive {
            TrtPrimitive::Input => summary.network_inputs += 1,
            TrtPrimitive::ConstantWeight => {
                summary.network_inputs += 1;
                summary.unbound_weights += 1;
            }
            _ => summary.layers += 1,
        }
    }
    Ok(summary)
}

/// Defines a real TensorRT network with runtime-provided weight inputs.
/// Requires TensorRT-RTX at execution time; not callable through the mock backend.
#[cfg(feature = "tensorrt-api")]
pub fn lower_blueprint(
    network: &mut NetworkDefinition<'_>,
    plan: &TrtBlueprint,
) -> Result<LoweringSummary, LoweringError> {
    let summary = preflight_blueprint(plan)?;
    let mut tensors: BTreeMap<Binding, Tensor<'_>> = BTreeMap::new();
    for step in &plan.steps {
        let args = step
            .arguments
            .iter()
            .map(|b| tensors.get(b).ok_or(LoweringError::MissingOperand(*b)))
            .collect::<Result<Vec<_>, _>>()?;
        let trt = |e: trtx::Error| LoweringError::TensorRt(e.to_string());
        let value = match step.primitive {
            TrtPrimitive::Input | TrtPrimitive::ConstantWeight => {
                let dtype = match step.output_type.dtype {
                    ScalarType::F32 => DataType::kFLOAT,
                    ScalarType::I32 => DataType::kINT32,
                    ScalarType::BF16 => DataType::kBF16,
                };
                let dims = step
                    .output_type
                    .shape
                    .iter()
                    .map(|&n| n as i64)
                    .collect::<Vec<_>>();
                network.add_input(&step.name, dtype, &dims).map_err(trt)?
            }
            TrtPrimitive::CastToBf16 => network
                .add_cast(args[0], DataType::kBF16)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?,
            TrtPrimitive::Identity => network
                .add_identity(args[0])
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?,
            TrtPrimitive::Sigmoid => network
                .add_activation(args[0], ActivationType::kSIGMOID)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?,
            TrtPrimitive::ElementwiseSum | TrtPrimitive::AddBroadcastBias => network
                .add_elementwise(args[0], args[1], ElementWiseOperation::kSUM)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?,
            TrtPrimitive::ElementwiseProduct => network
                .add_elementwise(args[0], args[1], ElementWiseOperation::kPROD)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?,
            TrtPrimitive::MatrixMultiplyTransposeRhs => {
                let lhs_shape = args[0].dimensions(network).map_err(trt)?;
                let rhs_shape = args[1].dimensions(network).map_err(trt)?;
                let padded_rhs = if rhs_shape.len() < lhs_shape.len() {
                    let mut shape = vec![1; lhs_shape.len() - rhs_shape.len()];
                    shape.extend(rhs_shape);
                    let mut shuffle = network.add_shuffle(args[1]).map_err(trt)?;
                    shuffle.set_reshape_dimensions(network, &shape).map_err(trt)?;
                    Some(shuffle.output(network, 0).map_err(trt)?)
                } else {
                    None
                };
                network.add_matrix_multiply(
                    args[0],
                    MatrixOperation::kNONE,
                    padded_rhs.as_ref().unwrap_or(args[1]),
                    MatrixOperation::kTRANSPOSE,
                )
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?
            }
            TrtPrimitive::RmsNorm { epsilon_bits } => {
                let shape = args[0].dimensions(network).map_err(trt)?;
                let gamma_shape = args[1].dimensions(network).map_err(trt)?;
                if shape.len() != 3
                    || gamma_shape != [shape[2]]
                    || args[0].data_type(network) != DataType::kFLOAT
                    || args[1].data_type(network) != DataType::kFLOAT
                {
                    return Err(LoweringError::InvalidBlueprint(format!(
                        "{}: RMSNorm requires FP32 [B,T,D] and gamma [D]",
                        step.name
                    )));
                }
                let eps = f32::from_bits(epsilon_bits);
                if !eps.is_finite() || eps <= 0.0 {
                    return Err(LoweringError::InvalidBlueprint(
                        "invalid RMSNorm epsilon".into(),
                    ));
                }
                let squared = network
                    .add_elementwise(args[0], args[0], ElementWiseOperation::kPROD)
                    .map_err(trt)?
                    .output(network, 0)
                    .map_err(trt)?;
                let mean = network
                    .add_reduce(&squared, ReduceOperation::kAVG, Axes::new([2]), true)
                    .map_err(trt)?
                    .output(network, 0)
                    .map_err(trt)?;
                let epsilon = network
                    .add_small_constant_copied(
                        &[1, 1, 1],
                        &eps.to_le_bytes(),
                        DataType::kFLOAT,
                        None,
                    )
                    .map_err(trt)?
                    .output(network, 0)
                    .map_err(trt)?;
                let adjusted = network
                    .add_elementwise(&mean, &epsilon, ElementWiseOperation::kSUM)
                    .map_err(trt)?
                    .output(network, 0)
                    .map_err(trt)?;
                let root = network
                    .add_unary(&adjusted, UnaryOperation::kSQRT)
                    .map_err(trt)?
                    .output(network, 0)
                    .map_err(trt)?;
                let normalized = network
                    .add_elementwise(args[0], &root, ElementWiseOperation::kDIV)
                    .map_err(trt)?
                    .output(network, 0)
                    .map_err(trt)?;
                let mut reshape = network.add_shuffle(args[1]).map_err(trt)?;
                reshape
                    .set_reshape_dimensions(network, &[1, 1, shape[2]])
                    .map_err(trt)?;
                let gamma = reshape.output(network, 0).map_err(trt)?;
                network
                    .add_elementwise(&normalized, &gamma, ElementWiseOperation::kPROD)
                    .map_err(trt)?
                    .output(network, 0)
                    .map_err(trt)?
            }
            TrtPrimitive::FishEmbedding {
                num_codebooks,
                codebook_size,
                share_codebook_embeddings,
            } => crate::trtx_ar_ops::lower_fish_embedding(
                network,
                args[0],
                args[1],
                args[2],
                args[3],
                num_codebooks,
                codebook_size,
                share_codebook_embeddings,
            )?,
            TrtPrimitive::EmbeddingGather => {
                crate::trtx_ar_ops::lower_token_embedding(network, args[0], args[1])?
            }
            TrtPrimitive::SplitPackedQkv {
                part,
                q_heads,
                kv_heads,
                head_dim,
            } => crate::trtx_ar_ops::lower_qkv_split(
                network, args[0], part, q_heads, kv_heads, head_dim,
            )?,
            TrtPrimitive::RotaryInterleaved {
                position,
                base_bits,
            } => crate::trtx_ar_ops::lower_rotary_interleaved(
                network,
                args[0],
                position,
                f32::from_bits(base_bits),
            )?,
            TrtPrimitive::GqaAttention {
                position,
                q_heads,
                kv_heads,
            } => crate::trtx_ar_ops::lower_gqa_attention(
                network, args[0], args[1], args[2], position, q_heads, kv_heads,
            )?,
            TrtPrimitive::RotaryCoefficients => {
                crate::trtx_ar_ops::lower_rotary_coefficients(network, args[0], args[1], args[2])?
            }
            TrtPrimitive::GqaAttentionMasked { q_heads, kv_heads } => {
                crate::trtx_ar_ops::lower_gqa_attention_masked(
                    network, args[0], args[1], args[2], args[3], q_heads, kv_heads,
                )?
            }
            TrtPrimitive::KvActivePrefix { active_len } => {
                let dims = args[0].dimensions(network).map_err(trt)?;
                if dims.len() != 4
                    || args[0].data_type(network) != DataType::kBF16
                    || active_len == 0
                    || i64::try_from(active_len).is_err()
                    || (active_len as i64) > dims[2]
                {
                    return Err(LoweringError::InvalidBlueprint(format!(
                        "{}: invalid BF16 active-prefix KV cache",
                        step.name
                    )));
                }
                network
                    .add_slice(
                        args[0],
                        &[0, 0, 0, 0],
                        &[dims[0], dims[1], active_len as i64, dims[3]],
                        &[1, 1, 1, 1],
                    )
                    .map_err(trt)?
                    .output(network, 0)
                    .map_err(trt)?
            }
            TrtPrimitive::SliceLastSequenceToken => {
                let shape = args[0].dimensions(network).map_err(trt)?;
                if shape.len() != 3 || shape[1] < 1 {
                    return Err(LoweringError::InvalidBlueprint(format!(
                        "{}: expected nonempty [B,T,D] sequence",
                        step.name
                    )));
                }
                network
                    .add_slice(
                        args[0],
                        &[0, shape[1] - 1, 0],
                        &[shape[0], 1, shape[2]],
                        &[1, 1, 1],
                    )
                    .map_err(trt)?
                    .output(network, 0)
                    .map_err(trt)?
            }
            TrtPrimitive::Deferred { .. } => {
                unreachable!("unsupported operations rejected before modifying network")
            }
        };
        tensors.insert(step.result, value);
    }
    for &output in &plan.outputs {
        let tensor = tensors
            .get(&output)
            .ok_or(LoweringError::MissingOperand(output))?;
        tensor
            .set_name(network, &crate::trtx_engine::output_tensor_name(output))
            .map_err(|e| LoweringError::TensorRt(e.to_string()))?;
        network.mark_output(tensor);
    }
    Ok(summary)
}

/// Lowers one weighted Firefly convolution from NCT into TensorRT-RTX NCHW (H=1).
/// The caller owns graph orchestration and supplies precisely validated FP32 tensors.
/// No GPU work is claimed until the network is built and executed on an RTX device.
#[cfg(feature = "tensorrt-api")]
pub fn lower_firefly_conv2d<'network>(
    network: &mut NetworkDefinition<'network>,
    input: &Tensor<'network>,
    plan: &fish_speech_core::firefly_trtx_conv::FireflyConv2dPlan,
    kernel_f32: &[u8],
    bias_f32: &[u8],
) -> Result<Tensor<'network>, LoweringError> {
    use fish_speech_core::firefly_trtx_conv::Conv2dKind;
    use trtx::{OwnedConvWeights, OwnedWeights};

    plan.validate_runtime_weights(kernel_f32.len(), bias_f32.len())
        .map_err(|e| LoweringError::InvalidBlueprint(format!("{e:?}")))?;
    let conv_err = |error: trtx::Error| LoweringError::TensorRt(error.to_string());
    let expected = plan
        .input_nchw
        .iter()
        .map(|&dim| {
            i64::try_from(dim)
                .map_err(|_| LoweringError::UnsupportedType("NCHW dimension exceeds i64".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if input.dimensions(network).map_err(conv_err)? != expected {
        return Err(LoweringError::InvalidBlueprint(format!(
            "{}: input NCHW must be {:?}",
            plan.name, expected
        )));
    }
    if input.data_type(network) != DataType::kFLOAT {
        return Err(LoweringError::UnsupportedType(format!(
            "{}: only FP32 convolution weights/activations are currently supported",
            plan.name
        )));
    }
    let convert = |v: usize| {
        i64::try_from(v)
            .map_err(|_| LoweringError::UnsupportedType("convolution parameter exceeds i64".into()))
    };
    let weights = OwnedConvWeights {
        kernel: OwnedWeights {
            shape: plan
                .kernel_nchw
                .iter()
                .map(|&v| convert(v))
                .collect::<Result<Vec<_>, _>>()?,
            data_type: DataType::kFLOAT,
            values: kernel_f32.to_vec(),
            name: Some(plan.kernel_key.clone()),
        },
        bias: Some(OwnedWeights {
            shape: vec![convert(plan.output_nchw[1])?],
            data_type: DataType::kFLOAT,
            values: bias_f32.to_vec(),
            name: Some(plan.bias_key.clone()),
        }),
    };
    let pre = [
        convert(plan.pad_before_hw[0])?,
        convert(plan.pad_before_hw[1])?,
    ];
    let post = [
        convert(plan.pad_after_hw[0])?,
        convert(plan.pad_after_hw[1])?,
    ];
    let padded = if pre != [0, 0] || post != [0, 0] {
        Some(
            network
                .add_padding(input, &pre, &post)
                .map_err(conv_err)?
                .output(network, 0)
                .map_err(conv_err)?,
        )
    } else {
        None
    };
    let conv_input = padded.as_ref().unwrap_or(input);
    let kernel = [
        1,
        i32::try_from(plan.kernel_nchw[3])
            .map_err(|_| LoweringError::UnsupportedType("kernel length exceeds i32".into()))?,
    ];
    let stride = [convert(plan.stride_hw[0])?, convert(plan.stride_hw[1])?];
    let dilation = [convert(plan.dilation_hw[0])?, convert(plan.dilation_hw[1])?];
    let output = match plan.kind {
        Conv2dKind::Convolution => {
            let mut layer = network
                .add_convolution_owned_weights(
                    conv_input,
                    i32::try_from(plan.output_nchw[1]).map_err(|_| {
                        LoweringError::UnsupportedType("output channels exceed i32".into())
                    })?,
                    &kernel,
                    weights,
                )
                .map_err(conv_err)?;
            layer.set_stride(network, &stride);
            layer.set_dilation(network, &dilation);
            layer.set_num_groups(network, convert(plan.groups)?);
            layer.output(network, 0).map_err(conv_err)?
        }
        Conv2dKind::Deconvolution => {
            let mut layer = network
                .add_deconvolution_owned_weights(
                    conv_input,
                    convert(plan.output_nchw[1])?,
                    &[i64::from(kernel[0]), i64::from(kernel[1])],
                    weights,
                )
                .map_err(conv_err)?;
            layer.set_stride(network, &stride).map_err(conv_err)?;
            layer.set_dilation(network, &dilation).map_err(conv_err)?;
            layer
                .set_num_groups(network, convert(plan.groups)?)
                .map_err(conv_err)?;
            let pre_crop = [
                convert(plan.trim_before_hw[0])?,
                convert(plan.trim_before_hw[1])?,
            ];
            let post_crop = [
                convert(plan.trim_after_hw[0])?,
                convert(plan.trim_after_hw[1])?,
            ];
            layer
                .set_pre_padding(network, &pre_crop)
                .map_err(conv_err)?;
            layer
                .set_post_padding(network, &post_crop)
                .map_err(conv_err)?;
            layer.output(network, 0).map_err(conv_err)?
        }
    };
    let output_dims = output.dimensions(network).map_err(conv_err)?;
    let expected_out = plan
        .output_nchw
        .iter()
        .map(|&v| convert(v))
        .collect::<Result<Vec<_>, _>>()?;
    if output_dims != expected_out {
        return Err(LoweringError::TensorRt(format!(
            "{}: TRT convolution output {:?} differs from {:?}",
            plan.name, output_dims, expected_out
        )));
    }
    Ok(output)
}

/// Strictly binds the safe, hash-verified Firefly checkpoint without permitting
/// silently reinterpreted BF16/F16 weights or an incorrect Conv/ConvTranspose layout.
#[cfg(feature = "tensorrt-api")]
pub fn lower_firefly_conv2d_from_checkpoint<'network>(
    network: &mut NetworkDefinition<'network>,
    input: &Tensor<'network>,
    plan: &fish_speech_core::firefly_trtx_conv::FireflyConv2dPlan,
    checkpoint: &crate::weights::LoadedWeights,
) -> Result<Tensor<'network>, LoweringError> {
    use safetensors::tensor::Dtype;
    let to_error = |error: crate::weights::WeightError| {
        LoweringError::InvalidBlueprint(format!("checkpoint tensor: {error}"))
    };
    let kernel = checkpoint.tensor(&plan.kernel_key).map_err(to_error)?;
    let bias = checkpoint.tensor(&plan.bias_key).map_err(to_error)?;
    if kernel.dtype() != Dtype::F32
        || bias.dtype() != Dtype::F32
        || kernel.shape() != plan.source_weight_shape
        || bias.shape() != [plan.output_nchw[1]]
    {
        return Err(LoweringError::InvalidBlueprint(format!(
            "{}: Firefly tensor dtype or original 1D shape mismatch",
            plan.name
        )));
    }
    lower_firefly_conv2d(network, input, plan, kernel.data(), bias.data())
}
