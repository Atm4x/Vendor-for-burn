use crate::trtx_lowering::LoweringError;
use fish_speech_core::firefly_trtx_pointwise::{PointwiseKind, PointwisePlan};
use safetensors::tensor::Dtype;
use trtx::{ActivationType, Axes, DataType, ElementWiseOperation, NetworkDefinition, Tensor};

fn invalid(s: impl Into<String>) -> LoweringError {
    LoweringError::InvalidBlueprint(s.into())
}
fn trt(e: trtx::Error) -> LoweringError {
    LoweringError::TensorRt(e.to_string())
}
fn affine_tensor<'n>(
    network: &mut NetworkDefinition<'n>,
    bytes: &[u8],
    channels: usize,
    name: &str,
) -> Result<Tensor<'n>, LoweringError> {
    let channels = i64::try_from(channels).map_err(|_| invalid("channel extent overflow"))?;
    network
        .add_constant_owned(
            &[1, channels, 1, 1],
            bytes.to_vec(),
            DataType::kFLOAT,
            Some(name),
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)
}

/// Converts Fish Speech pointwise nodes with native TensorRT-RTX 1.7 operations.
/// Assumes original NCT tensors are represented as [N,C,1,T], so both Python
/// channel-first and ConvNeXt channel-last LayerNorm normalize TRT axis 1.
/// Correct engine execution still requires an SDK/GPU integration test.
pub fn lower_firefly_pointwise<'n>(
    network: &mut NetworkDefinition<'n>,
    inputs: &[&Tensor<'n>],
    plan: &PointwisePlan,
    affine_f32: &[&[u8]],
) -> Result<Tensor<'n>, LoweringError> {
    if inputs.len() != plan.inputs.len() {
        return Err(invalid(format!("{}: input count mismatch", plan.name)));
    }
    plan.validate_affine_bytes(affine_f32)
        .map_err(|e| invalid(format!("{}: {e:?}", plan.name)))?;
    let dims = plan
        .nchw
        .iter()
        .map(|&n| i64::try_from(n).map_err(|_| invalid("dimension overflow")))
        .collect::<Result<Vec<_>, _>>()?;
    for input in inputs {
        if input.data_type(network) != DataType::kFLOAT
            || input.dimensions(network).map_err(trt)? != dims
        {
            return Err(invalid(format!(
                "{}: expected FP32 tensor {dims:?}",
                plan.name
            )));
        }
    }
    let result = match plan.kind {
        PointwiseKind::SiLu => {
            let sigmoid = network
                .add_activation(inputs[0], ActivationType::kSIGMOID)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?;
            network
                .add_elementwise(inputs[0], &sigmoid, ElementWiseOperation::kPROD)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?
        }
        PointwiseKind::GeluErf => network
            .add_activation(inputs[0], ActivationType::kGELU_ERF)
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt)?,
        PointwiseKind::Tanh => network
            .add_activation(inputs[0], ActivationType::kTANH)
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt)?,
        PointwiseKind::Add => network
            .add_elementwise(inputs[0], inputs[1], ElementWiseOperation::kSUM)
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt)?,
        PointwiseKind::ParallelMean => {
            let first = network
                .add_elementwise(inputs[0], inputs[1], ElementWiseOperation::kSUM)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?;
            let second = network
                .add_elementwise(&first, inputs[2], ElementWiseOperation::kSUM)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?;
            let factor = network
                .add_small_constant_copied(
                    &[1, 1, 1, 1],
                    &(1f32 / 3.0).to_le_bytes(),
                    DataType::kFLOAT,
                    None,
                )
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?;
            network
                .add_elementwise(&second, &factor, ElementWiseOperation::kPROD)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?
        }
        PointwiseKind::LayerScale => {
            let gamma = affine_tensor(
                network,
                affine_f32[0],
                plan.nchw[1],
                &plan.checkpoint_keys[0],
            )?;
            network
                .add_elementwise(inputs[0], &gamma, ElementWiseOperation::kPROD)
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?
        }
        PointwiseKind::ChannelLayerNorm => {
            let gamma = affine_tensor(
                network,
                affine_f32[0],
                plan.nchw[1],
                &plan.checkpoint_keys[0],
            )?;
            let beta = affine_tensor(
                network,
                affine_f32[1],
                plan.nchw[1],
                &plan.checkpoint_keys[1],
            )?;
            let mut norm = network
                .add_normalization(inputs[0], &gamma, &beta, Axes::new([1]))
                .map_err(trt)?;
            norm.set_epsilon(network, 1e-6);
            norm.output(network, 0).map_err(trt)?
        }
    };
    let actual = result.dimensions(network).map_err(trt)?;
    if actual != dims {
        return Err(LoweringError::TensorRt(format!(
            "{}: output {:?} != {:?}",
            plan.name, actual, dims
        )));
    }
    Ok(result)
}

pub fn lower_firefly_pointwise_from_checkpoint<'n>(
    network: &mut NetworkDefinition<'n>,
    inputs: &[&Tensor<'n>],
    plan: &PointwisePlan,
    checkpoint: &crate::weights::LoadedWeights,
) -> Result<Tensor<'n>, LoweringError> {
    let mut tensors = Vec::new();
    for key in &plan.checkpoint_keys {
        let tensor = checkpoint
            .tensor(key)
            .map_err(|e| invalid(format!("{key}: {e}")))?;
        if tensor.dtype() != Dtype::F32 || tensor.shape() != [plan.nchw[1]] {
            return Err(invalid(format!(
                "{key}: expected F32 affine [{}]",
                plan.nchw[1]
            )));
        }
        tensors.push(tensor);
    }
    let bytes = tensors.iter().map(|v| v.data()).collect::<Vec<_>>();
    lower_firefly_pointwise(network, inputs, plan, &bytes)
}
