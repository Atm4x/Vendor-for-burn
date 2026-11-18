#[cfg(feature = "tensorrt-api")]
use std::collections::BTreeMap;

#[cfg(feature = "tensorrt-api")]
use fish_speech_core::graph_ir::ScalarType;
use fish_speech_core::trtx_blueprint::{Binding, TrtBlueprint, TrtPrimitive};
#[cfg(feature = "tensorrt-api")]
use trtx::{
    ActivationType, DataType, ElementWiseOperation, MatrixOperation, NetworkDefinition, Tensor,
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
    let mut unsupported = plan
        .steps
        .iter()
        .filter_map(|step| match step.primitive {
            TrtPrimitive::Deferred { reason, .. } => Some(format!("{}: {reason}", step.name)),
            _ => None,
        })
        .collect::<Vec<_>>();
    unsupported.extend(plan.steps.iter().filter_map(|step| match step.primitive {
        TrtPrimitive::SliceLastSequenceToken => Some(format!(
            "{}: static slice semantics still need TensorRT parity",
            step.name
        )),
        _ => None,
    }));
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
            TrtPrimitive::MatrixMultiplyTransposeRhs => network
                .add_matrix_multiply(
                    args[0],
                    MatrixOperation::kNONE,
                    args[1],
                    MatrixOperation::kTRANSPOSE,
                )
                .map_err(trt)?
                .output(network, 0)
                .map_err(trt)?,
            TrtPrimitive::SliceLastSequenceToken | TrtPrimitive::Deferred { .. } => {
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
