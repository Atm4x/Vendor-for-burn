use crate::trtx_lowering::LoweringError;
use fish_speech_core::firefly_trtx_mask::FireflyMaskPlan;
use trtx::{DataType, ElementWiseOperation, NetworkDefinition, Tensor};

/// Static, uniform-length 0/1 mask matching original sequence_mask(...).
/// Each model profile fixes the valid length. Varying per-batch lengths requires
/// replacing this static constant with a runtime tensor binding before production.
pub fn lower_firefly_mask<'n>(
    network: &mut NetworkDefinition<'n>,
    input: &Tensor<'n>,
    plan: &FireflyMaskPlan,
) -> Result<Tensor<'n>, LoweringError> {
    let trt = |e: trtx::Error| LoweringError::TensorRt(e.to_string());
    let dims = plan
        .nchw
        .iter()
        .map(|&n| {
            i64::try_from(n)
                .map_err(|_| LoweringError::InvalidBlueprint("mask dimension too large".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if input.data_type(network) != DataType::kFLOAT
        || input.dimensions(network).map_err(trt)? != dims
    {
        return Err(LoweringError::InvalidBlueprint(format!(
            "{}: invalid mask input NCHW",
            plan.name
        )));
    }
    if plan.valid_frames == plan.nchw[3] {
        return network
            .add_identity(input)
            .map_err(trt)?
            .output(network, 0)
            .map_err(trt);
    }
    let mask_bytes = plan
        .mask_f32_bytes()
        .map_err(|e| LoweringError::InvalidBlueprint(format!("{}: {e:?}", plan.name)))?;
    let mask = network
        .add_constant_owned(
            &[1, 1, 1, dims[3]],
            mask_bytes,
            DataType::kFLOAT,
            Some(&plan.name),
        )
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    let output = network
        .add_elementwise(input, &mask, ElementWiseOperation::kPROD)
        .map_err(trt)?
        .output(network, 0)
        .map_err(trt)?;
    if output.dimensions(network).map_err(trt)? != dims {
        return Err(LoweringError::TensorRt(format!(
            "{}: mask output shape mismatch",
            plan.name
        )));
    }
    Ok(output)
}
