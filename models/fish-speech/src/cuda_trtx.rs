use std::{collections::BTreeMap, sync::Arc};

use cudarc::driver::CudaStream;
use fish_speech_core::{
    error::{FishError, Result},
    trtx_blueprint::{Binding, TrtBlueprint, TrtPrimitive},
};
use trtx::ExecutionContext;

use crate::{
    cuda_stream::CudaBuffer,
    trtx_engine::{DeviceRegion, EngineIoContract},
};

pub fn segment_regions(
    plan: &TrtBlueprint,
    buffers: &BTreeMap<Binding, CudaBuffer>,
    stream: &CudaStream,
) -> Result<(EngineIoContract, BTreeMap<String, DeviceRegion>)> {
    let contract = EngineIoContract::from_blueprint(plan)
        .map_err(|_| FishError::InvalidShape("TensorRT segment contract invalid"))?;
    let mut regions = BTreeMap::new();
    for step in &plan.steps {
        if matches!(
            step.primitive,
            TrtPrimitive::Input | TrtPrimitive::ConstantWeight
        ) {
            let buffer = buffers.get(&step.result).ok_or(FishError::InvalidShape(
                "TensorRT segment input buffer missing",
            ))?;
            regions.insert(
                step.name.clone(),
                DeviceRegion::new(buffer.address(stream) as usize, buffer.bytes())
                    .map_err(|_| FishError::InvalidShape("TensorRT input address invalid"))?,
            );
        }
    }
    for output in &plan.outputs {
        let buffer = buffers.get(output).ok_or(FishError::InvalidShape(
            "TensorRT segment output buffer missing",
        ))?;
        let name = crate::trtx_engine::output_tensor_name(*output);
        regions.insert(
            name,
            DeviceRegion::new(buffer.address(stream) as usize, buffer.bytes())
                .map_err(|_| FishError::InvalidShape("TensorRT output address invalid"))?,
        );
    }
    contract
        .validate_regions(&regions)
        .map_err(|_| FishError::InvalidShape("TensorRT buffer shapes incompatible"))?;
    Ok((contract, regions))
}

/// # Safety
/// The context must correspond to the compiled plan, the buffers must remain alive until the
/// stream completes, and the context must not be used concurrently by another request.
pub unsafe fn enqueue_trt_segment(
    context: &mut ExecutionContext<'_>,
    plan: &TrtBlueprint,
    buffers: &BTreeMap<Binding, CudaBuffer>,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let (contract, regions) = segment_regions(plan, buffers, stream)?;
    unsafe {
        crate::trtx_engine::enqueue_with_regions(
            context,
            &contract,
            &regions,
            stream.cu_stream().cast(),
        )
    }
    .map_err(|_| FishError::InvalidTransition("TensorRT segment enqueue failed"))
}
