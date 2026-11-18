use std::{collections::BTreeMap, sync::Arc};

use cudarc::driver::CudaStream;
use fish_speech_core::{
    ar_stream_plan::ArStreamPlan,
    error::{FishError, Result},
    trtx_blueprint::{Binding, TrtBlueprint, TrtPrimitive},
};
use trtx::ExecutionContext;

use crate::{
    ar_weight_bindings::required_ar_weights,
    cuda_stream::{CudaArDevice, CudaBuffer},
    trtx_engine::{DeviceRegion, EngineIoContract},
    weights::LoadedWeights,
};

pub fn upload_ar_weights<F>(
    device: &mut CudaArDevice<F>,
    plan: &ArStreamPlan,
    checkpoint: &LoadedWeights,
) -> Result<BTreeMap<String, CudaBuffer>> {
    let required = required_ar_weights(plan)
        .map_err(|_| FishError::InvalidShape("TensorRT AR weight plan invalid"))?;
    let mut uploaded = BTreeMap::new();
    let upload = (|| {
        for (name, ty) in &required {
            let bytes = checkpoint
                .semantic_f32_upload_bytes(name, ty)
                .map_err(|_| FishError::InvalidShape("TensorRT checkpoint weight incompatible"))?;
            uploaded.insert(name.clone(), device.upload_tensor_bytes(ty, &bytes)?);
        }
        Ok(())
    })();
    let sync = device
        .stream()
        .synchronize()
        .map_err(|_| FishError::InvalidTransition("CUDA weight upload synchronization failed"));
    sync?;
    upload?;
    Ok(uploaded)
}

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
