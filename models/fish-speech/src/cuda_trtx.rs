use std::{
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::Arc,
};

use cudarc::driver::{CudaContext, CudaStream};
use fish_speech_core::{
    ar_stream_executor::ArStreamExecutor,
    ar_stream_plan::{ArStreamPhase, ArStreamPlan},
    error::{FishError, Result},
    graph_ir::TensorType,
    trtx_blueprint::{Binding, TrtBlueprint, TrtPrimitive},
};
use trtx::ExecutionContext;

use crate::{
    ar_input_contract::{ArInputContract, collect_ar_weight_types},
    cuda_stream::{CudaArDevice, CudaBuffer},
    device_memory::DeviceMemoryBudget,
    trtx_cache::{EngineIdentity, EngineRegistry},
    trtx_engine::{DeviceRegion, EngineError, EngineIoContract},
    weights::LoadedWeights,
};

pub fn upload_plan_set_weights<F>(
    device: &mut CudaArDevice<F>,
    plans: &[&ArStreamPlan],
    checkpoint: &LoadedWeights,
) -> Result<BTreeMap<String, CudaBuffer>> {
    let required = collect_ar_weight_types(plans)
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

pub fn upload_ar_weights<F>(
    device: &mut CudaArDevice<F>,
    plan: &ArStreamPlan,
    checkpoint: &LoadedWeights,
) -> Result<BTreeMap<String, CudaBuffer>> {
    upload_plan_set_weights(device, &[plan], checkpoint)
}

pub struct WeightedCudaArSession<F>
where
    F: FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()>,
{
    executor: ArStreamExecutor<CudaArDevice<F>>,
    weights: BTreeMap<String, CudaBuffer>,
}

impl<F> WeightedCudaArSession<F>
where
    F: FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()>,
{
    pub fn open(
        mut device: CudaArDevice<F>,
        plans: &[&ArStreamPlan],
        checkpoint: &LoadedWeights,
    ) -> Result<Self> {
        let weights = upload_plan_set_weights(&mut device, plans, checkpoint)?;
        Ok(Self {
            executor: ArStreamExecutor::new(device),
            weights,
        })
    }

    pub fn upload_runtime_input(&mut self, ty: &TensorType, bytes: &[u8]) -> Result<CudaBuffer> {
        self.executor.device_mut().upload_tensor_bytes(ty, bytes)
    }

    pub fn enqueue(
        &mut self,
        plan: &ArStreamPlan,
        runtime_inputs: &BTreeMap<String, CudaBuffer>,
    ) -> Result<BTreeMap<Binding, CudaBuffer>> {
        let contract = ArInputContract::from_plan(plan)
            .map_err(|_| FishError::InvalidShape("TensorRT AR input contract invalid"))?;
        let external = contract
            .bind(&self.weights, runtime_inputs, CudaBuffer::bytes)
            .map_err(|_| {
                FishError::InvalidShape("TensorRT AR device inputs missing or undersized")
            })?;
        self.executor.enqueue(plan, &external)
    }

    pub fn retire_completed(&mut self) -> Result<()> {
        self.executor.retire_completed()
    }

    pub fn read_output_bytes(&mut self, output: &CudaBuffer) -> Result<Vec<u8>> {
        self.executor.retire_completed()?;
        self.executor.device_mut().download_tensor_bytes(output)
    }

    pub fn reset_fast_prefix(&mut self) -> Result<()> {
        self.executor.reset_fast_prefix()
    }

    pub fn reset_request(&mut self) -> Result<()> {
        self.executor.reset_request()
    }

    pub fn inflight_submissions(&self) -> usize {
        self.executor.inflight_submissions()
    }

    pub fn cache_active_len(&self, name: &str) -> Option<usize> {
        self.executor.cache_active_len(name)
    }

    pub fn device_mut(&mut self) -> &mut CudaArDevice<F> {
        self.executor.device_mut()
    }
}

pub struct CachedArDispatchConfig<'a> {
    pub model_sha256: &'a str,
    pub gpu_identity: &'a str,
    pub tensorrt_version: &'a str,
    pub driver_version: &'a str,
    pub precision: &'a str,
    pub shape_profile: &'a str,
    pub build_flags: &'a str,
    pub workspace_bytes: usize,
}

#[deprecated(
    note = "synchronizes every segment; use with_resident_cuda_ar_session for prefill/decode"
)]
pub fn cached_ar_dispatch<'a>(
    registry: &'a EngineRegistry,
    logger: &'a trtx::Logger,
    config: CachedArDispatchConfig<'a>,
) -> impl FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()> + 'a
{
    move |plan, buffers, stream| {
        let identity = EngineIdentity::for_blueprint(
            plan,
            config.model_sha256,
            config.gpu_identity,
            config.tensorrt_version,
            config.driver_version,
            config.precision,
            config.shape_profile,
            config.build_flags,
        )
        .map_err(|_| FishError::InvalidShape("TensorRT engine cache identity invalid"))?;
        registry
            .with_cached_session(
                logger,
                plan,
                &identity,
                config.workspace_bytes,
                |context| {
                    unsafe { enqueue_trt_segment(context, plan, buffers, stream) }
                        .map_err(|e| EngineError::TensorRt(e.to_string()))
                },
                || {
                    stream
                        .synchronize()
                        .map_err(|e| EngineError::TensorRt(e.to_string()))
                },
            )
            .map_err(|_| FishError::InvalidTransition("TensorRT segment execution failed"))
    }
}

/// A single-stream, request-scoped TensorRT dispatcher. The entire prefill/decode sequence
/// reuses each compiled engine and its execution context without synchronizing per segment.
/// The stream is synchronized while the weighted session and its CUDA buffers are still alive,
/// even when the request returns an error or panics.
pub type ResidentArDispatch<'a> = Box<
    dyn FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()> + 'a,
>;

pub fn with_resident_cuda_ar_session<T>(
    registry: &EngineRegistry,
    logger: &trtx::Logger,
    config: CachedArDispatchConfig<'_>,
    plans: &[&ArStreamPlan],
    checkpoint: &LoadedWeights,
    device_ordinal: usize,
    memory: DeviceMemoryBudget,
    run: impl for<'session> FnOnce(
        &mut WeightedCudaArSession<ResidentArDispatch<'session>>,
    ) -> Result<T>,
) -> Result<T> {
    let mut prepared = Vec::new();
    let mut keys = BTreeMap::new();
    for plan in plans {
        plan.validate()?;
        for phase in &plan.phases {
            let ArStreamPhase::Network(network) = phase else {
                continue;
            };
            let identity = EngineIdentity::for_blueprint(
                network,
                config.model_sha256,
                config.gpu_identity,
                config.tensorrt_version,
                config.driver_version,
                config.precision,
                config.shape_profile,
                config.build_flags,
            )
            .map_err(|_| FishError::InvalidShape("TensorRT AR engine identity invalid"))?;
            let key = identity
                .key()
                .map_err(|_| FishError::InvalidShape("TensorRT AR engine key invalid"))?;
            if !keys.contains_key(&key) {
                keys.insert(key, ());
                prepared.push((network, identity));
            }
        }
    }
    let context = CudaContext::new(device_ordinal)
        .map_err(|_| FishError::InvalidTransition("CUDA device context initialization failed"))?;
    registry
        .with_resident_sessions(
            logger,
            &prepared,
            config.workspace_bytes,
            |contexts| {
                let dispatch: ResidentArDispatch<'_> = Box::new(|plan, buffers, stream| {
                    let identity = EngineIdentity::for_blueprint(
                        plan,
                        config.model_sha256,
                        config.gpu_identity,
                        config.tensorrt_version,
                        config.driver_version,
                        config.precision,
                        config.shape_profile,
                        config.build_flags,
                    )
                    .map_err(|_| FishError::InvalidShape("TensorRT request graph mismatch"))?;
                    let key = identity
                        .key()
                        .map_err(|_| FishError::InvalidShape("TensorRT engine lookup failed"))?;
                    let context = contexts.get_mut(&key).ok_or(FishError::InvalidShape(
                        "TensorRT graph is not prepared for the resident session",
                    ))?;
                    unsafe { enqueue_trt_segment(context, plan, buffers, stream) }
                });
                let device = CudaArDevice::with_context(context, memory, dispatch)
                    .map_err(|error| EngineError::TensorRt(error.to_string()))?;
                let mut session = WeightedCudaArSession::open(device, plans, checkpoint)
                    .map_err(|error| EngineError::TensorRt(error.to_string()))?;
                let outcome = catch_unwind(AssertUnwindSafe(|| run(&mut session)));
                let retired = session.retire_completed();
                drop(session);
                match outcome {
                    Err(panic) => resume_unwind(panic),
                    Ok(value) => {
                        retired.map_err(|error| EngineError::TensorRt(error.to_string()))?;
                        value.map_err(|error| EngineError::TensorRt(error.to_string()))
                    }
                }
            },
            || Ok(()),
        )
        .map_err(|_| FishError::InvalidTransition("resident TensorRT AR session failed"))
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
