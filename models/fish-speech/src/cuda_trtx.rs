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
    cuda_stream::{CudaArDevice, CudaBuffer, CudaScratchPool, CudaWorkspace},
    device_memory::DeviceMemoryBudget,
    trtx_budget_allocator::BudgetedTrtAllocator,
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
    upload_required_weights(device, &required, checkpoint)
}

fn upload_required_weights<F>(
    device: &mut CudaArDevice<F>,
    required: &BTreeMap<String, TensorType>,
    checkpoint: &LoadedWeights,
) -> Result<BTreeMap<String, CudaBuffer>> {
    let mut uploaded = BTreeMap::new();
    let upload = (|| {
        for (name, ty) in required {
            let bytes = checkpoint
                .semantic_upload_bytes(name, ty)
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
    weights: Arc<BTreeMap<String, CudaBuffer>>,
    kernel_inflight: Vec<CudaBuffer>,
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
        Ok(Self::from_weights(device, Arc::new(weights)))
    }

    fn from_weights(device: CudaArDevice<F>, weights: Arc<BTreeMap<String, CudaBuffer>>) -> Self {
        Self {
            executor: ArStreamExecutor::new(device),
            weights,
            kernel_inflight: Vec::new(),
        }
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
        self.enqueue_with_contract(plan, &contract, runtime_inputs)
    }

    pub(crate) fn enqueue_with_contract(
        &mut self,
        plan: &ArStreamPlan,
        contract: &ArInputContract,
        runtime_inputs: &BTreeMap<String, CudaBuffer>,
    ) -> Result<BTreeMap<Binding, CudaBuffer>> {
        let external = contract
            .bind(&self.weights, runtime_inputs, CudaBuffer::bytes)
            .map_err(|_| {
                FishError::InvalidShape("TensorRT AR device inputs missing or undersized")
            })?;
        self.executor.enqueue(plan, &external)
    }

    pub fn retain_cuda_buffer(&mut self, buffer: CudaBuffer) {
        self.kernel_inflight.push(buffer);
    }

    pub fn retire_completed(&mut self) -> Result<()> {
        self.executor.retire_completed()?;
        self.kernel_inflight.clear();
        Ok(())
    }

    pub fn read_output_bytes(&mut self, output: &CudaBuffer) -> Result<Vec<u8>> {
        let bytes = self.executor.download_and_retire(output)?;
        self.kernel_inflight.clear();
        Ok(bytes)
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

/// One model-owned set of contexts. Field order drops dependents before their owners.
pub(crate) struct ResidentTrtContexts {
    contexts: BTreeMap<String, ExecutionContext<'static>>,
    workspaces: Vec<CudaWorkspace>,
    engines: Vec<trtx::CudaEngine<'static>>,
    runtime: trtx::Runtime<'static>,
    pub(crate) logger: Box<trtx::Logger>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) device: Arc<CudaContext>,
}

impl Drop for ResidentTrtContexts {
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        let _ = self.stream.synchronize();
    }
}

pub(crate) struct ResidentCudaAr {
    contexts: ResidentTrtContexts,
    weight_types: BTreeMap<String, TensorType>,
    weights: Arc<BTreeMap<String, CudaBuffer>>,
}

pub(crate) fn prepare_resident_cuda_ar(
    slot: &mut Option<ResidentCudaAr>,
    registry: &EngineRegistry,
    config: &CachedArDispatchConfig<'_>,
    plans: &[&ArStreamPlan],
    checkpoint: &LoadedWeights,
    device_ordinal: usize,
    memory: &DeviceMemoryBudget,
    scratch_pool: &CudaScratchPool,
) -> Result<usize> {
    let mut prepared = BTreeMap::new();
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
            prepared.entry(key).or_insert((network.as_ref(), identity));
        }
    }

    if let Some(current) = slot.as_ref() {
        if current.contexts.contexts.keys().eq(prepared.keys()) {
            return Ok(current.contexts.contexts.len());
        }
    }

    let weight_types = collect_ar_weight_types(plans)
        .map_err(|_| FishError::InvalidShape("TensorRT AR weight plan invalid"))?;
    // Drop the previous contexts before allocating a replacement. Keep immutable
    // weights when both sets of profiles use the same weight contract.
    let previous = slot.take();
    let shared_weights = previous
        .as_ref()
        .filter(|state| state.weight_types == weight_types)
        .map(|state| Arc::clone(&state.weights));
    drop(previous);
    let contexts = ResidentTrtContexts::new(
        registry,
        config,
        prepared,
        device_ordinal,
        memory,
        scratch_pool,
    )
    .map_err(|error| FishError::Backend(format!("prepare TensorRT AR session: {error:?}")))?;
    let weights = if let Some(weights) = shared_weights {
        weights
    } else {
        let dispatch: ResidentArDispatch<'_> = Box::new(|_, _, _| {
            Err(FishError::InvalidTransition(
                "weight upload cannot dispatch TensorRT",
            ))
        });
        let mut uploader = CudaArDevice::with_context_stream(
            Arc::clone(&contexts.device),
            Arc::clone(&contexts.stream),
            memory.clone(),
            dispatch,
        )?;
        Arc::new(upload_required_weights(
            &mut uploader,
            &weight_types,
            checkpoint,
        )?)
    };
    let count = contexts.contexts.len();
    *slot = Some(ResidentCudaAr {
        contexts,
        weight_types,
        weights,
    });
    Ok(count)
}

impl ResidentTrtContexts {
    fn new(
        registry: &EngineRegistry,
        config: &CachedArDispatchConfig<'_>,
        prepared: BTreeMap<String, (&TrtBlueprint, EngineIdentity)>,
        device_ordinal: usize,
        memory: &DeviceMemoryBudget,
        scratch_pool: &CudaScratchPool,
    ) -> std::result::Result<Self, EngineError> {
        let mut resident = Self::empty(device_ordinal, memory, scratch_pool)?;
        for (key, (network, identity)) in prepared {
            identity.validate_blueprint(network)?;
            let image = registry.get_or_build(&identity, || {
                crate::trtx_engine::build_serialized_blueprint_budgeted(
                    &resident.logger,
                    network,
                    config.workspace_bytes,
                    Some(resident.allocator(memory, scratch_pool)?),
                )
            })?;
            resident.insert_with_graph(
                key,
                image.bytes.as_ref(),
                memory,
                scratch_pool,
                config
                    .build_flags
                    .split(',')
                    .any(|flag| flag.trim() == "cuda-graph"),
            )?;
        }
        Ok(resident)
    }

    pub(crate) fn empty(
        device_ordinal: usize,
        memory: &DeviceMemoryBudget,
        scratch_pool: &CudaScratchPool,
    ) -> std::result::Result<Self, EngineError> {
        let cuda_error = |e: cudarc::driver::DriverError| EngineError::TensorRt(e.to_string());
        let trt_error = |e: trtx::Error| EngineError::TensorRt(e.to_string());
        let device = CudaContext::new(device_ordinal).map_err(cuda_error)?;
        device.bind_to_thread().map_err(cuda_error)?;
        let stream = device.new_stream().map_err(cuda_error)?;
        let make_allocator = || {
            trtx::interfaces::GpuAllocator::new(Box::new(BudgetedTrtAllocator::new(
                Arc::clone(&device),
                memory.clone(),
                scratch_pool.clone(),
            )))
            .map_err(trt_error)
        };
        let logger = Box::new(trtx::Logger::stderr().map_err(trt_error)?);
        let mut runtime = trtx::Runtime::new(&logger).map_err(trt_error)?;
        runtime.set_gpu_allocator(make_allocator()?);
        // The logger lives at a stable address and is dropped after the runtime.
        let runtime =
            unsafe { std::mem::transmute::<trtx::Runtime<'_>, trtx::Runtime<'static>>(runtime) };
        Ok(Self {
            contexts: BTreeMap::new(),
            workspaces: Vec::new(),
            engines: Vec::new(),
            runtime,
            logger,
            stream: Arc::clone(&stream),
            device: Arc::clone(&device),
        })
    }

    pub(crate) fn allocator(
        &self,
        memory: &DeviceMemoryBudget,
        scratch_pool: &CudaScratchPool,
    ) -> std::result::Result<std::pin::Pin<Box<trtx::interfaces::GpuAllocator>>, EngineError> {
        trtx::interfaces::GpuAllocator::new(Box::new(BudgetedTrtAllocator::new(
            Arc::clone(&self.device),
            memory.clone(),
            scratch_pool.clone(),
        )))
        .map_err(|e| EngineError::TensorRt(e.to_string()))
    }

    /// The caller owns valid buffers for every binding through stream completion.
    pub(crate) unsafe fn enqueue_firefly(
        &mut self,
        input_name: &str,
        input: *mut std::ffi::c_void,
        output: *mut std::ffi::c_void,
    ) -> std::result::Result<(), EngineError> {
        let error = |e: trtx::Error| EngineError::TensorRt(e.to_string());
        let context = self
            .contexts
            .get_mut("firefly")
            .ok_or_else(|| EngineError::InvalidPlan("missing Firefly context".into()))?;
        unsafe {
            context
                .set_tensor_address(input_name, input)
                .map_err(error)?;
            context
                .set_tensor_address("waveform", output)
                .map_err(error)?;
            context
                .enqueue_v3(self.stream.cu_stream().cast())
                .map_err(error)
        }
    }

    pub(crate) fn insert(
        &mut self,
        key: String,
        image: &[u8],
        memory: &DeviceMemoryBudget,
        scratch_pool: &CudaScratchPool,
    ) -> std::result::Result<(), EngineError> {
        self.insert_with_graph(key, image, memory, scratch_pool, false)
    }

    fn insert_with_graph(
        &mut self,
        key: String,
        image: &[u8],
        memory: &DeviceMemoryBudget,
        scratch_pool: &CudaScratchPool,
        capture: bool,
    ) -> std::result::Result<(), EngineError> {
        let trt_error = |e: trtx::Error| EngineError::TensorRt(e.to_string());
        let mut engine = self
            .runtime
            .deserialize_cuda_engine(image)
            .map_err(trt_error)?;
        let required = engine.device_memory_size_v2().map_err(trt_error)?;
        let mut execution = if capture {
            let mut options = engine.create_runtime_config().map_err(trt_error)?;
            options.set_execution_context_allocation_strategy(
                trtx::ExecutionContextAllocationStrategy::kUSER_MANAGED,
            );
            options
                .set_cuda_graph_strategy(trtx::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE)
                .map_err(trt_error)?;
            engine.create_execution_context_with_config(std::rc::Rc::new(options))
        } else {
            engine.create_execution_context_user_managed()
        }
        .map_err(trt_error)?;
        if required != 0 {
            let size = i64::try_from(required).map_err(|_| {
                EngineError::InvalidPlan("TensorRT context memory exceeds i64".into())
            })?;
            let allocation = CudaWorkspace::allocate(&self.stream, memory, required)
                .or_else(|_| {
                    scratch_pool.clear()?;
                    CudaWorkspace::allocate(&self.stream, memory, required)
                })
                .map_err(|e| EngineError::TensorRt(e.to_string()))?;
            let (address, bytes) = allocation.region(&self.stream);
            let region = DeviceRegion::new(address, bytes)?;
            if region.bytes < required {
                return Err(EngineError::InvalidBindings(
                    "TensorRT workspace allocation too small".into(),
                ));
            }
            unsafe { execution.set_device_memory_v2(region.address as *mut _, size) };
            self.workspaces.push(allocation);
        }
        // The wrapper lifetime tracks the native engine, not the Rust wrapper address.
        // Store the context and its engine together and never expose the erased lifetime.
        let execution = unsafe {
            std::mem::transmute::<ExecutionContext<'_>, ExecutionContext<'static>>(execution)
        };
        self.engines.push(engine);
        self.contexts.insert(key, execution);
        Ok(())
    }
}

pub(crate) fn with_resident_cuda_ar_session<T>(
    registry: &EngineRegistry,
    resident: &mut Option<ResidentCudaAr>,
    config: CachedArDispatchConfig<'_>,
    plans: &[&ArStreamPlan],
    checkpoint: &LoadedWeights,
    device_ordinal: usize,
    memory: DeviceMemoryBudget,
    scratch_pool: CudaScratchPool,
    run: impl for<'session> FnOnce(
        &mut WeightedCudaArSession<ResidentArDispatch<'session>>,
    ) -> Result<T>,
) -> Result<T> {
    prepare_resident_cuda_ar(
        resident,
        registry,
        &config,
        plans,
        checkpoint,
        device_ordinal,
        &memory,
        &scratch_pool,
    )?;
    let batch = catch_unwind(AssertUnwindSafe(|| {
        let resident = resident
            .as_mut()
            .expect("resident context set was prepared");
        let weights = Arc::clone(&resident.weights);
        let resident = &mut resident.contexts;
        let device = Arc::clone(&resident.device);
        let stream = Arc::clone(&resident.stream);
        device
            .bind_to_thread()
            .map_err(|e| EngineError::TensorRt(e.to_string()))?;
        let result = (|| {
            let mut segments = BTreeMap::new();
            for plan in plans {
                for phase in &plan.phases {
                    let ArStreamPhase::Network(network) = phase else {
                        continue;
                    };
                    let key = EngineIdentity::for_blueprint(
                        network,
                        config.model_sha256,
                        config.gpu_identity,
                        config.tensorrt_version,
                        config.driver_version,
                        config.precision,
                        config.shape_profile,
                        config.build_flags,
                    )
                    .and_then(|identity| identity.key())
                    .map_err(|_| EngineError::InvalidPlan("invalid resident blueprint".into()))?;
                    let contract = EngineIoContract::from_blueprint(network)?;
                    let mut bindings = Vec::new();
                    let execution = resident.contexts.get_mut(&key).ok_or_else(|| {
                        EngineError::InvalidPlan("missing prepared context".into())
                    })?;
                    for step in &network.steps {
                        match step.primitive {
                            TrtPrimitive::ConstantWeight => {
                                let buffer = weights
                                    .get(&step.name)
                                    .filter(|b| b.bytes() >= contract.inputs[&step.name])
                                    .ok_or_else(|| {
                                        EngineError::InvalidBindings(step.name.clone())
                                    })?;
                                unsafe {
                                    execution.set_tensor_address(
                                        &step.name,
                                        buffer.address(&stream) as *mut _,
                                    )
                                }
                                .map_err(|e| EngineError::TensorRt(e.to_string()))?;
                            }
                            TrtPrimitive::Input => bindings.push((
                                step.result,
                                step.name.clone(),
                                contract.inputs[&step.name],
                            )),
                            _ => {}
                        }
                    }
                    for &binding in &network.outputs {
                        let name = crate::trtx_engine::output_tensor_name(binding);
                        bindings.push((binding, name.clone(), contract.outputs[&name]));
                    }
                    segments.insert(Arc::as_ptr(network), (key, bindings));
                }
            }
            let dispatch: ResidentArDispatch<'_> = Box::new(|plan, buffers, stream| {
                let (key, bindings) =
                    segments
                        .get(&std::ptr::from_ref(plan))
                        .ok_or(FishError::InvalidShape(
                            "blueprint outside prepared request",
                        ))?;
                let execution = resident
                    .contexts
                    .get_mut(key)
                    .ok_or(FishError::InvalidShape(
                        "TensorRT graph is not prepared for the resident session",
                    ))?;
                for (binding, name, bytes) in bindings {
                    let buffer = buffers
                        .get(binding)
                        .filter(|buf| buf.bytes() >= *bytes)
                        .ok_or(FishError::InvalidShape(
                            "missing or undersized resident binding",
                        ))?;
                    unsafe { execution.set_tensor_address(name, buffer.address(stream) as *mut _) }
                        .map_err(|_| {
                            FishError::InvalidTransition("TensorRT tensor binding failed")
                        })?;
                }
                unsafe { execution.enqueue_v3(stream.cu_stream().cast()) }
                    .map_err(|_| FishError::InvalidTransition("TensorRT segment enqueue failed"))
            });
            let device = CudaArDevice::with_context_stream(
                Arc::clone(&device),
                Arc::clone(&stream),
                memory.clone(),
                dispatch,
            )
            .and_then(|device| device.with_scratch_pool(scratch_pool))
            .map_err(|error| EngineError::TensorRt(error.to_string()))?;
            let mut session = WeightedCudaArSession::from_weights(device, weights);
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
        })();
        stream
            .synchronize()
            .map_err(|e| EngineError::TensorRt(e.to_string()))?;
        result
    }));
    match batch {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => {
            *resident = None;
            Err(FishError::Backend(format!(
                "resident TensorRT AR session: {error:?}"
            )))
        }
        Err(panic) => {
            *resident = None;
            resume_unwind(panic)
        }
    }
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
