use std::collections::{BTreeMap, BTreeSet};
use fish_speech_core::{
    graph_ir::ScalarType,
    trtx_blueprint::{Binding, TrtBlueprint, TrtPrimitive},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    InvalidPlan(String),
    InvalidBindings(String),
    TensorRt(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineIoContract {
    pub inputs: BTreeMap<String, usize>,
    pub outputs: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRegion {
    pub address: usize,
    pub bytes: usize,
}

impl DeviceRegion {
    pub fn new(address: usize, bytes: usize) -> Result<Self, EngineError> {
        if address == 0 || bytes == 0 {
            return Err(EngineError::InvalidBindings("empty device region".into()));
        }
        Ok(Self { address, bytes })
    }
}

pub fn output_tensor_name(binding: Binding) -> String {
    match binding {
        Binding::Value(id) => format!("fish_output_value_{}", id.0),
        Binding::Temp(id) => format!("fish_output_temp_{id}"),
    }
}

fn tensor_bytes(shape: &[usize], dtype: ScalarType) -> Result<usize, EngineError> {
    let element = match dtype {
        ScalarType::I32 | ScalarType::F32 => 4usize,
        ScalarType::BF16 => 2usize,
    };
    if shape.is_empty() || shape.contains(&0) {
        return Err(EngineError::InvalidPlan("invalid tensor dimensions".into()));
    }
    shape.iter().try_fold(element, |n, dim| {
        n.checked_mul(*dim)
            .ok_or_else(|| EngineError::InvalidPlan("tensor byte count overflow".into()))
    })
}

impl EngineIoContract {
    pub fn from_blueprint(plan: &TrtBlueprint) -> Result<Self, EngineError> {
        plan.validate().map_err(|e| EngineError::InvalidPlan(format!("{e:?}")))?;
        let mut inputs = BTreeMap::new();
        let mut outputs = BTreeMap::new();
        let mut produced = BTreeMap::new();
        let mut input_ids = BTreeSet::new();
        for step in &plan.steps {
            if matches!(step.primitive, TrtPrimitive::Deferred { .. }) {
                return Err(EngineError::InvalidPlan(format!("unsupported step: {}", step.name)));
            }
            let size = tensor_bytes(&step.output_type.shape, step.output_type.dtype)?;
            if produced.insert(step.result, size).is_some() {
                return Err(EngineError::InvalidPlan("duplicate tensor binding".into()));
            }
            if matches!(step.primitive, TrtPrimitive::Input | TrtPrimitive::ConstantWeight) {
                if inputs.insert(step.name.clone(), size).is_some() {
                    return Err(EngineError::InvalidPlan("duplicate input name".into()));
                }
                input_ids.insert(step.result);
            }
        }
        for id in &plan.outputs {
            if input_ids.contains(id) {
                return Err(EngineError::InvalidPlan("direct input output needs identity layer".into()));
            }
            let size = *produced.get(id)
                .ok_or_else(|| EngineError::InvalidPlan("output producer missing".into()))?;
            let name = output_tensor_name(*id);
            if inputs.contains_key(&name) || outputs.insert(name, size).is_some() {
                return Err(EngineError::InvalidPlan("duplicate output name".into()));
            }
        }
        Ok(Self { inputs, outputs })
    }

    pub fn validate_regions(&self, regions: &BTreeMap<String, DeviceRegion>) -> Result<(), EngineError> {
        if regions.len() != self.inputs.len() + self.outputs.len() {
            return Err(EngineError::InvalidBindings("wrong buffer count".into()));
        }
        for (name, min_bytes) in self.inputs.iter().chain(self.outputs.iter()) {
            let region = regions.get(name)
                .ok_or_else(|| EngineError::InvalidBindings(format!("missing {name}")))?;
            if region.address == 0 || region.bytes < *min_bytes {
                return Err(EngineError::InvalidBindings(format!("insufficient allocation: {name}")));
            }
        }
        if regions.keys().any(|key| !self.inputs.contains_key(key) && !self.outputs.contains_key(key)) {
            return Err(EngineError::InvalidBindings("unexpected buffer".into()));
        }
        Ok(())
    }
}

#[cfg(feature = "tensorrt-api")]
pub fn build_serialized_blueprint(
    logger: &trtx::Logger,
    plan: &TrtBlueprint,
    workspace_bytes: usize,
) -> Result<Vec<u8>, EngineError> {
    use trtx::{Builder, MemoryPoolType, builder::network_flags};
    EngineIoContract::from_blueprint(plan)?;
    if workspace_bytes == 0 {
        return Err(EngineError::InvalidPlan("workspace must be nonzero".into()));
    }
    let err = |e: trtx::Error| EngineError::TensorRt(e.to_string());
    let mut builder = Builder::new(logger).map_err(err)?;
    let mut network = builder.create_network(network_flags::EXPLICIT_BATCH).map_err(err)?;
    crate::trtx_lowering::lower_blueprint(&mut network, plan)
        .map_err(|e| EngineError::TensorRt(format!("{e:?}")))?;
    let mut config = builder.create_builder_config().map_err(err)?;
    config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, workspace_bytes);
    let serialized = builder.build_serialized_network(&mut network, &mut config).map_err(err)?;
    if serialized.is_empty() {
        return Err(EngineError::TensorRt("empty serialized engine".into()));
    }
    Ok(serialized.as_ref().to_vec())
}

/// # Safety
/// All addresses must refer to valid CUDA allocations of the declared size and remain valid until
/// stream completion. The context and stream must belong to the same device, and callers must
/// serialize mutable context use.
#[cfg(feature = "tensorrt-api")]
pub unsafe fn enqueue_with_regions(
    context: &mut trtx::ExecutionContext<'_>,
    contract: &EngineIoContract,
    regions: &BTreeMap<String, DeviceRegion>,
    stream: *mut std::ffi::c_void,
) -> Result<(), EngineError> {
    contract.validate_regions(regions)?;
    if stream.is_null() {
        return Err(EngineError::InvalidBindings("null CUDA stream".into()));
    }
    let err = |e: trtx::Error| EngineError::TensorRt(e.to_string());
    for name in contract.inputs.keys().chain(contract.outputs.keys()) {
        unsafe { context.set_tensor_address(name, regions[name].address as *mut _) }.map_err(err)?;
    }
    unsafe { context.enqueue_v3(stream) }.map_err(err)
}
