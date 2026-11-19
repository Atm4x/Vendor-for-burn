use std::collections::{BTreeMap, BTreeSet};

use crate::ar_stream_plan::{ArStreamPhase, ArStreamPlan};
use crate::error::{FishError, Result};
use crate::graph_ir::{ScalarType, TensorType};
use crate::kv_device_owner::{KvDeviceOwner, KvStreamDevice};
use crate::trtx_blueprint::{Binding, TrtBlueprint, TrtPrimitive};

/// All implementations must use the same ordered stream for TensorRT segment execution and
/// the `KvStreamDevice::copy_2d_async` calls. Network compilation is outside this interface;
/// each segment must be compiled or looked up before the first enqueue.
pub trait ArStreamDevice: KvStreamDevice
where
    Self::Buffer: Clone,
{
    fn allocate_tensor(&mut self, ty: &TensorType) -> Result<Self::Buffer>;
    fn enqueue_network(
        &mut self,
        network: &TrtBlueprint,
        bindings: &BTreeMap<Binding, Self::Buffer>,
    ) -> Result<()>;
}

/// Stream-ordered submission coordinator. Owns physical caches across Slow/Fast calls and
/// retains temporary device allocations until `reset_request` synchronizes the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchLimits {
    pub max_inflight_bytes: usize,
    pub max_inflight_submissions: usize,
}

impl Default for ScratchLimits {
    fn default() -> Self {
        Self {
            max_inflight_bytes: 512 * 1024 * 1024,
            max_inflight_submissions: 128,
        }
    }
}

pub struct ArStreamExecutor<D>
where
    D: ArStreamDevice,
    D::Buffer: Clone,
{
    kv: KvDeviceOwner<D>,
    inflight: Vec<Vec<D::Buffer>>,
    inflight_bytes: usize,
    limits: ScratchLimits,
}

impl<D> ArStreamExecutor<D>
where
    D: ArStreamDevice,
    D::Buffer: Clone,
{
    pub fn new(device: D) -> Self {
        Self::with_scratch_limits(device, ScratchLimits::default())
            .expect("default scratch limits must be valid")
    }

    pub fn with_scratch_limits(device: D, limits: ScratchLimits) -> Result<Self> {
        if limits.max_inflight_bytes == 0 || limits.max_inflight_submissions == 0 {
            return Err(FishError::InvalidConfig("scratch limits must be positive"));
        }
        Ok(Self {
            kv: KvDeviceOwner::new(device),
            inflight: Vec::new(),
            inflight_bytes: 0,
            limits,
        })
    }

    fn scratch_bytes_for(plan: &ArStreamPlan) -> Result<usize> {
        let mut seen = BTreeSet::new();
        let mut bytes = 0usize;
        for phase in &plan.phases {
            if let ArStreamPhase::Network(network) = phase {
                for output in &network.outputs {
                    if !seen.insert(*output) {
                        continue;
                    }
                    let step = network
                        .steps
                        .iter()
                        .find(|step| step.result == *output)
                        .ok_or(FishError::InvalidShape(
                            "AR output missing tensor definition",
                        ))?;
                    bytes = bytes
                        .checked_add(tensor_bytes(&step.output_type)?)
                        .ok_or(FishError::InvalidShape("scratch byte count overflow"))?;
                }
            }
        }
        Ok(bytes)
    }

    fn preflight_external_inputs(
        &self,
        plan: &ArStreamPlan,
        external: &BTreeMap<String, D::Buffer>,
    ) -> Result<()> {
        let cache_names = plan
            .phases
            .iter()
            .filter_map(|phase| match phase {
                ArStreamPhase::KvWrites(writes) => Some([
                    writes.key.cache_name.as_str(),
                    writes.value.cache_name.as_str(),
                ]),
                ArStreamPhase::Network(_) => None,
            })
            .flatten()
            .collect::<BTreeSet<_>>();
        let mut produced = BTreeSet::new();
        for phase in &plan.phases {
            match phase {
                ArStreamPhase::Network(network) => {
                    for step in &network.steps {
                        if matches!(
                            step.primitive,
                            TrtPrimitive::Input | TrtPrimitive::ConstantWeight
                        ) && !produced.contains(&step.result)
                            && !cache_names.contains(step.name.as_str())
                            && !external.contains_key(&step.name)
                        {
                            return Err(FishError::InvalidShape("missing AR segment device input"));
                        }
                    }
                    produced.extend(network.outputs.iter().copied());
                }
                ArStreamPhase::KvWrites(writes) => {
                    produced.insert(writes.key.updated_alias);
                    produced.insert(writes.value.updated_alias);
                }
            }
        }
        Ok(())
    }

    pub fn enqueue(
        &mut self,
        plan: &ArStreamPlan,
        external: &BTreeMap<String, D::Buffer>,
    ) -> Result<BTreeMap<Binding, D::Buffer>> {
        if self.kv.is_poisoned() {
            return Err(FishError::InvalidTransition(
                "KV stream poisoned; reset required",
            ));
        }
        plan.validate()?;
        self.kv
            .preflight_writes(plan.phases.iter().filter_map(|phase| match phase {
                ArStreamPhase::KvWrites(writes) => Some(writes),
                ArStreamPhase::Network(_) => None,
            }))?;
        self.preflight_external_inputs(plan, external)?;
        let reservation = Self::scratch_bytes_for(plan)?;
        let expected = self
            .inflight_bytes
            .checked_add(reservation)
            .ok_or(FishError::InvalidShape("scratch reservation overflow"))?;
        if self.inflight.len() >= self.limits.max_inflight_submissions
            || expected > self.limits.max_inflight_bytes
        {
            return Err(FishError::InvalidTransition(
                "AR scratch budget exhausted; retire completed stream work",
            ));
        }
        let mut bindings = BTreeMap::new();
        let mut retained = Vec::new();
        let result = self.enqueue_inner(plan, external, &mut bindings, &mut retained);
        retained.extend(bindings.into_values());
        if result.is_err() {
            self.kv.poison();
        }
        self.inflight.push(retained);
        self.inflight_bytes = expected;
        result
    }

    fn enqueue_inner(
        &mut self,
        plan: &ArStreamPlan,
        external: &BTreeMap<String, D::Buffer>,
        bindings: &mut BTreeMap<Binding, D::Buffer>,
        retained: &mut Vec<D::Buffer>,
    ) -> Result<BTreeMap<Binding, D::Buffer>> {
        if self.kv.is_poisoned() {
            return Err(FishError::InvalidTransition(
                "KV stream poisoned; reset required",
            ));
        }
        for phase in &plan.phases {
            if let ArStreamPhase::KvWrites(writes) = phase {
                self.kv.ensure_cache(&writes.key)?;
                self.kv.ensure_cache(&writes.value)?;
            }
        }
        for phase in &plan.phases {
            match phase {
                ArStreamPhase::Network(segment) => {
                    for step in &segment.steps {
                        if matches!(
                            step.primitive,
                            TrtPrimitive::Input | TrtPrimitive::ConstantWeight
                        ) && !bindings.contains_key(&step.result)
                        {
                            let handle = if let Some(buffer) = external.get(&step.name) {
                                buffer.clone()
                            } else if let Some(buffer) = self.kv.cache(&step.name) {
                                buffer.clone()
                            } else {
                                let result =
                                    Err(FishError::InvalidShape("missing AR segment device input"));
                                self.kv.poison();
                                return result;
                            };
                            bindings.insert(step.result, handle);
                        }
                    }
                    for output in &segment.outputs {
                        if !bindings.contains_key(output) {
                            let step = segment
                                .steps
                                .iter()
                                .find(|step| step.result == *output)
                                .ok_or(FishError::InvalidShape(
                                    "AR segment missing output producer",
                                ))?;
                            let allocation =
                                match self.kv.device_mut().allocate_tensor(&step.output_type) {
                                    Ok(allocation) => allocation,
                                    Err(err) => {
                                        self.kv.poison();
                                        return Err(err);
                                    }
                                };
                            bindings.insert(*output, allocation.clone());
                            retained.push(allocation);
                        }
                    }
                    if let Err(err) = self.kv.device_mut().enqueue_network(segment, &bindings) {
                        self.kv.poison();
                        return Err(err);
                    }
                }
                ArStreamPhase::KvWrites(writes) => {
                    let key = bindings
                        .get(&writes.key.source)
                        .ok_or(FishError::InvalidShape("missing projected BF16 key buffer"))?;
                    let value =
                        bindings
                            .get(&writes.value.source)
                            .ok_or(FishError::InvalidShape(
                                "missing projected BF16 value buffer",
                            ))?;
                    if let Err(err) = self.kv.enqueue_pair(writes, key, value) {
                        self.kv.poison();
                        return Err(err);
                    }
                    bindings.insert(
                        writes.key.updated_alias,
                        self.kv.cache(&writes.key.cache_name).unwrap().clone(),
                    );
                    bindings.insert(
                        writes.value.updated_alias,
                        self.kv.cache(&writes.value.cache_name).unwrap().clone(),
                    );
                }
            }
        }
        let mut outputs = BTreeMap::new();
        for output in &plan.final_outputs {
            let handle = bindings
                .get(output)
                .ok_or(FishError::InvalidShape("missing final AR output"))?;
            outputs.insert(*output, handle.clone());
        }
        Ok(outputs)
    }

    pub fn retire_completed(&mut self) -> Result<()> {
        self.kv.device_mut().synchronize()?;
        self.inflight.clear();
        self.inflight_bytes = 0;
        Ok(())
    }

    pub fn reset_fast_prefix(&mut self) -> Result<()> {
        self.kv.reset_fast_prefix()
    }

    pub fn reset_request(&mut self) -> Result<()> {
        self.kv.reset_request()?;
        self.inflight.clear();
        self.inflight_bytes = 0;
        Ok(())
    }

    /// Access the same stream-owning device for ordered input uploads and downloads.
    /// Callers must not free buffers referenced by pending submissions before retirement.
    pub fn device_mut(&mut self) -> &mut D {
        self.kv.device_mut()
    }

    pub fn inflight_bytes(&self) -> usize {
        self.inflight_bytes
    }

    pub fn cache_active_len(&self, name: &str) -> Option<usize> {
        self.kv.active_len(name)
    }

    pub fn inflight_submissions(&self) -> usize {
        self.inflight.len()
    }
}

pub fn tensor_bytes(ty: &TensorType) -> Result<usize> {
    let size: usize = match ty.dtype {
        ScalarType::I32 | ScalarType::F32 => 4,
        ScalarType::BF16 => 2,
    };
    ty.shape
        .iter()
        .try_fold(size, |bytes, &dim| bytes.checked_mul(dim))
        .ok_or(FishError::InvalidShape(
            "tensor allocation byte count overflow",
        ))
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use super::*;
    use crate::{
        config::FishModelConfig, kv_device::KvWrite2d, transformer_ir::TransformerIrFactory,
    };

    struct FakeBackend {
        operations: Rc<RefCell<Vec<&'static str>>>,
    }

    impl KvStreamDevice for FakeBackend {
        type Buffer = Rc<RefCell<Vec<u8>>>;
        fn allocate_bf16(&mut self, bytes: usize) -> Result<Self::Buffer> {
            Ok(Rc::new(RefCell::new(vec![0; bytes])))
        }
        fn copy_2d_async(
            &mut self,
            dest: &Self::Buffer,
            src: &Self::Buffer,
            geometry: KvWrite2d,
        ) -> Result<()> {
            self.operations.borrow_mut().push("copy");
            let src = src.borrow();
            let mut dest = dest.borrow_mut();
            for row in 0..geometry.rows {
                let dst =
                    row * geometry.destination_pitch_bytes + geometry.destination_offset_bytes;
                let start = row * geometry.source_pitch_bytes;
                dest[dst..dst + geometry.row_bytes]
                    .copy_from_slice(&src[start..start + geometry.row_bytes]);
            }
            Ok(())
        }
        fn synchronize(&mut self) -> Result<()> {
            self.operations.borrow_mut().push("sync");
            Ok(())
        }
    }

    impl ArStreamDevice for FakeBackend {
        fn allocate_tensor(&mut self, ty: &TensorType) -> Result<Self::Buffer> {
            Ok(Rc::new(RefCell::new(vec![0; tensor_bytes(ty)?])))
        }
        fn enqueue_network(
            &mut self,
            net: &TrtBlueprint,
            bindings: &BTreeMap<Binding, Self::Buffer>,
        ) -> Result<()> {
            self.operations.borrow_mut().push("network");
            for output in &net.outputs {
                bindings[output].borrow_mut().fill(0);
            }
            Ok(())
        }
    }

    #[test]
    fn network_and_two_kv_copies_follow_same_order_without_token_sync() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let graph = TransformerIrFactory::new(&config)
            .unwrap()
            .slow_prefill(4)
            .unwrap();
        let plan = ArStreamPlan::from_graph(&graph).unwrap();
        let operations = Rc::new(RefCell::new(Vec::new()));
        let mut executor = ArStreamExecutor::new(FakeBackend {
            operations: operations.clone(),
        });
        let mut external = BTreeMap::new();
        for phase in &plan.phases {
            if let ArStreamPhase::Network(net) = phase {
                for step in &net.steps {
                    if matches!(
                        step.primitive,
                        TrtPrimitive::Input | TrtPrimitive::ConstantWeight
                    ) && !step.name.ends_with("_updated")
                        && !step.name.ends_with("_cache")
                    {
                        external.entry(step.name.clone()).or_insert_with(|| {
                            Rc::new(RefCell::new(vec![
                                0;
                                tensor_bytes(&step.output_type).unwrap()
                            ]))
                        });
                    }
                }
            }
        }
        let results = executor.enqueue(&plan, &external).unwrap();
        assert_eq!(results.len(), plan.final_outputs.len());
        assert_eq!(executor.inflight_submissions(), 1);
        assert_eq!(
            operations.borrow().as_slice(),
            &["network", "copy", "copy", "network", "copy", "copy", "network"]
        );
        executor.reset_request().unwrap();
        assert_eq!(executor.inflight_submissions(), 0);
        assert_eq!(operations.borrow().last(), Some(&"sync"));
    }

    #[test]
    fn scratch_budget_rejects_before_enqueue_and_retires_without_resetting_kv() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let graph = TransformerIrFactory::new(&config)
            .unwrap()
            .slow_prefill(4)
            .unwrap();
        let plan = ArStreamPlan::from_graph(&graph).unwrap();
        let operations = Rc::new(RefCell::new(Vec::new()));
        let mut executor = ArStreamExecutor::with_scratch_limits(
            FakeBackend {
                operations: operations.clone(),
            },
            ScratchLimits {
                max_inflight_bytes: 1024 * 1024,
                max_inflight_submissions: 1,
            },
        )
        .unwrap();
        let mut external = BTreeMap::new();
        for phase in &plan.phases {
            if let ArStreamPhase::Network(net) = phase {
                for step in &net.steps {
                    if matches!(
                        step.primitive,
                        TrtPrimitive::Input | TrtPrimitive::ConstantWeight
                    ) && !step.name.ends_with("_updated")
                        && !step.name.ends_with("_cache")
                    {
                        external.entry(step.name.clone()).or_insert_with(|| {
                            Rc::new(RefCell::new(vec![
                                0;
                                tensor_bytes(&step.output_type).unwrap()
                            ]))
                        });
                    }
                }
            }
        }
        executor.enqueue(&plan, &external).unwrap();
        let bytes = executor.inflight_bytes();
        assert!(bytes > 0);
        let active = executor.cache_active_len("slow_layers.0.k_cache");
        let before = operations.borrow().len();
        assert!(matches!(
            executor.enqueue(&plan, &external),
            Err(FishError::InvalidTransition(_))
        ));
        assert_eq!(operations.borrow().len(), before);
        executor.retire_completed().unwrap();
        assert_eq!(executor.inflight_bytes(), 0);
        assert_eq!(executor.cache_active_len("slow_layers.0.k_cache"), active);
        assert_eq!(executor.inflight_submissions(), 0);
        assert_eq!(operations.borrow().last(), Some(&"sync"));
    }

    #[test]
    fn invalid_scratch_limits_and_too_small_budget_fail_before_submission() {
        assert!(ArStreamExecutor::with_scratch_limits(
            FakeBackend {
                operations: Rc::default()
            },
            ScratchLimits {
                max_inflight_bytes: 0,
                max_inflight_submissions: 1
            },
        )
        .is_err());
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let graph = TransformerIrFactory::new(&config)
            .unwrap()
            .slow_prefill(4)
            .unwrap();
        let plan = ArStreamPlan::from_graph(&graph).unwrap();
        let operations = Rc::new(RefCell::new(Vec::new()));
        let mut executor = ArStreamExecutor::with_scratch_limits(
            FakeBackend {
                operations: operations.clone(),
            },
            ScratchLimits {
                max_inflight_bytes: 1,
                max_inflight_submissions: 1,
            },
        )
        .unwrap();
        assert!(executor.enqueue(&plan, &BTreeMap::new()).is_err());
        assert!(operations.borrow().is_empty());
        assert_eq!(executor.inflight_submissions(), 0);
    }

    #[test]
    fn rejects_stale_kv_prefix_before_submitting_any_network() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&config).unwrap();
        let prefill = ArStreamPlan::from_graph(&factory.slow_prefill(4).unwrap()).unwrap();
        let decode = ArStreamPlan::from_graph(&factory.slow_decode(5).unwrap()).unwrap();
        let first_cache = prefill
            .phases
            .iter()
            .find_map(|phase| match phase {
                ArStreamPhase::KvWrites(write) => Some(write.key.cache_name.as_str()),
                _ => None,
            })
            .unwrap();
        let operations = Rc::new(RefCell::new(Vec::new()));
        let mut executor = ArStreamExecutor::new(FakeBackend {
            operations: operations.clone(),
        });
        let external_for = |plan: &ArStreamPlan| {
            let mut external = BTreeMap::new();
            for phase in &plan.phases {
                if let ArStreamPhase::Network(net) = phase {
                    for step in &net.steps {
                        if matches!(
                            step.primitive,
                            TrtPrimitive::Input | TrtPrimitive::ConstantWeight
                        ) && !step.name.ends_with("_cache")
                            && !step.name.ends_with("_updated")
                        {
                            external.entry(step.name.clone()).or_insert_with(|| {
                                Rc::new(RefCell::new(vec![
                                    0;
                                    tensor_bytes(&step.output_type).unwrap()
                                ]))
                            });
                        }
                    }
                }
            }
            external
        };
        let prefill_inputs = external_for(&prefill);
        executor.enqueue(&prefill, &prefill_inputs).unwrap();
        let before = operations.borrow().len();
        let inflight = executor.inflight_submissions();
        assert!(executor.enqueue(&prefill, &prefill_inputs).is_err());
        assert_eq!(operations.borrow().len(), before);
        assert_eq!(executor.inflight_submissions(), inflight);
        assert_eq!(executor.cache_active_len(first_cache), Some(4));
        executor.enqueue(&decode, &external_for(&decode)).unwrap();
        assert!(operations.borrow().len() > before);
    }

    #[test]
    fn resident_slow_prefill_and_multiple_decodes_retire_once() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&config).unwrap();
        let plans = [
            ArStreamPlan::from_graph(&factory.slow_prefill(4).unwrap()).unwrap(),
            ArStreamPlan::from_graph(&factory.slow_decode(5).unwrap()).unwrap(),
            ArStreamPlan::from_graph(&factory.slow_decode(6).unwrap()).unwrap(),
        ];
        let cache_name = plans[0]
            .phases
            .iter()
            .find_map(|phase| match phase {
                ArStreamPhase::KvWrites(writes) => Some(writes.key.cache_name.clone()),
                ArStreamPhase::Network(_) => None,
            })
            .unwrap();
        let operations = Rc::new(RefCell::new(Vec::new()));
        let mut executor = ArStreamExecutor::new(FakeBackend {
            operations: operations.clone(),
        });
        for plan in &plans {
            let mut external = BTreeMap::new();
            for phase in &plan.phases {
                if let ArStreamPhase::Network(network) = phase {
                    for step in &network.steps {
                        if matches!(
                            step.primitive,
                            TrtPrimitive::Input | TrtPrimitive::ConstantWeight
                        ) && !step.name.ends_with("_cache")
                            && !step.name.ends_with("_updated")
                        {
                            external.entry(step.name.clone()).or_insert_with(|| {
                                Rc::new(RefCell::new(vec![
                                    0;
                                    tensor_bytes(&step.output_type).unwrap()
                                ]))
                            });
                        }
                    }
                }
            }
            executor.enqueue(plan, &external).unwrap();
        }
        assert_eq!(executor.inflight_submissions(), 3);
        assert_eq!(executor.cache_active_len(&cache_name), Some(6));
        assert!(!operations.borrow().contains(&"sync"));
        executor.retire_completed().unwrap();
        assert_eq!(
            operations.borrow().iter().filter(|x| **x == "sync").count(),
            1
        );
        assert_eq!(executor.inflight_submissions(), 0);
        assert_eq!(executor.cache_active_len(&cache_name), Some(6));
    }

    #[test]
    fn missing_later_segment_input_rejected_before_first_enqueue() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let plan = ArStreamPlan::from_graph(
            &TransformerIrFactory::new(&config)
                .unwrap()
                .slow_prefill(4)
                .unwrap(),
        )
        .unwrap();
        let first_network = match &plan.phases[0] {
            ArStreamPhase::Network(network) => network,
            _ => unreachable!(),
        };
        let first_inputs = first_network
            .steps
            .iter()
            .filter(|step| {
                matches!(
                    step.primitive,
                    TrtPrimitive::Input | TrtPrimitive::ConstantWeight
                )
            })
            .map(|step| step.name.as_str())
            .collect::<BTreeSet<_>>();
        let mut external = BTreeMap::new();
        for phase in &plan.phases {
            if let ArStreamPhase::Network(network) = phase {
                for step in &network.steps {
                    if matches!(
                        step.primitive,
                        TrtPrimitive::Input | TrtPrimitive::ConstantWeight
                    ) && !step.name.ends_with("_cache")
                        && !step.name.ends_with("_updated")
                    {
                        external.entry(step.name.clone()).or_insert_with(|| {
                            Rc::new(RefCell::new(vec![
                                0;
                                tensor_bytes(&step.output_type).unwrap()
                            ]))
                        });
                    }
                }
            }
        }
        let missing_later = external
            .keys()
            .find(|name| !first_inputs.contains(name.as_str()))
            .cloned()
            .expect("fixture needs an input first used in a later segment");
        external.remove(&missing_later);
        let operations = Rc::new(RefCell::new(Vec::new()));
        let mut executor = ArStreamExecutor::new(FakeBackend {
            operations: operations.clone(),
        });
        assert!(executor.enqueue(&plan, &external).is_err());
        assert!(operations.borrow().is_empty());
        assert_eq!(executor.inflight_submissions(), 0);
        assert!(executor.cache_active_len("slow_layers.0.k_cache").is_none());
    }

    #[test]
    fn tensor_allocation_is_checked() {
        assert_eq!(
            tensor_bytes(&TensorType::new(
                [1, 8, 3, 64],
                ScalarType::BF16,
                crate::graph_ir::Layout::Plain
            ))
            .unwrap(),
            3072
        );
        assert!(tensor_bytes(&TensorType::new(
            [usize::MAX, 2],
            ScalarType::F32,
            crate::graph_ir::Layout::Plain
        ))
        .is_err());
    }
}
