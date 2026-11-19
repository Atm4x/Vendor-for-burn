use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::error::{FishError, Result};
use crate::graph_ir::{Graph, Op, ScalarType, TensorType};
use crate::kv_device::{Bf16KvLayout, KvWrite2d};
use crate::stream_binding::Binding;
use crate::trtx_blueprint::{TrtBlueprint, TrtPrimitive, TrtStep};

/// A backend-independent view of one executable network segment. The engine
/// implementation owns its graph representation and validates internal nodes.
/// The stream executor only requires the named external inputs and typed outputs.
pub trait ArNetworkSegment {
    fn validate_segment(&self) -> Result<()>;
    fn external_inputs(&self) -> Vec<(Binding, &str)>;
    fn outputs(&self) -> Result<Vec<(Binding, &TensorType)>>;
}

impl ArNetworkSegment for TrtBlueprint {
    fn validate_segment(&self) -> Result<()> {
        self.validate()?;
        self.require_no_deferred_ops()
    }

    fn external_inputs(&self) -> Vec<(Binding, &str)> {
        self.steps
            .iter()
            .filter_map(|step| {
                matches!(
                    step.primitive,
                    TrtPrimitive::Input | TrtPrimitive::ConstantWeight
                )
                .then_some((step.result, step.name.as_str()))
            })
            .collect()
    }

    fn outputs(&self) -> Result<Vec<(Binding, &TensorType)>> {
        self.outputs
            .iter()
            .map(|binding| {
                self.steps
                    .iter()
                    .find(|step| step.result == *binding)
                    .map(|step| (*binding, &step.output_type))
                    .ok_or(FishError::InvalidShape(
                        "AR output missing tensor definition",
                    ))
            })
            .collect()
    }
}

impl<N: ArNetworkSegment> ArNetworkSegment for Arc<N> {
    fn validate_segment(&self) -> Result<()> {
        self.as_ref().validate_segment()
    }
    fn external_inputs(&self) -> Vec<(Binding, &str)> {
        self.as_ref().external_inputs()
    }
    fn outputs(&self) -> Result<Vec<(Binding, &TensorType)>> {
        self.as_ref().outputs()
    }
}

#[derive(Debug, Clone)]
pub struct KvTransfer {
    pub source: Binding,
    pub cache: Binding,
    pub cache_name: String,
    pub updated_alias: Binding,
    pub geometry: KvWrite2d,
}

#[derive(Debug, Clone)]
pub struct KvWriteBarrier {
    pub layer: String,
    pub position: usize,
    pub key: KvTransfer,
    pub value: KvTransfer,
}

#[derive(Debug, Clone)]
pub enum ArStreamPhase<N = Arc<TrtBlueprint>> {
    Network(N),
    KvWrites(KvWriteBarrier),
}

#[derive(Debug, Clone)]
pub struct ArStreamPlan<N = Arc<TrtBlueprint>> {
    pub phases: Vec<ArStreamPhase<N>>,
    pub cache_aliases: BTreeMap<Binding, Binding>,
    pub final_outputs: Vec<Binding>,
}

impl ArStreamPlan<Arc<TrtBlueprint>> {
    pub fn from_graph(graph: &Graph) -> Result<Self> {
        let original = TrtBlueprint::from_graph(graph)?;
        let mut catalogue = BTreeMap::<Binding, TrtStep>::new();
        for step in &original.steps {
            catalogue.insert(step.result, step.clone());
        }
        let computational = original
            .steps
            .iter()
            .filter(|step| {
                !matches!(
                    step.primitive,
                    TrtPrimitive::Input | TrtPrimitive::ConstantWeight
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut raw = Vec::<RawPhase>::new();
        let mut current = Vec::<TrtStep>::new();
        let mut aliases = BTreeMap::new();
        let mut next_temp = original
            .steps
            .iter()
            .filter_map(|step| match step.result {
                Binding::Temp(id) => Some(id),
                _ => None,
            })
            .max()
            .map_or(0, |max| max + 1);
        let mut index = 0;
        while index < computational.len() {
            let step = &computational[index];
            match step.primitive {
                TrtPrimitive::Deferred {
                    original: Op::KvUpdate { position },
                    ..
                } => {
                    let other = computational.get(index + 1).ok_or(FishError::InvalidShape(
                        "AR KV update must have adjacent K/V operations",
                    ))?;
                    if !matches!(other.primitive, TrtPrimitive::Deferred { original: Op::KvUpdate { position: p }, .. } if p == position)
                    {
                        return Err(FishError::InvalidShape(
                            "AR KV update K/V must be adjacent at same position",
                        ));
                    }
                    let key_layer = step
                        .name
                        .strip_suffix(".k_updated")
                        .ok_or(FishError::InvalidShape("AR KV key update name mismatch"))?;
                    let value_layer = other
                        .name
                        .strip_suffix(".v_updated")
                        .ok_or(FishError::InvalidShape("AR KV value update name mismatch"))?;
                    if key_layer != value_layer || key_layer.is_empty() || current.is_empty() {
                        return Err(FishError::InvalidShape(
                            "AR KV updates do not belong to same projected layer",
                        ));
                    }
                    let key =
                        create_transfer(step, position, &catalogue, &mut next_temp, &mut current)?;
                    let value =
                        create_transfer(other, position, &catalogue, &mut next_temp, &mut current)?;
                    aliases.insert(key.updated_alias, key.cache);
                    aliases.insert(value.updated_alias, value.cache);
                    raw.push(RawPhase::Network(std::mem::take(&mut current)));
                    raw.push(RawPhase::Writes(KvWriteBarrier {
                        layer: key_layer.to_owned(),
                        position,
                        key,
                        value,
                    }));
                    index += 2;
                }
                TrtPrimitive::Deferred { .. } => {
                    return Err(FishError::Unsupported(
                        "AR stream split supports only external KV update barriers",
                    ));
                }
                _ => {
                    current.push(step.clone());
                    index += 1;
                }
            }
        }
        if current.is_empty() {
            return Err(FishError::InvalidShape(
                "AR stream graph has no terminal network",
            ));
        }
        raw.push(RawPhase::Network(current));
        let mut all_types = catalogue
            .iter()
            .map(|(binding, step)| {
                (
                    *binding,
                    (
                        step.name.clone(),
                        step.output_type.clone(),
                        matches!(step.primitive, TrtPrimitive::ConstantWeight),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for phase in &raw {
            if let RawPhase::Network(steps) = phase {
                for step in steps {
                    all_types.insert(
                        step.result,
                        (step.name.clone(), step.output_type.clone(), false),
                    );
                }
            }
        }
        let mut phases = Vec::new();
        for (phase_index, phase) in raw.iter().enumerate() {
            match phase {
                RawPhase::Writes(batch) => phases.push(ArStreamPhase::KvWrites(batch.clone())),
                RawPhase::Network(steps) => {
                    let produced = steps.iter().map(|s| s.result).collect::<BTreeSet<_>>();
                    let external = steps
                        .iter()
                        .flat_map(|step| step.arguments.iter().copied())
                        .filter(|id| !produced.contains(id))
                        .collect::<BTreeSet<_>>();
                    let mut local = Vec::new();
                    let mut weight_count = 0;
                    for id in external {
                        let (name, ty, is_weight) = all_types
                            .get(&id)
                            .ok_or(FishError::InvalidShape("unresolved AR boundary tensor"))?;
                        let primitive = if *is_weight {
                            weight_count += 1;
                            TrtPrimitive::ConstantWeight
                        } else {
                            TrtPrimitive::Input
                        };
                        local.push(TrtStep {
                            name: name.clone(),
                            primitive,
                            arguments: Vec::new(),
                            result: id,
                            output_type: ty.clone(),
                        });
                    }
                    local.extend_from_slice(steps);
                    let mut used_later = original.outputs.iter().copied().collect::<BTreeSet<_>>();
                    for next in raw.iter().skip(phase_index + 1) {
                        match next {
                            RawPhase::Network(next_steps) => {
                                used_later.extend(
                                    next_steps.iter().flat_map(|s| s.arguments.iter().copied()),
                                );
                            }
                            RawPhase::Writes(write) => {
                                used_later.insert(write.key.source);
                                used_later.insert(write.value.source);
                            }
                        }
                    }
                    let outputs = produced.intersection(&used_later).copied().collect();
                    let subgraph = TrtBlueprint {
                        steps: local,
                        outputs,
                        deferred: BTreeMap::new(),
                        weight_count,
                    };
                    subgraph.validate()?;
                    phases.push(ArStreamPhase::Network(subgraph.into()));
                }
            }
        }
        let plan = Self {
            phases,
            cache_aliases: aliases,
            final_outputs: original.outputs,
        };
        plan.validate()?;
        Ok(plan)
    }
}

impl<N: ArNetworkSegment> ArStreamPlan<N> {
    pub fn validate(&self) -> Result<()> {
        if self.phases.len() < 3 || self.phases.len() % 2 == 0 {
            return Err(FishError::InvalidShape("AR stream phase order invalid"));
        }
        let mut physical_writes = BTreeSet::<Binding>::new();
        for (i, phase) in self.phases.iter().enumerate() {
            match phase {
                ArStreamPhase::Network(plan) if i % 2 == 0 => {
                    plan.validate_segment()?;
                }
                ArStreamPhase::KvWrites(batch) if i % 2 == 1 => {
                    if batch.key.geometry.position != batch.position
                        || batch.value.geometry.position != batch.position
                        || batch.key.geometry != batch.value.geometry
                        || batch.key.cache == batch.value.cache
                        || self.cache_aliases.get(&batch.key.updated_alias)
                            != Some(&batch.key.cache)
                        || self.cache_aliases.get(&batch.value.updated_alias)
                            != Some(&batch.value.cache)
                        || !physical_writes.insert(batch.key.cache)
                        || !physical_writes.insert(batch.value.cache)
                    {
                        return Err(FishError::InvalidShape("invalid AR KV write barrier"));
                    }
                    let ArStreamPhase::Network(before) = &self.phases[i - 1] else {
                        unreachable!()
                    };
                    for source in [batch.key.source, batch.value.source] {
                        let outputs = before.outputs()?;
                        if !outputs
                            .iter()
                            .any(|(binding, ty)| *binding == source && ty.dtype == ScalarType::BF16)
                        {
                            return Err(FishError::InvalidShape(
                                "KV write needs BF16 network output",
                            ));
                        }
                    }
                }
                _ => return Err(FishError::InvalidShape("AR stream phase kind mismatch")),
            }
        }
        if self.final_outputs.is_empty() {
            return Err(FishError::InvalidShape("AR stream has no final outputs"));
        }
        Ok(())
    }

    pub fn network_count(&self) -> usize {
        self.phases
            .iter()
            .filter(|p| matches!(p, ArStreamPhase::Network(_)))
            .count()
    }

    pub fn external_copy_count(&self) -> usize {
        self.phases
            .iter()
            .filter(|p| matches!(p, ArStreamPhase::KvWrites(_)))
            .count()
            * 2
    }
}

enum RawPhase {
    Network(Vec<TrtStep>),
    Writes(KvWriteBarrier),
}

fn create_transfer(
    step: &TrtStep,
    position: usize,
    catalogue: &BTreeMap<Binding, TrtStep>,
    next_temp: &mut usize,
    current: &mut Vec<TrtStep>,
) -> Result<KvTransfer> {
    if step.arguments.len() != 2 {
        return Err(FishError::InvalidShape(
            "KV update needs cache + projected K/V",
        ));
    }
    let cache = &catalogue
        .get(&step.arguments[0])
        .ok_or(FishError::InvalidShape("missing KV storage type"))?
        .output_type;
    let source = &catalogue
        .get(&step.arguments[1])
        .ok_or(FishError::InvalidShape("missing KV projection type"))?
        .output_type;
    if cache.shape.len() != 4
        || source.shape.len() != 4
        || cache.dtype != ScalarType::BF16
        || source.dtype != ScalarType::F32
        || cache.shape[0..2] != source.shape[0..2]
        || cache.shape[3] != source.shape[3]
    {
        return Err(FishError::InvalidShape("KV BF16/F32 layout mismatch"));
    }
    let layout = Bf16KvLayout {
        batch: cache.shape[0],
        kv_heads: cache.shape[1],
        capacity: cache.shape[2],
        head_dim: cache.shape[3],
    };
    let geometry = layout.write_plan(position, source.shape[2])?;
    let output = Binding::Temp(*next_temp);
    *next_temp += 1;
    let mut ty: TensorType = source.clone();
    ty.dtype = ScalarType::BF16;
    current.push(TrtStep {
        name: format!("{}.copy_source_bf16", step.name),
        primitive: TrtPrimitive::CastToBf16,
        arguments: vec![step.arguments[1]],
        result: output,
        output_type: ty,
    });
    Ok(KvTransfer {
        source: output,
        cache: step.arguments[0],
        cache_name: catalogue[&step.arguments[0]].name.clone(),
        updated_alias: step.result,
        geometry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FishModelConfig;
    use crate::transformer_ir::TransformerIrFactory;

    #[test]
    fn prefill_decode_and_fast_have_stream_ordered_kv_barriers() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&config).unwrap();
        for graph in [
            factory.slow_prefill(5).unwrap(),
            factory.slow_decode(6).unwrap(),
            factory.fast_decode(0).unwrap(),
            factory.fast_decode(2).unwrap(),
        ] {
            let plan = ArStreamPlan::from_graph(&graph).unwrap();
            assert_eq!(plan.network_count(), 3);
            assert_eq!(plan.external_copy_count(), 4);
            assert_eq!(plan.cache_aliases.len(), 4);
            plan.validate().unwrap();
            for phase in &plan.phases {
                if let ArStreamPhase::KvWrites(writes) = phase {
                    assert_eq!(writes.key.geometry.position, writes.position);
                    assert_eq!(
                        writes.key.geometry.row_bytes,
                        writes.value.geometry.row_bytes
                    );
                    assert_eq!(writes.key.geometry.rows, 2);
                }
            }
        }
    }

    #[test]
    fn each_gpu_write_uses_only_new_frames_and_bf16_sources() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let graph = TransformerIrFactory::new(&config)
            .unwrap()
            .slow_decode(9)
            .unwrap();
        let plan = ArStreamPlan::from_graph(&graph).unwrap();
        for phase in &plan.phases {
            if let ArStreamPhase::KvWrites(batch) = phase {
                assert_eq!(batch.key.geometry.frames, 1);
                assert_eq!(batch.key.geometry.row_bytes, config.slow.head_dim() * 2);
                assert_eq!(batch.key.geometry.position, 8);
                assert_eq!(
                    batch.key.geometry.destination_pitch_bytes,
                    config.max_seq_len * config.slow.head_dim() * 2
                );
            }
        }
    }

    #[test]
    fn source_layout_is_never_treated_as_bf16_before_cast() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let mut graph = TransformerIrFactory::new(&config)
            .unwrap()
            .slow_prefill(3)
            .unwrap();
        let first = graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::KvUpdate { .. }))
            .unwrap()
            .inputs[1];
        graph.values[first.0].ty.dtype = ScalarType::I32;
        assert!(ArStreamPlan::from_graph(&graph).is_err());
    }
}
