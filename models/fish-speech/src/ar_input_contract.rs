use std::collections::{BTreeMap, BTreeSet};

use fish_speech_core::{
    ar_stream_executor::tensor_bytes,
    ar_stream_plan::{ArStreamPhase, ArStreamPlan},
    graph_ir::TensorType,
    trtx_blueprint::{Binding, TrtPrimitive},
};

use crate::{
    ar_weight_bindings::required_ar_weights,
    weights::{Result, WeightError},
};

pub fn collect_ar_weight_types(plans: &[&ArStreamPlan]) -> Result<BTreeMap<String, TensorType>> {
    if plans.is_empty() {
        return Err(WeightError::Invalid("no TensorRT AR plans supplied".into()));
    }
    let mut all = BTreeMap::new();
    for plan in plans {
        for (name, ty) in required_ar_weights(plan)? {
            if let Some(previous) = all.insert(name.clone(), ty.clone()) {
                if previous != ty {
                    return Err(WeightError::Invalid(format!(
                        "TensorRT AR weight shape conflicts across plans: {name}"
                    )));
                }
            }
        }
    }
    Ok(all)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArInputContract {
    pub weights: BTreeMap<String, TensorType>,
    pub runtime: BTreeMap<String, TensorType>,
}

fn register_input(
    target: &mut BTreeMap<String, TensorType>,
    name: &str,
    ty: &TensorType,
) -> Result<()> {
    match target.get(name) {
        Some(previous) if previous != ty => Err(WeightError::Invalid(format!(
            "AR input type conflicts across segments: {name}"
        ))),
        Some(_) => Ok(()),
        None => {
            target.insert(name.to_owned(), ty.clone());
            Ok(())
        }
    }
}

impl ArInputContract {
    pub fn from_plan(plan: &ArStreamPlan) -> Result<Self> {
        plan.validate()
            .map_err(|error| WeightError::Invalid(format!("AR stream plan: {error}")))?;
        let weights = required_ar_weights(plan)?;
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
        let mut produced = BTreeSet::<Binding>::new();
        let mut runtime = BTreeMap::new();
        for phase in &plan.phases {
            match phase {
                ArStreamPhase::Network(network) => {
                    for step in &network.steps {
                        if produced.contains(&step.result) {
                            continue;
                        }
                        match step.primitive {
                            TrtPrimitive::ConstantWeight => {
                                if !weights.contains_key(&step.name) {
                                    return Err(WeightError::Invalid(format!(
                                        "AR weight missing from plan: {}",
                                        step.name
                                    )));
                                }
                            }
                            TrtPrimitive::Input if !cache_names.contains(step.name.as_str()) => {
                                register_input(&mut runtime, &step.name, &step.output_type)?;
                            }
                            _ => {}
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
        if let Some(name) = weights.keys().find(|key| runtime.contains_key(*key)) {
            return Err(WeightError::Invalid(format!(
                "AR runtime input shadows checkpoint weight: {name}"
            )));
        }
        Ok(Self { weights, runtime })
    }

    pub fn bind<B: Clone>(
        &self,
        uploaded_weights: &BTreeMap<String, B>,
        runtime_inputs: &BTreeMap<String, B>,
        bytes: impl Fn(&B) -> usize,
    ) -> Result<BTreeMap<String, B>> {
        if let Some(name) = runtime_inputs
            .keys()
            .find(|name| !self.runtime.contains_key(*name))
        {
            return Err(WeightError::Invalid(format!(
                "unexpected runtime AR input: {name}"
            )));
        }
        let mut external = BTreeMap::new();
        for (expected, supplied) in [
            (&self.weights, uploaded_weights),
            (&self.runtime, runtime_inputs),
        ] {
            for (name, ty) in expected {
                let buffer = supplied.get(name).ok_or_else(|| {
                    WeightError::Invalid(format!("missing required AR input: {name}"))
                })?;
                let required = tensor_bytes(ty).map_err(|_| {
                    WeightError::Invalid(format!("AR input has invalid shape: {name}"))
                })?;
                if required == 0 || bytes(buffer) < required {
                    return Err(WeightError::Invalid(format!(
                        "undersized AR device input: {name}"
                    )));
                }
                external.insert(name.clone(), buffer.clone());
            }
        }
        Ok(external)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fish_speech_core::{config::FishModelConfig, transformer_ir::TransformerIrFactory};

    fn slow_plan() -> ArStreamPlan {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        ArStreamPlan::from_graph(
            &TransformerIrFactory::new(&config)
                .unwrap()
                .slow_prefill(4)
                .unwrap(),
        )
        .unwrap()
    }

    fn buffers(
        contract: &ArInputContract,
    ) -> (BTreeMap<String, Vec<u8>>, BTreeMap<String, Vec<u8>>) {
        let allocate = |items: &BTreeMap<String, TensorType>| {
            items
                .iter()
                .map(|(key, ty)| (key.clone(), vec![0; tensor_bytes(ty).unwrap()]))
                .collect::<BTreeMap<_, _>>()
        };
        (allocate(&contract.weights), allocate(&contract.runtime))
    }

    #[test]
    fn separates_model_weights_and_external_inputs_from_owned_kv() {
        let contract = ArInputContract::from_plan(&slow_plan()).unwrap();
        assert!(contract.weights.contains_key("embeddings.weight"));
        assert!(!contract.runtime.is_empty());
        assert!(
            contract
                .runtime
                .keys()
                .all(|name| !name.ends_with("_cache"))
        );
        assert!(
            contract
                .runtime
                .keys()
                .all(|name| !contract.weights.contains_key(name))
        );
        let (weights, runtime) = buffers(&contract);
        assert_eq!(
            contract.bind(&weights, &runtime, Vec::len).unwrap().len(),
            weights.len() + runtime.len()
        );
    }

    #[test]
    fn rejects_missing_or_short_inputs_before_device_submission() {
        let contract = ArInputContract::from_plan(&slow_plan()).unwrap();
        let (weights, runtime) = buffers(&contract);
        let missing = runtime.keys().next().unwrap().clone();
        let mut incomplete = runtime.clone();
        incomplete.remove(&missing);
        assert!(contract.bind(&weights, &incomplete, Vec::len).is_err());
        let mut short = runtime.clone();
        short.get_mut(&missing).unwrap().truncate(1);
        assert!(contract.bind(&weights, &short, Vec::len).is_err());
        let mut unknown = runtime;
        unknown.insert("not_an_ar_input".into(), vec![0; 4]);
        assert!(contract.bind(&weights, &unknown, Vec::len).is_err());
    }

    #[test]
    fn plan_set_reuses_shared_weights_and_deduplicates_transitions() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&config).unwrap();
        let prefill = ArStreamPlan::from_graph(&factory.slow_prefill(4).unwrap()).unwrap();
        let decode = ArStreamPlan::from_graph(&factory.slow_decode(4).unwrap()).unwrap();
        let single = ArInputContract::from_plan(&prefill).unwrap().weights;
        let other = ArInputContract::from_plan(&decode).unwrap().weights;
        let combined = collect_ar_weight_types(&[&prefill, &decode]).unwrap();
        assert!(combined.len() >= single.len());
        assert!(combined.len() <= single.len() + other.len());
        assert!(single.keys().all(|name| combined.contains_key(name)));
        assert!(collect_ar_weight_types(&[]).is_err());
    }
}
