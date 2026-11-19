use std::collections::BTreeMap;

use fish_speech_core::{
    ar_stream_plan::{ArStreamPhase, ArStreamPlan},
    graph_ir::{ScalarType, TensorType},
    trtx_blueprint::TrtPrimitive,
};

use crate::weights::{Result, WeightError};

pub fn required_ar_weights(plan: &ArStreamPlan) -> Result<BTreeMap<String, TensorType>> {
    plan.validate()
        .map_err(|e| WeightError::Invalid(format!("AR stream plan: {e}")))?;
    let mut required = BTreeMap::new();
    for phase in &plan.phases {
        if let ArStreamPhase::Network(network) = phase {
            for step in &network.steps {
                if step.primitive != TrtPrimitive::ConstantWeight {
                    continue;
                }
                if !matches!(step.output_type.dtype, ScalarType::F32 | ScalarType::BF16) {
                    return Err(WeightError::Invalid(format!(
                        "unsupported TensorRT AR weight dtype: {}",
                        step.name
                    )));
                }
                if let Some(previous) = required.insert(step.name.clone(), step.output_type.clone())
                {
                    if previous != step.output_type {
                        return Err(WeightError::Invalid(format!(
                            "inconsistent weight shape across AR segments: {}",
                            step.name
                        )));
                    }
                }
            }
        }
    }
    if required.is_empty() {
        return Err(WeightError::Invalid(
            "AR plan has no static weight bindings".into(),
        ));
    }
    Ok(required)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fish_speech_core::{
        config::{FishModelConfig, ModelKind},
        transformer_ir::TransformerIrFactory,
    };

    #[test]
    fn slow_fast_and_naive_plans_collect_stable_weight_bindings() {
        let dual = FishModelConfig::synthetic_dual_ar_fixture();
        let factory = TransformerIrFactory::new(&dual).unwrap();
        let slow = ArStreamPlan::from_graph(&factory.slow_prefill(4).unwrap()).unwrap();
        let slow_weights = required_ar_weights(&slow).unwrap();
        assert_eq!(slow_weights["embeddings.weight"].shape, vec![1024, 64]);
        assert_eq!(
            slow_weights["layers.0.attention.wqkv.weight"].shape,
            vec![96, 64]
        );
        let fast = ArStreamPlan::from_graph(&factory.fast_decode(0).unwrap()).unwrap();
        let fast_weights = required_ar_weights(&fast).unwrap();
        assert_eq!(
            fast_weights["fast_layers.0.attention.wqkv.weight"].shape,
            vec![96, 64]
        );
        let mut naive = dual.clone();
        naive.kind = ModelKind::Naive;
        naive.fast = None;
        let factory = TransformerIrFactory::new(&naive).unwrap();
        let plan = ArStreamPlan::from_graph(&factory.slow_prefill(4).unwrap()).unwrap();
        let naive_weights = required_ar_weights(&plan).unwrap();
        assert!(naive_weights.contains_key("codebook_output.weight"));
        assert!(!naive_weights.contains_key("fast_layers.0.attention.wqkv.weight"));
    }
}
