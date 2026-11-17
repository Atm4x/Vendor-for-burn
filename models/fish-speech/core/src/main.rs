use fish_speech_core::config::FishModelConfig;
use fish_speech_core::graph::{MockBackend, PlanFactory};
use fish_speech_core::shape_profiles::ArProfiles;
use fish_speech_core::transformer_ir::TransformerIrFactory;
use fish_speech_core::trtx_blueprint::TrtBlueprint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = FishModelConfig::synthetic_dual_ar_fixture();
    let factory = PlanFactory::new(&config)?;
    let plans = [
        factory.slow_prefill(32)?,
        factory.slow_decode(33)?,
        factory.fast_decode(0)?,
        factory.firefly_encode(127)?,
        factory.firefly_decode(32)?,
    ];
    let mock = MockBackend;
    for plan in plans {
        mock.validate_plan(&plan)?;
        println!(
            "STRUCTURAL {:?}: inputs={}, outputs={}, ops={}, kv_len={:?}",
            plan.stage,
            plan.inputs.len(),
            plan.outputs.len(),
            plan.operations.len(),
            plan.active_kv_len
        );
    }
    let ir_factory = TransformerIrFactory::new(&config)?;
    for graph in [
        ir_factory.slow_prefill(32)?,
        ir_factory.slow_decode(33)?,
        ir_factory.fast_decode(0)?,
    ] {
        let audit = graph.audit()?;
        let blueprint = TrtBlueprint::from_graph(&graph)?;
        assert!(blueprint.require_no_deferred_ops().is_err());
        println!(
            "TRTX_BLUEPRINT {:?}: steps={}, deferred_ops={}, deferred_reasons={:?}",
            graph.stage,
            blueprint.steps.len(),
            blueprint.deferred.values().sum::<usize>(),
            blueprint.deferred
        );
        println!(
            "WIRED_IR {:?}: nodes={}, weights={}, inputs={}, outputs={}, lowering={:?}",
            graph.stage,
            graph.nodes.len(),
            audit.weight_count,
            audit.input_count,
            audit.output_count,
            audit.counts
        );
    }
    let profiles = ArProfiles::from_config(&config, 32)?;
    println!("AR_PROFILES slow_prefill={} slow_decode={} fast_decode={} (BUCKET_PLAN_ONLY; GPU_UNTESTED)",
        profiles.slow_prefill.buckets.len(), profiles.slow_decode.buckets.len(),
        profiles.fast_decode.as_ref().map_or(0, |p| p.buckets.len()));
    println!("STRUCTURAL_AND_CPU_CONTRACTS_PASS; TRT_API_UNBUILT; GPU_INFERENCE_NOT_TESTED; MODEL_WEIGHTS_UNAVAILABLE");
    Ok(())
}
