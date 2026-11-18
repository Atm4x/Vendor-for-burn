use fish_speech_core::config::FishModelConfig;
use fish_speech_core::firefly_ir::FireflyGraph;
use fish_speech_core::firefly_trtx_conv::plan_firefly_convolutions;
use fish_speech_core::firefly_trtx_fsq::{FsqDecodeLookupPlan, FsqEncodePlan};
use fish_speech_core::firefly_trtx_mask::FireflyMaskPlan;
use fish_speech_core::firefly_trtx_pointwise::plan_pointwise;
use fish_speech_core::firefly_trtx_segment::structural_coverage;
use fish_speech_core::graph::{MockBackend, PlanFactory};
use fish_speech_core::shape_profiles::ArProfiles;
use fish_speech_core::transformer_ir::TransformerIrFactory;
use fish_speech_core::trtx_blueprint::TrtBlueprint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("--firefly-weight-shapes") {
        for (stage, graph) in [
            ("reference", FireflyGraph::reference_encode(1, 8192)?),
            ("waveform", FireflyGraph::waveform_decode(1, 8, 8)?),
        ] {
            for (key, ty) in graph.weights {
                let shape = ty
                    .shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                println!("{stage}\t{key}\t{shape}");
            }
        }
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("--firefly-trt-coverage") {
        for (stage, graph) in [
            ("reference", FireflyGraph::reference_encode(1, 8192)?),
            ("waveform", FireflyGraph::waveform_decode(1, 8, 8)?),
        ] {
            let plans = plan_firefly_convolutions(&graph)?;
            let (covered, total, blocked) = structural_coverage(&graph)?;
            let pointwise = plan_pointwise(&graph)?.len();
            let masks = (0..total)
                .filter(|&i| {
                    FireflyMaskPlan::from_node(&graph, i)
                        .ok()
                        .flatten()
                        .is_some()
                })
                .count();
            let fsq_decodes = (0..total)
                .filter(|&i| {
                    FsqDecodeLookupPlan::from_node(&graph, i)
                        .ok()
                        .flatten()
                        .is_some()
                })
                .count();
            let fsq_encodes = (0..total)
                .filter(|&i| FsqEncodePlan::from_node(&graph, i).ok().flatten().is_some())
                .count();
            assert_eq!(
                covered,
                plans.len() + pointwise + masks + fsq_decodes + fsq_encodes
            );
            println!(
                "{stage} nodes={total} trt_conv2d={} trt_pointwise={} trt_mask={} trt_fsq_encode={} trt_fsq_decode={} structurally_lowerable={covered} deferred={} weights={} status=FFI_COMPILED_GPU_UNTESTED",
                plans.len(), pointwise, masks, fsq_encodes, fsq_decodes, blocked.len(), graph.weights.len()
            );
            for (_, operation) in blocked {
                println!("{stage} DEFERRED {operation}");
            }
        }
        return Ok(());
    }
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
    for (stage, graph) in [
        ("reference_encode", FireflyGraph::reference_encode(1, 8192)?),
        ("waveform_decode", FireflyGraph::waveform_decode(1, 8, 8)?),
    ] {
        println!(
            "FIREFLY_TYPED_DAG stage={} nodes={} weights={} valid_frames={} (CPU_SHAPES_ONLY)",
            stage,
            graph.nodes.len(),
            graph.weights.len(),
            graph.valid_frames
        );
    }
    let profiles = ArProfiles::from_config(&config, 32)?;
    println!("AR_PROFILES slow_prefill={} slow_decode={} fast_decode={} (BUCKET_PLAN_ONLY; GPU_UNTESTED)",
        profiles.slow_prefill.buckets.len(), profiles.slow_decode.buckets.len(),
        profiles.fast_decode.as_ref().map_or(0, |p| p.buckets.len()));
    println!("STRUCTURAL_AND_CPU_CONTRACTS_PASS; TRT_ENGINE_UNBUILT; GPU_INFERENCE_NOT_TESTED; MODEL_WEIGHTS_UNAVAILABLE");
    Ok(())
}
