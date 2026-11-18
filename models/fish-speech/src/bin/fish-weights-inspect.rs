use std::env;
use std::path::Path;

use fish_speech_core::firefly_ir::FireflyGraph;
use fish_speech_vendor::weights::LoadedWeights;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 3 || args.len() > 4 || (args.len() == 4 && args[3] != "--validate-firefly") {
        return Err(
            "usage: fish-weights-inspect MODEL_DIRECTORY MANIFEST_JSON [--validate-firefly]".into(),
        );
    }
    let weights = LoadedWeights::open(Path::new(&args[1]), Path::new(&args[2]))?;
    let manifest = weights.manifest();
    println!(
        "SAFE_CHECKPOINT_VALID component={} tensors={} shards={} model={}",
        manifest.component,
        manifest.tensors.len(),
        manifest.shards.len(),
        manifest.model_family
    );
    if args.len() == 4 {
        if manifest.component != "firefly" {
            return Err("--validate-firefly requires a Firefly checkpoint".into());
        }
        let reference = FireflyGraph::reference_encode(1, 8192)?;
        let waveform = FireflyGraph::waveform_decode(1, 8, 8)?;
        let encoder = reference.weights.len();
        let decoder = waveform.weights.len();
        let mut requirements = reference.weights;
        for (name, ty) in waveform.weights {
            if let Some(existing) = requirements.insert(name.clone(), ty.clone()) {
                if existing != ty {
                    return Err(format!("conflicting Firefly graph requirement: {name}").into());
                }
            }
        }
        weights.validate_exact_requirements(&requirements)?;
        println!(
            "FIREFLY_WEIGHT_SHAPES_VALID encode={} decode={}",
            encoder, decoder
        );
    }
    println!("VALIDATION_ONLY; GPU_INFERENCE_NOT_TESTED");
    Ok(())
}
