#[path = "../../models/tts-with-rvc-rs/src/audio_io.rs"]
mod audio_io;
#[path = "../../models/tts-with-rvc-rs/src/preprocess.rs"]
mod preprocess;
#[path = "../../models/tts-with-rvc-rs/src/chunk_plan.rs"]
mod chunk_plan;
#[path = "../../models/tts-with-rvc-rs/src/chunked.rs"]
mod chunked;

use chunk_plan::{ChunkStrategy, CostModel};
use chunked::{ChunkConfig, planned_chunk_ranges};

fn report(pcm: &[f32], label: &str, config: ChunkConfig, strategy: ChunkStrategy) {
    let plan = planned_chunk_ranges(pcm, config, strategy).expect("plan");
    let mut model_frames=0usize;
    let mut quadratic=0u64;
    let mut cut_rms=0f64;
    for &(start,end) in &plan {
        let span=end-start+2*config.context_frames+config.trailing_frames;
        model_frames+=span;
        quadratic+=(span as u64)*(span as u64);
        if end < pcm.len().div_ceil(160) {
            let low=(end*160).saturating_sub(160);
            let high=(end*160+160).min(pcm.len());
            cut_rms+=pcm[low..high].iter().map(|&x|(x as f64).powi(2)).sum::<f64>()/(high-low) as f64;
        }
    }
    cut_rms/=plan.len().saturating_sub(1).max(1) as f64;
    println!("{{\"plan\":\"{label}\",\"chunks\":{},\"model_input_seconds\":{:.3},\"attention_area\":{},\"mean_boundary_energy\":{:.10},\"boundaries\":{:?}}}",
        plan.len(),model_frames as f64/100.0,quadratic,cut_rms,
        plan.iter().take(plan.len().saturating_sub(1)).map(|x|x.1).collect::<Vec<_>>());
}

fn main() -> Result<(), String> {
    let path=std::env::args().nth(1).ok_or("usage: measure_real input.wav".to_owned())?;
    let wave=audio_io::read_mono_wav(path)?;
    let pcm=audio_io::resample(&wave.samples,wave.sample_rate,16_000)?;
    let filtered=preprocess::highpass_48hz_zero_phase(&pcm)?;
    println!("{{\"input_hz\":{},\"pcm_samples\":{},\"frames\":{}}}",wave.sample_rate,filtered.len(),filtered.len().div_ceil(160));
    let config=ChunkConfig {core_frames:900,..ChunkConfig::default()};
    report(&filtered,"fixed300",ChunkConfig::default(),ChunkStrategy::FixedOverlap);
    report(&filtered,"fixed900",config,ChunkStrategy::FixedOverlap);
    report(&filtered,"quiet900extra0",config,ChunkStrategy::QuietSplice{extra_chunks:0});
    report(&filtered,"quiet900extra1",config,ChunkStrategy::QuietSplice{extra_chunks:1});
    for (name,model) in [
        ("cost-default-proxy",CostModel::default()),
        ("cost-launch-dominated",CostModel{launch:100_000,linear:10,attention_milli:10,splice:100}),
        ("cost-attention-dominated",CostModel{launch:10_000,linear:10,attention_milli:100,splice:100}),
        ("cost-cut-sensitive",CostModel{launch:10_000,linear:10,attention_milli:10,splice:10_000}),
    ] {report(&filtered,name,config,ChunkStrategy::CostAware(model));}
    Ok(())
}
