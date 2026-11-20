#[path = "../../models/tts-with-rvc-rs/src/audio_io.rs"]
mod audio_io;
#[path = "../../models/tts-with-rvc-rs/src/preprocess.rs"]
mod preprocess;
#[path = "../../models/tts-with-rvc-rs/src/chunk_plan.rs"]
mod chunk_plan;
#[path = "../../models/tts-with-rvc-rs/src/chunked.rs"]
mod chunked;

use chunk_plan::ChunkStrategy;
use chunked::{ChunkConfig, planned_chunk_ranges};

fn measured_energy(pcm: &[f32], position: usize) -> f64 {
    let center = (position * 160).min(pcm.len());
    let low = center.saturating_sub(160);
    let high = center.saturating_add(160).min(pcm.len());
    pcm[low..high].iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / (high - low) as f64
}

fn print_plan(pcm: &[f32], label: &str, config: ChunkConfig, strategy: ChunkStrategy) {
    let chunks = planned_chunk_ranges(pcm, config, strategy).expect("plan");
    let total_frames: usize = chunks.iter().map(|(start, end)| end - start).sum();
    let model_frames = total_frames + chunks.len() * (2 * config.context_frames + config.trailing_frames);
    let mut invocations = 0;
    let mut i = 0;
    let mut distinct_lengths = std::collections::HashSet::new();
    while i < chunks.len() {
        let length = chunks[i].1 - chunks[i].0;
        distinct_lengths.insert(length);
        let group_size = if i + 1 < chunks.len()
            && chunks[i + 1].1 - chunks[i + 1].0 == length { 2 } else { 1 };
        i += group_size;
        invocations += 1;
    }
    let boundary_rms = chunks.iter().take(chunks.len().saturating_sub(1))
        .map(|&(_, end)| measured_energy(pcm, end))
        .sum::<f64>() / chunks.len().saturating_sub(1).max(1) as f64;
    println!("{{\"strategy\":\"{label}\",\"core_frames\":{},\"chunks\":{},\"model_input_frames\":{},\"model_input_seconds\":{:.3},\"model_invocations_at_batch2\":{},\"distinct_chunk_sizes\":{},\"mean_boundary_energy\":{:.10},\"boundaries\":{:?}}}",
        config.core_frames, chunks.len(), model_frames, model_frames as f64 / 100.0,
        invocations, distinct_lengths.len(), boundary_rms,
        chunks.iter().take(chunks.len().saturating_sub(1)).map(|x|x.1).collect::<Vec<_>>());
}

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: measure_real INPUT.wav".to_owned())?;
    let wave = audio_io::read_mono_wav(path)?;
    let pcm = audio_io::resample(&wave.samples, wave.sample_rate, 16_000)?;
    let filtered = preprocess::highpass_48hz_zero_phase(&pcm)?;
    println!("{{\"original_rate\":{},\"original_samples\":{},\"filtered_samples\":{},\"filtered_seconds\":{:.4}}}",
        wave.sample_rate, wave.samples.len(), filtered.len(), filtered.len() as f64 / 16000.0);
    print_plan(&filtered, "fixed-300", ChunkConfig::default(), ChunkStrategy::FixedOverlap);
    print_plan(&filtered, "fixed-900", ChunkConfig { core_frames: 900, ..ChunkConfig::default() }, ChunkStrategy::FixedOverlap);
    print_plan(&filtered, "quiet-splice-900", ChunkConfig { core_frames: 900, ..ChunkConfig::default() }, ChunkStrategy::QuietSplice { extra_chunks: 0 });
    for extra in 1..=2 {
        print_plan(&filtered, &format!("quiet-splice-900-plus{extra}"), ChunkConfig { core_frames: 900, ..ChunkConfig::default() }, ChunkStrategy::QuietSplice { extra_chunks: extra });
    }
    Ok(())
}
