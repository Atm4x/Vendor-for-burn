use fish_speech_vendor::reference_loader::ReferenceAudioLoader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: fish-reference-inspect <audio-file> [--mel]")?;
    let mel = args.next().is_some_and(|arg| arg == "--mel");
    let pcm = ReferenceAudioLoader::default().load_path(&path)?;
    println!(
        "source_rate={} source_channels={} output_rate={} frames={}",
        pcm.source_rate,
        pcm.source_channels,
        pcm.sample_rate,
        pcm.pcm.len()
    );
    if mel {
        let mels = pcm.prepare_mel()?;
        println!(
            "mels={}x{} valid_code_frames={}",
            mels.mels.channels, mels.mels.frames, mels.lengths.valid_code_frames
        );
    }
    Ok(())
}
