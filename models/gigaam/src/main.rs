use anyhow::{Context, Result, bail};
use burn::tensor::Device;
use gigaam_rs::{GigaAm, audio};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 && args[1] == "probe-audio" {
        let samples = audio::load_wav_16k(Path::new(&args[2]))?;
        let pieces = audio::speech_chunks(&samples, 24 * audio::SAMPLE_RATE as usize);
        println!(
            "sample_rate={} samples={} duration={:.3} chunks={}",
            audio::SAMPLE_RATE,
            samples.len(),
            samples.len() as f64 / audio::SAMPLE_RATE as f64,
            pieces.len()
        );
        for (i, chunk) in pieces.iter().enumerate() {
            let feat = audio::log_mel(chunk, 64)?;
            println!(
                "chunk={i} samples={} mel_shape=[{},{}] finite={}",
                chunk.len(),
                feat.n_mels,
                feat.frames,
                feat.values.iter().all(|x| x.is_finite())
            );
        }
        return Ok(());
    }
    if args.len() < 4
        || !["transcribe", "parity", "parity-encoder", "embed", "emotion"]
            .contains(&args[1].as_str())
    {
        eprintln!("Usage: gigaam-rs transcribe <model_dir> <audio.wav> [--device cpu|cuda|wgpu]");
        eprintln!("       gigaam-rs probe-audio <audio.wav>");
        eprintln!("       gigaam-rs parity <ctc_model_dir> <audio.wav> <out_dir>");
        eprintln!("       gigaam-rs embed <ssl_model_dir> <audio.wav> <out_file.f32>");
        eprintln!("       gigaam-rs emotion <emo_model_dir> <audio.wav>");
        eprintln!("       gigaam-rs parity-encoder <ssl_model_dir> <audio.wav> <out_dir>");
        eprintln!("       gigaam-rs transcribe <model_dir> <audio.wav> --word-timestamps");
        std::process::exit(2);
    }
    let selected = args
        .windows(2)
        .find(|x| x[0] == "--device")
        .map(|x| x[1].as_str())
        .unwrap_or("cpu");
    let device = match selected {
        "cpu" => {
            #[cfg(feature = "flex")]
            {
                Device::flex()
            }
            #[cfg(all(not(feature = "flex"), feature = "cpu"))]
            {
                Device::cpu()
            }
            #[cfg(all(not(feature = "flex"), not(feature = "cpu")))]
            {
                bail!("build with --features flex or cpu")
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::cuda(0)
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("build with --features cuda")
            }
        }
        "wgpu" => {
            #[cfg(feature = "wgpu")]
            {
                Device::wgpu(Default::default())
            }
            #[cfg(not(feature = "wgpu"))]
            {
                bail!("build with --features wgpu")
            }
        }
        _ => bail!("unknown backend: {selected}"),
    };
    let model = GigaAm::load(Path::new(&args[2]), device).context("loading GigaAM weights")?;
    let samples = audio::load_wav_16k(Path::new(&args[3]))?;
    if args[1] == "emotion" {
        let probs = model.emotion_probs_16k(&samples)?;
        for (label, prob) in probs {
            println!("{label}: {prob:.6}");
        }
        return Ok(());
    }
    if args[1] == "parity" {
        if args.len() < 5 {
            bail!("parity requires <out_dir>")
        }
        let out = Path::new(&args[4]);
        std::fs::create_dir_all(out)?;
        let features = audio::log_mel(&samples, model.n_mels())?;
        let (logits, time) = model.infer_logits_16k(&samples)?;
        let mel: Vec<u8> = features
            .values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let values: Vec<u8> = logits.into_iter().flat_map(f32::to_le_bytes).collect();
        std::fs::write(out.join("mel.f32"), mel)?;
        std::fs::write(out.join("logits.f32"), values)?;
        println!("parity frames={time} output={}", out.display());
        return Ok(());
    }
    if args[1] == "parity-encoder" {
        if args.len() < 5 {
            bail!("parity-encoder requires <out_dir>")
        }
        let out = Path::new(&args[4]);
        std::fs::create_dir_all(out)?;
        let features = audio::log_mel(&samples, model.n_mels())?;
        let (encoded, frames, channels) = model.infer_embeddings_16k(&samples)?;
        std::fs::write(
            out.join("mel.f32"),
            features
                .values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<u8>>(),
        )?;
        std::fs::write(
            out.join("encoder.f32"),
            encoded
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<u8>>(),
        )?;
        println!(
            "mel=[{},{}] encoder=[{frames},{channels}]",
            model.n_mels(),
            features.frames
        );
        return Ok(());
    }
    if args[1] == "embed" {
        if args.len() < 5 {
            bail!("embed requires <out_file.f32>")
        }
        let (values, frames, channels) = model.infer_embeddings_16k(&samples)?;
        std::fs::write(
            &args[4],
            values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<u8>>(),
        )?;
        println!("embeddings shape=[{frames},{channels}] file={}", args[4]);
        return Ok(());
    }
    if args.iter().any(|arg| arg == "--word-timestamps") {
        let chunks = model.transcribe_long_with_timestamps_16k(&samples)?;
        for (i, segment) in chunks.iter().enumerate() {
            println!(
                "[{i}] [{:.3}-{:.3}] {}",
                segment.start, segment.end, segment.transcription.text
            );
            for word in &segment.words {
                println!("  [{:.3}-{:.3}] {}", word.start, word.end, word.text);
            }
        }
    } else {
        let chunks = model.transcribe_long_16k(&samples)?;
        for (i, segment) in chunks.iter().enumerate() {
            println!("[{i}] {}", segment.text);
        }
    }
    Ok(())
}
