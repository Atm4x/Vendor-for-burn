use beat_this_rs::{audio, checkpoint::Checkpoint, model::BeatThisModel, postprocess};
use burn::tensor::Device;
use std::{env, fs, path::PathBuf, time::Instant};

fn main() {
    if let Err(err) = run() {
        eprintln!("beat-this-rs: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut checkpoint = None;
    let mut input = None;
    let mut output = None;
    let mut backend = "flex".to_owned();
    let mut limit_seconds = None;
    let mut dump_logits = None;
    let mut spect_json = None;
    let mut save_spect_json = None;
    let mut args = env::args().skip(1);
    while let Some(option) = args.next() {
        match option.as_str() {
            "--checkpoint" => checkpoint = args.next(),
            "--input" => input = args.next(),
            "--output" => output = args.next(),
            "--device" => backend = args.next().ok_or("missing --device value")?,
            "--limit-seconds" => {
                limit_seconds = Some(
                    args.next()
                        .ok_or("missing --limit-seconds value")?
                        .parse::<f64>()
                        .map_err(|e| e.to_string())?,
                )
            }
            "--logits-json" => dump_logits = args.next(),
            "--spect-json" => spect_json = args.next(),
            "--save-spect-json" => save_spect_json = args.next(),
            "--help" | "-h" => {
                println!(
                    "beat-this-rs — native Beat This! inference\n\nUsage: beat-this-rs --checkpoint MODEL.ckpt --input AUDIO [options]\n\nOptions:\n  --output FILE.beats        TSV timestamps and beat numbers (default: AUDIO.beats)\n  --device flex|cpu|cuda    Burn backend (must be compiled with the matching feature)\n  --limit-seconds SECONDS   Analyze an initial excerpt after resampling\n  --logits-json FILE        Write beat/downbeat logits as JSON\n  --spect-json FILE         Use a [frames][128] log-mel JSON instead of decoding audio\n  --save-spect-json FILE    Dump calculated log-mel features as JSON\n  -h, --help                Show this help\n\nResampler: BEAT_THIS_RESAMPLER=soxr|rubato|auto (default: auto).\nSoXR gives best parity with PyTorch; Rubato fallback may change predictions."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {option}")),
        }
    }
    let checkpoint = checkpoint.ok_or("missing --checkpoint")?;
    let input = input
        .or_else(|| spect_json.clone())
        .ok_or("missing --input or --spect-json")?;
    let device = match backend.as_str() {
        #[cfg(feature = "flex")]
        "flex" => Device::flex(),
        #[cfg(feature = "cpu")]
        "cpu" => Device::cpu(),
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        other => return Err(format!("backend '{other}' is not enabled in this build")),
    };
    let begin = Instant::now();
    let checkpoint = Checkpoint::open(&checkpoint)?;
    let model = BeatThisModel::from_checkpoint(&checkpoint, &device)?;
    eprintln!(
        "loaded checkpoint: {} tensors, transformer dim {}, {:?}",
        model.parameter_tensors,
        model.transformer_dim,
        begin.elapsed()
    );
    let spect: Vec<[f32; 128]> = if let Some(spect_path) = spect_json {
        let bytes = fs::read(spect_path).map_err(|e| e.to_string())?;
        let rows: Vec<Vec<f32>> =
            serde_json::from_slice(&bytes).map_err(|e| format!("invalid --spect-json: {e}"))?;
        rows.into_iter()
            .map(|row| {
                row.try_into()
                    .map_err(|row: Vec<f32>| format!("expected 128 mel bins, got {}", row.len()))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let audio = audio::decode(&input)?;
        let mut pcm = audio::resample_mono(&audio.pcm, audio.sample_rate)?;
        if let Some(seconds) = limit_seconds {
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err("--limit-seconds must be positive".into());
            }
            pcm.truncate(((seconds * 22050.0) as usize).min(pcm.len()));
        }
        audio::log_mel(&pcm)?
    };
    eprintln!(
        "decoded {} frames at 50 fps, {:?}",
        spect.len(),
        begin.elapsed()
    );
    if let Some(path) = save_spect_json {
        let rows: Vec<Vec<f32>> = spect.iter().map(|r| r.to_vec()).collect();
        fs::write(path, serde_json::to_vec(&rows).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    let logits = beat_this_rs::predict_logits(&model, &spect)?;
    if let Some(path) = dump_logits {
        let json = serde_json::to_vec_pretty(&logits).map_err(|e| e.to_string())?;
        fs::write(path, json).map_err(|e| e.to_string())?;
    }
    let times = postprocess::postprocess(&logits.beat, &logits.downbeat);
    let path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&input).with_extension("beats"));
    fs::write(&path, beat_this_rs::export_beats_tsv(&times)).map_err(|e| e.to_string())?;
    println!(
        "beats={} downbeats={} frames={} elapsed={:?} saved={}",
        times.beats.len(),
        times.downbeats.len(),
        spect.len(),
        begin.elapsed(),
        path.display()
    );
    Ok(())
}
