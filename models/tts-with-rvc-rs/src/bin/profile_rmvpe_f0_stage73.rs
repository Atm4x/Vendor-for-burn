#[cfg(feature = "cuda")]
fn main() -> Result<(), String> {
    use burn::tensor::Device;
    use serde_json::json;
    use std::{env, fs, time::Instant};
    use tts_with_rvc_rs::{
        audio_io::{read_mono_wav, resample},
        checkpoint::RvcCheckpoint,
        rmvpe_mel::RmvpeAudioModel,
    };

    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        return Err("usage: profile_rmvpe_f0_stage73.exe rmvpe.pt input.wav output_prefix start_seconds duration_seconds".into());
    }
    let start: f64 = args[4].parse().map_err(|_| "invalid start seconds")?;
    let duration: f64 = args[5].parse().map_err(|_| "invalid duration seconds")?;
    if !start.is_finite() || !duration.is_finite() || start < 0.0 || !(0.5..=10.0).contains(&duration) {
        return Err("start/duration must be finite, duration 0.5..=10 seconds".into());
    }
    let wav = read_mono_wav(&args[2])?;
    let first = (start * wav.sample_rate as f64) as usize;
    let last = ((start + duration) * wav.sample_rate as f64) as usize;
    if first >= wav.samples.len() || last > wav.samples.len() || last <= first {
        return Err("RMVPE requested input segment is outside the WAV".into());
    }
    let pcm = resample(&wav.samples[first..last], wav.sample_rate, 16_000)?;
    let input = vec![pcm];
    let device = Device::cuda(0);
    let checkpoint = RvcCheckpoint::open_root(&args[1])?;
    let model = RmvpeAudioModel::load(&checkpoint, &device)?;
    let mut passes = Vec::new();
    let mut saved = Vec::new();
    let mut frames = 0;
    for pass in 0..2 {
        let started = Instant::now();
        let (salience, count) = model.forward_pcm_batch(&input, &device)?;
        frames = count;
        let values = salience.into_data().try_into_vec::<f32>()
            .map_err(|e| format!("salience readback failed: {e:?}"))?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err("RMVPE produced nonfinite salience".into());
        }
        passes.push(started.elapsed().as_secs_f64() * 1000.0);
        if pass == 1 {
            saved = values;
        }
    }
    let started = Instant::now();
    let pitch = model.infer_f0_batch_thresholds(&input, &device, &[0.03])?;
    let pitch_ms = started.elapsed().as_secs_f64() * 1000.0;
    let pitch = pitch.into_iter().next().ok_or("F0 batch empty")?;
    if pitch.len() != frames || pitch.iter().any(|x| !x.is_finite()) {
        return Err("F0 sequence invalid".into());
    }
    let encode = |values: &[f32]| -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    fs::write(format!("{}.salience.f32le", &args[3]), encode(&saved))
        .map_err(|e| e.to_string())?;
    fs::write(format!("{}.f0.f32le", &args[3]), encode(&pitch))
        .map_err(|e| e.to_string())?;
    println!("{}", json!({
        "stage": 73,
        "experiment": "full-real-RMVPE-F0-salience-parity",
        "start_seconds": start,
        "duration_seconds": duration,
        "source_rate": wav.sample_rate,
        "frames": frames,
        "salience_values": saved.len(),
        "first_salience_ms": passes[0],
        "warm_salience_ms": passes[1],
        "f0_inference_ms": pitch_ms,
        "voiced_frames": pitch.iter().filter(|x| **x > 0.0).count(),
        "f0_max_hz": pitch.iter().copied().fold(0.0_f32, f32::max),
        "gru_persistent": env::var("RVC_RMVPE_GRU_PERSISTENT").as_deref() == Ok("1"),
        "unet_conv2d_gemm": env::var("RVC_RMVPE_UNET_CONV2D_GEMM").as_deref() == Ok("1"),
        "unet_transpose2d_gemm": env::var("RVC_RMVPE_UNET_TRANSPOSE2D_GEMM").as_deref() == Ok("1"),
    }));
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Stage73 RMVPE parity requires the CUDA feature");
    std::process::exit(2);
}
