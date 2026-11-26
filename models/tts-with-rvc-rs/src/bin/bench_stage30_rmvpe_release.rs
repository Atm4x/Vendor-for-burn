use std::time::Instant;
use burn::tensor::Device;
use tts_with_rvc_rs::{audio_io::{read_mono_wav, resample}, checkpoint::RvcCheckpoint, rmvpe_mel::RmvpeAudioModel};

fn main() -> Result<(), String> {
    let audio = read_mono_wav("/mnt/data/stage24_input_1_15s.wav")?;
    let pcm = resample(&audio.samples, audio.sample_rate, 16000)?;
    let device = Device::flex();
    let ckpt = RvcCheckpoint::open_root("/mnt/data/rmvpe.pt")?;
    let model = RmvpeAudioModel::load(&ckpt, &device)?;
    let mut times = Vec::new();
    let mut reference: Option<Vec<f32>> = None;
    for round in 0..8 {
        let now = Instant::now();
        let f0 = model.infer_f0(&pcm, &device, 0.03)?;
        let ms = now.elapsed().as_secs_f64() * 1000.0;
        if f0.iter().any(|v| !v.is_finite() || *v < 0.0) { return Err("invalid F0".into()); }
        if let Some(r) = &reference {
            if r.len()!=f0.len() || r.iter().zip(f0.iter()).any(|(&a,&b)| (a-b).abs()>1e-3) {return Err("nondeterministic F0".into());}
        } else {reference=Some(f0.clone());}
        println!("round={round} ms={ms:.3} frames={} voiced={}",f0.len(),f0.iter().filter(|&&v| v>0.0).count());
        if round>0 { times.push(ms); }
    }
    times.sort_by(|a,b| a.total_cmp(b));
    let values=reference.unwrap();
    let output=serde_json::json!({"f0":values});
    let path=std::env::var("STAGE30_F0_OUT").unwrap_or_else(|_|"/mnt/data/stage22-work/stage30-evidence/stage30-f0.json".into());
    std::fs::write(&path,serde_json::to_vec(&output).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    println!("median_ms={:.3} min_ms={:.3} max_ms={:.3} f0_path={path}",times[times.len()/2],times[0],times[times.len()-1]);
    Ok(())
}
