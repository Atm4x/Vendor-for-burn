use std::{env, process::ExitCode, time::Instant};
use burn::tensor::{Device, Int, Tensor, TensorData};
use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, decoder::NsfWaveformDecoder, flow::ResidualCouplingBlock, nsf_source::SourceModuleHnNSF, text_encoder::TextEncoder};

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.len() > 3 { return Err("usage: bench_rvc_synth <model.pth> [frames=24] [iterations=2]".into()); }
    let frames: usize = args.get(1).map(|s| s.parse()).transpose().map_err(|e| format!("frames: {e}"))?.unwrap_or(24);
    let iterations: usize = args.get(2).map(|s| s.parse()).transpose().map_err(|e| format!("iterations: {e}"))?.unwrap_or(2);
    if !(2..=512).contains(&frames) || !(1..=50).contains(&iterations) { return Err("frames must be 2..=512 and iterations 1..=50".into()); }
    let device = Device::flex();
    let start = Instant::now();
    let checkpoint = RvcCheckpoint::open(&args[0])?;
    let encoder = TextEncoder::load(&checkpoint, &device)?;
    let flow = ResidualCouplingBlock::load(&checkpoint, &device)?;
    let decoder = NsfWaveformDecoder::load(&checkpoint, &[10, 10, 2, 2], 3, [1, 3, 5], &device)?;
    let source = SourceModuleHnNSF::load(&checkpoint, 40_000, 0, &device)?;
    let speaker = checkpoint.tensor::<2>("emb_g.weight", &device)?.slice([0..1, 0..256]).reshape([1, 256, 1]);
    println!("load_s {:.6}", start.elapsed().as_secs_f64());
    for i in 0..=iterations {
        let phone_values: Vec<_> = (0..frames*768).map(|j| ((j as f32 * 0.013).sin()) * 0.1).collect();
        let phone = Tensor::from_data(TensorData::new(phone_values, [1, frames, 768]), &device);
        let pitch = Tensor::<2, Int>::from_data(TensorData::new(vec![128_i64; frames], [1, frames]), &device);
        let f0 = Tensor::<2>::from_data(TensorData::new(vec![175.0_f32; frames], [1, frames]), &device);
        let first = Instant::now();
        let encoded = encoder.forward(phone, Some(pitch), &[frames], None);
        let encoder_s = first.elapsed().as_secs_f64();
        let second = Instant::now();
        let latent = flow.forward(encoded.mean * encoded.mask.clone(), encoded.mask.clone(), Some(speaker.clone()), true);
        let flow_s = second.elapsed().as_secs_f64();
        let third = Instant::now();
        let waveform = decoder.forward_from_f0(latent * encoded.mask, f0, Some(speaker.clone()), &source,
            Tensor::zeros([1, 1, 1], &device), Tensor::zeros([1, frames * 400, 1], &device))?;
        let samples = waveform.dims()[2];
        let values = waveform.into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
        let nsf_s = third.elapsed().as_secs_f64();
        if samples != frames * 400 || values.iter().any(|x| !x.is_finite()) {return Err("invalid output".into());}
        if i == 0 { println!("iteration 0 warmup samples={samples}"); }
        else { println!("iteration {i} encoder_s={encoder_s:.6} flow_s={flow_s:.6} nsf_s={nsf_s:.6} total_s={:.6} samples={samples}", encoder_s+flow_s+nsf_s); }
    }
    Ok(())
}
fn main() -> ExitCode {
    match run() { Ok(()) => ExitCode::SUCCESS, Err(e) => {eprintln!("{e}");ExitCode::FAILURE} }
}
