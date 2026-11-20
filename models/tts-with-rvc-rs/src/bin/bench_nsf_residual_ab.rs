use std::{env, process::ExitCode, time::Instant};
use burn::tensor::{Device, Int, Tensor, TensorData};
use tts_with_rvc_rs::{
    checkpoint::RvcCheckpoint,
    decoder::{NsfWaveformDecoder, ResidualExecution},
    flow::ResidualCouplingBlock,
    nsf_source::SourceModuleHnNSF,
    text_encoder::TextEncoder,
};

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.len() > 3 {
        return Err("usage: bench_nsf_residual_ab <model.pth> [frames=24] [pairs=5]".into());
    }
    let frames = args.get(1).map(|s| s.parse::<usize>()).transpose().map_err(|e| format!("frames: {e}"))?.unwrap_or(24);
    let pairs = args.get(2).map(|s| s.parse::<usize>()).transpose().map_err(|e| format!("pairs: {e}"))?.unwrap_or(5);
    if !(2..=512).contains(&frames) || !(1..=20).contains(&pairs) {
        return Err("invalid frames (2..=512) or pairs (1..=20)".into());
    }
    let device = Device::flex();
    let ckpt = RvcCheckpoint::open(&args[0])?;
    let encoder = TextEncoder::load(&ckpt, &device)?;
    let flow = ResidualCouplingBlock::load(&ckpt, &device)?;
    let decoder = NsfWaveformDecoder::load(&ckpt, &[10, 10, 2, 2], 3, [1, 3, 5], &device)?;
    let source = SourceModuleHnNSF::load(&ckpt, 40_000, 0, &device)?;
    let speaker = ckpt.tensor::<2>("emb_g.weight", &device)?.slice([0..1, 0..256]).reshape([1, 256, 1]);
    let phones = (0..frames * 768).map(|j| (j as f32 * 0.013).sin() * 0.1).collect::<Vec<_>>();
    let phone = Tensor::from_data(TensorData::new(phones, [1, frames, 768]), &device);
    let pitch = Tensor::<2, Int>::from_data(TensorData::new(vec![128_i64; frames], [1, frames]), &device);
    let f0 = Tensor::<2>::from_data(TensorData::new(vec![175.0_f32; frames], [1, frames]), &device);
    let encoded = encoder.forward(phone, Some(pitch), &[frames], None);
    let latent = flow.forward(encoded.mean * encoded.mask.clone(), encoded.mask.clone(), Some(speaker.clone()), true) * encoded.mask;
    let (harmonic, _) = source.forward_with_random(
        f0, 400,
        Tensor::zeros([1, 1, 1], &device),
        Tensor::zeros([1, frames * 400, 1], &device),
    )?;
    let harmonic = harmonic.permute([0, 2, 1]);
    let mut serial = Vec::with_capacity(pairs);
    let mut parallel = Vec::with_capacity(pairs);
    let mut reference: Option<Vec<f32>> = None;
    let mut max_abs = 0.0f32;
    for cycle in 0..pairs + 1 {
        let order = if cycle % 2 == 0 {
            [ResidualExecution::Sequential, ResidualExecution::Parallel]
        } else {
            [ResidualExecution::Parallel, ResidualExecution::Sequential]
        };
        for execution in order {
            let t = Instant::now();
            let values = decoder
                .forward_with_execution(latent.clone(), harmonic.clone(), Some(speaker.clone()), execution)
                .into_data().try_into_vec::<f32>().map_err(|e| e.to_string())?;
            let elapsed = t.elapsed().as_secs_f64() * 1000.0;
            if values.len() != frames * 400 || values.iter().any(|x| !x.is_finite()) {
                return Err("invalid NSF output".into());
            }
            if execution == ResidualExecution::Sequential {
                reference = Some(values);
            } else if let Some(ref expected) = reference {
                for (a, b) in values.iter().zip(expected) {
                    max_abs = max_abs.max((a - b).abs());
                }
            }
            if cycle > 0 {
                match execution {
                    ResidualExecution::Sequential => serial.push(elapsed),
                    ResidualExecution::Parallel => parallel.push(elapsed),
                }
                println!("cycle={cycle} mode={execution:?} ms={elapsed:.3}");
            }
        }
    }
    let serial_median = median(&mut serial);
    let parallel_median = median(&mut parallel);
    println!("SUMMARY frames={frames} pairs={pairs} threads={} serial_ms={serial_median:.3} parallel_ms={parallel_median:.3} speedup={:.3} max_abs={max_abs:.9}",
        env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into()), serial_median / parallel_median);
    if max_abs > 1e-5 {return Err(format!("NSF parity exceeded tolerance: {max_abs}"));}
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {eprintln!("{error}"); ExitCode::FAILURE}
    }
}
