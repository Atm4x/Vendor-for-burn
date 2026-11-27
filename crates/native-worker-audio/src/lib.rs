#![forbid(unsafe_code)]

use anyhow::{Context, Result, ensure};
use audioadapter_buffers::direct::InterleavedSlice;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rubato::{Fft, FixedSync, Resampler, audioadapter::Adapter};
use std::io::Cursor;

pub fn decode_pcm(bytes: &[u8], output_rate: usize, max_seconds: usize) -> Result<Vec<f32>> {
    let mut wav = WavReader::new(Cursor::new(bytes))?;
    let spec = wav.spec();
    let rate = spec.sample_rate as usize;
    let channels = spec.channels as usize;
    ensure!(
        (8_000..=192_000).contains(&rate),
        "unsupported WAV sample rate"
    );
    ensure!((1..=8).contains(&channels), "unsupported WAV channels");
    ensure!(
        (1..=max_seconds * rate).contains(&(wav.duration() as usize)),
        "WAV duration is out of range"
    );
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => wav.samples::<f32>().collect::<Result<_, _>>()?,
        SampleFormat::Int => {
            ensure!(
                (8..=32).contains(&spec.bits_per_sample),
                "unsupported WAV sample width"
            );
            let scale = (1_u64 << (spec.bits_per_sample - 1)) as f32;
            wav.samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    ensure!(
        samples.len() % channels == 0 && samples.iter().all(|s| s.is_finite()),
        "invalid WAV samples"
    );
    let pcm: Vec<f32> = samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    resample_mono(&pcm, rate, output_rate)
}

pub fn resample_mono(pcm: &[f32], rate: usize, output_rate: usize) -> Result<Vec<f32>> {
    if rate == output_rate {
        return Ok(pcm.to_vec());
    }
    let input = InterleavedSlice::new(pcm, 1, pcm.len())?;
    let mut resampler = Fft::<f32>::new(rate, output_rate, 1024, 1, FixedSync::Both)?;
    let output = resampler.process_all(&input, pcm.len(), None)?;
    (0..output.frames())
        .map(|i| output.read_sample(0, i).context("missing resampled frame"))
        .collect()
}

pub fn encode_pcm16(pcm: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    ensure!(
        !pcm.is_empty() && pcm.iter().all(|s| s.is_finite()),
        "invalid output PCM"
    );
    let mut bytes = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(
        &mut bytes,
        WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        },
    )?;
    for &sample in pcm {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
    }
    writer.finalize()?;
    Ok(bytes.into_inner())
}
