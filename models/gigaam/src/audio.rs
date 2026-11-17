use anyhow::{Context, Result, ensure};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::path::Path;

pub const SAMPLE_RATE: u32 = 16_000;
pub const FFT_SIZE: usize = 400;
pub const HOP: usize = 160;

pub fn load_wav_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    ensure!(
        spec.sample_rate >= 8000 && spec.sample_rate <= 192000,
        "unsupported sample rate"
    );
    ensure!(
        spec.channels >= 1 && spec.channels <= 8,
        "unsupported channel count"
    );
    let data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            ensure!(
                (8..=32).contains(&spec.bits_per_sample),
                "unsupported integer WAV bit depth"
            );
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|x| x.map(|v| v as f32 / scale))
                .collect::<std::result::Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
    };
    let mono: Vec<f32> = data
        .chunks_exact(spec.channels as usize)
        .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
        .collect();
    if spec.sample_rate == SAMPLE_RATE {
        return Ok(mono);
    }
    Ok(resample_sinc(&mono, spec.sample_rate, SAMPLE_RATE))
}

pub fn resample_sinc(samples: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    if input_rate == output_rate {
        return samples.to_vec();
    }
    let output_len = (samples.len() as u64 * output_rate as u64 / input_rate as u64) as usize;
    let ratio = input_rate as f64 / output_rate as f64;
    let cutoff = (0.95f64 * output_rate as f64 / input_rate as f64).min(0.95);
    let width: isize = 32;
    (0..output_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let pivot = pos.floor() as isize;
            let mut sum = 0.0f64;
            let mut weights = 0.0f64;
            for j in (pivot - width + 1)..=(pivot + width) {
                if j < 0 || j >= samples.len() as isize {
                    continue;
                }
                let t = pos - j as f64;
                let x = std::f64::consts::PI * t;
                let sinc = if x.abs() < 1e-10 {
                    cutoff
                } else {
                    (cutoff * x).sin() / x
                };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * t / width as f64).cos();
                let w = sinc * window;
                sum += w * samples[j as usize] as f64;
                weights += w;
            }
            if weights.abs() < 1e-12 {
                0.0
            } else {
                (sum / weights) as f32
            }
        })
        .collect()
}

fn reflect_index(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (2 * (len - 1)) as isize;
    let r = index.rem_euclid(period);
    if r < len as isize {
        r as usize
    } else {
        (period - r) as usize
    }
}

fn htk_hz_to_mel(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}
fn htk_mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn mel_weights(n_mels: usize) -> Vec<f32> {
    let bins = FFT_SIZE / 2 + 1;
    let low = htk_hz_to_mel(0.0);
    let high = htk_hz_to_mel(SAMPLE_RATE as f32 / 2.0);
    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| htk_mel_to_hz(low + (high - low) * i as f32 / (n_mels + 1) as f32))
        .collect();
    let mut weights = vec![0f32; n_mels * bins];
    for m in 0..n_mels {
        for k in 0..bins {
            let f = k as f32 * SAMPLE_RATE as f32 / FFT_SIZE as f32;
            let lo = (f - mel_points[m]) / (mel_points[m + 1] - mel_points[m]);
            let hi = (mel_points[m + 2] - f) / (mel_points[m + 2] - mel_points[m + 1]);
            weights[m * bins + k] = lo.min(hi).max(0.0);
        }
    }
    weights
}

#[derive(Debug, Clone)]
pub struct Features {
    /// [mel_bins, frames], channels-first
    pub values: Vec<f32>,
    pub n_mels: usize,
    pub frames: usize,
}

pub fn log_mel(samples: &[f32], n_mels: usize) -> Result<Features> {
    ensure!(!samples.is_empty(), "empty audio input");
    ensure!((1..=256).contains(&n_mels), "n_mels outside 1..=256");
    let frames = samples.len() / HOP + 1;
    let bins = FFT_SIZE / 2 + 1;
    let weights = mel_weights(n_mels);
    let mut data = vec![0f32; n_mels * frames];
    let hann: Vec<f32> = (0..FFT_SIZE)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / FFT_SIZE as f32).cos())
        .collect();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut scratch = vec![Complex::<f32>::default(); fft.get_inplace_scratch_len()];
    let mut buffer = vec![Complex::<f32>::default(); FFT_SIZE];
    for t in 0..frames {
        let center = (t * HOP) as isize;
        for i in 0..FFT_SIZE {
            let sample = samples
                [reflect_index(center + i as isize - (FFT_SIZE / 2) as isize, samples.len())];
            buffer[i] = Complex::new(sample * hann[i], 0.0);
        }
        fft.process_with_scratch(&mut buffer, &mut scratch);
        for m in 0..n_mels {
            let mut mel = 0.0f32;
            for k in 0..bins {
                mel += (buffer[k].norm_sqr()) * weights[m * bins + k]
            }
            data[m * frames + t] = mel.clamp(1e-9, 1e9).ln();
        }
    }
    Ok(Features {
        values: data,
        n_mels,
        frames,
    })
}

/// Produces <=max_samples chunks, preferring a low-energy boundary within the final second.
pub fn speech_chunks(samples: &[f32], max_samples: usize) -> Vec<&[f32]> {
    if samples.is_empty() || max_samples == 0 {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut pos = 0;
    while pos < samples.len() {
        let end = (pos + max_samples).min(samples.len());
        let mut split = end;
        if end < samples.len() && max_samples > SAMPLE_RATE as usize {
            let begin = end.saturating_sub(SAMPLE_RATE as usize).max(pos + 1);
            let step = 160;
            let mut energy = f64::MAX;
            for i in (begin..end).step_by(step) {
                let e = samples[i..(i + step).min(samples.len())]
                    .iter()
                    .map(|x| (*x as f64) * (*x as f64))
                    .sum::<f64>();
                if e < energy {
                    energy = e;
                    split = i
                }
            }
        }
        chunks.push(&samples[pos..split]);
        pos = split;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mel_frame_count() {
        let m = log_mel(&vec![0.0; 16000], 64).unwrap();
        assert_eq!(m.frames, 101);
        assert_eq!(m.values.len(), 64 * 101);
        assert!(m.values.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn resampling_ratio() {
        assert_eq!(resample_sinc(&vec![0.0; 44100], 44100, 16000).len(), 16000);
    }
    #[test]
    fn speech_chunk_limits() {
        let x = vec![0f32; 16000 * 59];
        let c = speech_chunks(&x, 16000 * 24);
        assert!(c.len() >= 3);
        assert!(c.iter().all(|x| x.len() <= 16000 * 24));
        assert_eq!(c.iter().map(|x| x.len()).sum::<usize>(), x.len());
    }
}
