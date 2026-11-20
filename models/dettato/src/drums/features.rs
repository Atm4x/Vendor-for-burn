use crate::resample;
use half::f16;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

pub const RATE: u32 = 44_100;
pub const BANDS: usize = 280;
pub const MEL_BANDS: usize = 160;
pub const CQ_BANDS: usize = 120;
const FLOOR: f32 = 1e-5;

#[derive(Clone, Debug)]
pub struct DrumFeatures {
    pub frames: usize,
    /// Row-major (frames, 280) log-power features; 128 low mel, 32 high mel, 120 CQ.
    pub data: Vec<f32>,
}

type SparseBank = Vec<Vec<(usize, f32)>>;

fn hz_to_mel(freq: f64) -> f64 { 2595.0 * (1.0 + freq / 700.0).log10() }
fn mel_to_hz(mel: f64) -> f64 { 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0) }

fn mel_bank(rate: usize, fft: usize, bands: usize, fmin: f64, fmax: f64) -> SparseBank {
    let left = hz_to_mel(fmin);
    let right = hz_to_mel(fmax);
    let hz = (0..bands+2).map(|i| mel_to_hz(left + (right - left) * i as f64 / (bands+1) as f64))
        .collect::<Vec<_>>();
    (0..bands).map(|b| {
        let (lo, centre, hi) = (hz[b], hz[b+1], hz[b+2]);
        let factor = 2.0 / (hi-lo);
        (0..=fft/2).filter_map(|i| {
            let f = i as f64 * rate as f64 / fft as f64;
            let w = ((f-lo)/(centre-lo)).min((hi-f)/(hi-centre)).max(0.0) * factor;
            (w > 0.0).then_some((i, w as f32))
        }).collect::<Vec<_>>()
    }).collect()
}

fn cq_bank() -> SparseBank {
    const FFT: usize = 6144;
    const RATE: usize = 16000;
    let edges = (0..CQ_BANDS+2).map(|i| 36.708f64 * 2.0f64.powf((i as f64 - 1.0)/24.0))
        .collect::<Vec<_>>();
    (0..CQ_BANDS).map(|b| {
        let (lo, centre, hi) = (edges[b],edges[b+1],edges[b+2]);
        let mut row = (0..=FFT/2).filter_map(|i| {
            let f = i as f64 * RATE as f64 / FFT as f64;
            let v = ((f-lo)/(centre-lo)).min((hi-f)/(hi-centre)).max(0.0);
            (v > 0.0).then_some((i,v))
        }).collect::<Vec<_>>();
        let sum: f64 = row.iter().map(|(_,v)| *v).sum();
        if sum < 1.0 {
            let at = centre * FFT as f64 / RATE as f64;
            let bin = at.floor() as usize;
            row = vec![(bin, 1.0-(at-bin as f64)),(bin+1, at-bin as f64)];
        }
        let total: f64 = row.iter().map(|(_,v)| *v).sum();
        row.into_iter().map(|(i,v)|(i,(v/total) as f32)).collect::<Vec<_>>()
    }).collect()
}

fn log_spectrum(audio: &[f32], fft_size: usize, hop: usize, bank: &SparseBank, frames: usize) -> Vec<f32> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let hann = (0..fft_size).map(|i| (0.5 - 0.5*(2.0*PI*i as f64/fft_size as f64).cos()) as f32)
        .collect::<Vec<_>>();
    let mut buffer = vec![Complex::new(0.0,0.0);fft_size];
    let mut output = Vec::with_capacity(frames * bank.len());
    for frame in 0..frames {
        let centre = frame as isize * hop as isize;
        let start = centre - fft_size as isize/2;
        for (i,z) in buffer.iter_mut().enumerate() {
            let position = start + i as isize;
            z.re = if position >= 0 { audio.get(position as usize).copied().unwrap_or(0.0) }
                else { 0.0 } * hann[i];
            z.im = 0.0;
        }
        fft.process(&mut buffer);
        for row in bank {
            let power = row.iter().map(|&(index, weight)| weight * buffer[index].norm_sqr()).sum::<f32>();
            output.push((power.max(0.0)+FLOOR).ln());
        }
    }
    output
}

/// Python `drums.features.features`: 44.1 kHz mono -> frame-major [T,280].
/// Source gain, filter layout and centered STFT follow the upstream model.
/// Conversion to 16 kHz uses a native windowed-sinc resampler rather than soxr HQ.
pub fn extract(audio44k: &[f32]) -> Result<DrumFeatures, String> {
    if audio44k.iter().any(|x| !x.is_finite()) { return Err("non-finite drum audio".into()); }
    if audio44k.is_empty() { return Ok(DrumFeatures { frames: 0, data: vec![] }); }
    let rms = (audio44k.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / audio44k.len() as f64).sqrt() + 1e-9;
    let gain = 10f64.powf(-18.0/20.0) / rms;
    let normalized = audio44k.iter().map(|&x| (x as f64 * gain) as f32).collect::<Vec<_>>();
    let low = resample::resample(&normalized, RATE, 16000.0)?;
    let frames = low.len()/160;
    if frames == 0 { return Ok(DrumFeatures { frames: 0, data: vec![] }); }
    let lo = log_spectrum(&low, 512, 160, &mel_bank(16000, 512, 128, 20.0, 7800.0), frames);
    let hi = log_spectrum(&normalized, 1411, 441, &mel_bank(44100, 1411, 32, 7800.0, 20000.0), frames);
    let cq = log_spectrum(&low, 6144, 160, &cq_bank(), frames);
    let mut data = Vec::with_capacity(frames * BANDS);
    for t in 0..frames {
        data.extend_from_slice(&lo[t*128..(t+1)*128]);
        data.extend_from_slice(&hi[t*32..(t+1)*32]);
        data.extend_from_slice(&cq[t*120..(t+1)*120]);
    }
    for value in &mut data { *value = f16::from_f32(*value).to_f32(); }
    Ok(DrumFeatures { frames, data })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn filters_preserve_expected_number_and_normalization() {
        let mel = mel_bank(16000,512,128,20.0,7800.0);
        assert_eq!(mel.len(),128);
        assert!(mel.iter().all(|r| !r.is_empty()));
        let cq = cq_bank();
        assert_eq!(cq.len(),120);
        for row in cq {
            let sum: f32 = row.iter().map(|x| x.1).sum();
            assert!((sum-1.0).abs()<1e-6);
        }
    }
    #[test]
    fn empty_and_silent_audio() {
        assert_eq!(extract(&[]).unwrap().frames,0);
        let result = extract(&[0.0; 4410]).unwrap();
        assert_eq!(result.frames,10);
        assert_eq!(result.data.len(),2800);
        for v in result.data { assert!((v-FLOOR.ln()).abs()<1e-5); }
    }
}
