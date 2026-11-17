use std::f64::consts::PI;

use crate::audio::{HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE};
use crate::error::{FishError, Result};
use crate::firefly_cpu::AudioTensor;

#[derive(Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

fn fft_in_place(x: &mut [Complex]) -> Result<()> {
    let n = x.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(FishError::InvalidShape("FFT length must be power of two"));
    }
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    let mut span = 2;
    while span <= n {
        let half = span / 2;
        let angle = -2.0 * PI / span as f64;
        for offset in (0..n).step_by(span) {
            for k in 0..half {
                let (sin, cos) = (angle * k as f64).sin_cos();
                let right = x[offset + k + half];
                let rotate = Complex {
                    re: right.re * cos - right.im * sin,
                    im: right.re * sin + right.im * cos,
                };
                let left = x[offset + k];
                x[offset + k] = Complex {
                    re: left.re + rotate.re,
                    im: left.im + rotate.im,
                };
                x[offset + k + half] = Complex {
                    re: left.re - rotate.re,
                    im: left.im - rotate.im,
                };
            }
        }
        span = span
            .checked_mul(2)
            .ok_or(FishError::InvalidShape("FFT stage overflow"))?;
    }
    Ok(())
}

fn reflect_value(audio: &[f32], logical_index: isize) -> f64 {
    let len = audio.len() as isize;
    let ix = if logical_index < 0 {
        -logical_index
    } else if logical_index >= len {
        2 * len - logical_index - 2
    } else {
        logical_index
    };
    f64::from(audio[ix as usize])
}

fn hz_to_mel(freq: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    if freq < 1000.0 {
        freq / f_sp
    } else {
        15.0 + (freq / 1000.0).ln() / (6.4_f64.ln() / 27.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15.0 {
        mel * (200.0 / 3.0)
    } else {
        1000.0 * ((mel - 15.0) * (6.4_f64.ln() / 27.0)).exp()
    }
}

pub struct FishLogMel {
    filter_bank: Vec<f64>,
    hann_window: Vec<f64>,
}

impl FishLogMel {
    pub fn new() -> Self {
        let frequencies = N_FFT / 2 + 1;
        let mel_max = hz_to_mel((SAMPLE_RATE / 2) as f64);
        let points: Vec<_> = (0..N_MELS + 2)
            .map(|i| mel_to_hz(mel_max * i as f64 / (N_MELS + 1) as f64))
            .collect();
        let mut filter_bank = vec![0.0; N_MELS * frequencies];
        for m in 0..N_MELS {
            let left = points[m];
            let center = points[m + 1];
            let right = points[m + 2];
            let slaney_normalization = 2.0 / (right - left);
            for freq in 0..frequencies {
                let hz = freq as f64 * (SAMPLE_RATE / 2) as f64 / (frequencies - 1) as f64;
                let left_slope = (hz - left) / (center - left);
                let right_slope = (right - hz) / (right - center);
                filter_bank[m * frequencies + freq] =
                    left_slope.min(right_slope).max(0.0) * slaney_normalization;
            }
        }
        let hann_window = (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / N_FFT as f64).cos())
            .collect();
        Self {
            filter_bank,
            hann_window,
        }
    }

    pub fn transform_mono(&self, audio: &[f32]) -> Result<AudioTensor> {
        if audio.len() <= (N_FFT - HOP_LENGTH) / 2 {
            return Err(FishError::InvalidShape(
                "reflect padding requires audio longer than 768 samples",
            ));
        }
        if audio.len() > isize::MAX as usize || audio.iter().any(|x| !x.is_finite()) {
            return Err(FishError::InvalidShape("invalid mono waveform samples"));
        }
        let frames = audio.len() / HOP_LENGTH;
        let num_freqs = N_FFT / 2 + 1;
        let mut output = vec![0.0f32; N_MELS * frames];
        let mut buffer = vec![Complex { re: 0.0, im: 0.0 }; N_FFT];
        let mut magnitudes = vec![0.0; num_freqs];
        let left_padding = (N_FFT - HOP_LENGTH) / 2;
        for frame in 0..frames {
            let start = (frame * HOP_LENGTH) as isize - left_padding as isize;
            for (n, v) in buffer.iter_mut().enumerate() {
                *v = Complex {
                    re: reflect_value(audio, start + n as isize) * self.hann_window[n],
                    im: 0.0,
                };
            }
            fft_in_place(&mut buffer)?;
            for bin in 0..num_freqs {
                let v = buffer[bin];
                magnitudes[bin] = (v.re * v.re + v.im * v.im + 1e-6).sqrt();
            }
            for mel in 0..N_MELS {
                let filter = &self.filter_bank[mel * num_freqs..(mel + 1) * num_freqs];
                let amplitude = filter
                    .iter()
                    .zip(&magnitudes)
                    .map(|(&weight, &mag)| weight * mag)
                    .sum::<f64>();
                output[mel * frames + frame] = amplitude.max(1e-5).ln() as f32;
            }
        }
        AudioTensor::new(1, N_MELS, frames, output)
    }
}

impl Default for FishLogMel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_impulse_is_constant() {
        let mut data = vec![Complex { re: 0.0, im: 0.0 }; 16];
        data[0].re = 2.0;
        fft_in_place(&mut data).unwrap();
        assert!(data
            .iter()
            .all(|v| (v.re - 2.0).abs() < 1e-12 && v.im.abs() < 1e-12));
    }
    #[test]
    fn fft_sine_has_the_two_expected_bins() {
        let mut data = (0..32)
            .map(|n| Complex {
                re: (2.0 * PI * 3.0 * n as f64 / 32.0).cos(),
                im: 0.0,
            })
            .collect::<Vec<_>>();
        fft_in_place(&mut data).unwrap();
        assert!((data[3].re - 16.0).abs() < 1e-10);
        assert!((data[29].re - 16.0).abs() < 1e-10);
        assert!(data
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 3 && *i != 29)
            .all(|(_, v)| v.re.abs() < 1e-10 && v.im.abs() < 1e-10));
    }
    #[test]
    fn mel_scale_frequencies_are_invertible() {
        for hz in [0.0, 500.0, 1000.0, 2000.0, 5000.0, 22050.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-8);
        }
    }
    #[test]
    fn waveform_validation() {
        let mel = FishLogMel::new();
        assert!(mel.transform_mono(&vec![0.0; 768]).is_err());
        let mut bad = vec![0.0; 1024];
        bad[123] = f32::NAN;
        assert!(mel.transform_mono(&bad).is_err());
    }
}
