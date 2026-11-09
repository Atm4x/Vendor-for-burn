use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::{f64::consts::PI, sync::Arc};

use burn::tensor::{Device, Tensor, TensorData};
use crate::{checkpoint::RvcCheckpoint, rmvpe::{F0_BINS, salience_to_f0}, rmvpe_gru::RmvpeNetwork};

pub const SAMPLE_RATE: usize = 16_000;
pub const FFT_SIZE: usize = 1_024;
pub const HOP_LENGTH: usize = 160;
pub const MEL_BINS: usize = 128;
const MEL_FMIN: f64 = 30.0;
const MEL_FMAX: f64 = 8_000.0;
const LOG_CLAMP: f32 = 1e-5;

pub struct RmvpeMelSpectrogram {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    basis: Vec<f32>,
}

impl Default for RmvpeMelSpectrogram {
    fn default() -> Self { Self::new() }
}

impl RmvpeMelSpectrogram {
    pub fn new() -> Self {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        let window = (0..FFT_SIZE).map(|index| {
            (0.5 - 0.5 * (2.0 * PI * index as f64 / FFT_SIZE as f64).cos()) as f32
        }).collect();
        let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
        let min_mel = hz_to_mel(MEL_FMIN);
        let max_mel = hz_to_mel(MEL_FMAX);
        let frequencies: Vec<f64> = (0..MEL_BINS + 2)
            .map(|index| mel_to_hz(min_mel + (max_mel - min_mel) * index as f64 / (MEL_BINS + 1) as f64))
            .collect();
        let mut basis = vec![0.0_f32; MEL_BINS * (FFT_SIZE / 2 + 1)];
        for mel in 0..MEL_BINS {
            let low = frequencies[mel];
            let center = frequencies[mel + 1];
            let high = frequencies[mel + 2];
            let scale = 2.0 / (high - low);
            for bin in 0..=FFT_SIZE / 2 {
                let frequency = bin as f64 * SAMPLE_RATE as f64 / FFT_SIZE as f64;
                let lower = (frequency - low) / (center - low);
                let upper = (high - frequency) / (high - center);
                basis[mel * (FFT_SIZE / 2 + 1) + bin] = (lower.min(upper).max(0.0) * scale) as f32;
            }
        }
        Self { fft, window, basis }
    }

    pub fn forward(&self, audio: &[f32]) -> Result<(Vec<f32>, usize), String> {
        if audio.len() <= FFT_SIZE / 2 || audio.iter().any(|v| !v.is_finite()) {
            return Err("RMVPE requires at least 513 finite mono PCM16k samples".into());
        }
        let frames = audio.len() / HOP_LENGTH + 1;
        let mut output = vec![0.0f32; MEL_BINS * frames];
        let mut spectrum = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
        let mut magnitude = vec![0.0_f32; FFT_SIZE / 2 + 1];
        for frame in 0..frames {
            let start = (frame * HOP_LENGTH) as isize - (FFT_SIZE / 2) as isize;
            for (index, value) in spectrum.iter_mut().enumerate() {
                let position = reflect_index(start + index as isize, audio.len());
                *value = Complex32::new(audio[position] * self.window[index], 0.0);
            }
            self.fft.process(&mut spectrum);
            for (bin, value) in magnitude.iter_mut().enumerate() {
                let number = spectrum[bin];
                *value = (number.re * number.re + number.im * number.im).sqrt();
            }
            for mel in 0..MEL_BINS {
                let basis = &self.basis[mel * magnitude.len()..(mel + 1) * magnitude.len()];
                let energy: f32 = basis.iter().zip(magnitude.iter()).map(|(a, b)| a * b).sum();
                output[mel * frames + frame] = energy.max(LOG_CLAMP).ln();
            }
        }
        Ok((output, frames))
    }
}

fn reflect_index(mut index: isize, len: usize) -> usize {
    let limit = len as isize;
    while index < 0 || index >= limit {
        if index < 0 { index = -index; }
        if index >= limit { index = 2 * limit - 2 - index; }
    }
    index as usize
}

pub struct RmvpeAudioModel {
    mel: RmvpeMelSpectrogram,
    network: RmvpeNetwork,
}

impl RmvpeAudioModel {
    pub fn load(checkpoint: &RvcCheckpoint, device: &Device) -> Result<Self, String> {
        let network = RmvpeNetwork::load(checkpoint, 5, 4, MEL_BINS, device)?;
        Ok(Self { mel: RmvpeMelSpectrogram::new(), network })
    }

    pub fn forward_pcm(&self, audio_16k: &[f32], device: &Device) -> Result<(Tensor<3>, usize), String> {
        let (features, frames) = self.mel.forward(audio_16k)?;
        let mut mel = Tensor::from_data(TensorData::new(features, [1, MEL_BINS, frames]), device);
        let padded = frames.div_ceil(32) * 32;
        if padded != frames {
            let zeros = Tensor::zeros([1, MEL_BINS, padded - frames], device);
            mel = Tensor::cat(vec![mel, zeros], 2);
        }
        let result = self.network.forward(mel);
        Ok((result.slice([0..1, 0..frames, 0..F0_BINS]), frames))
    }

    pub fn infer_f0(&self, audio_16k: &[f32], device: &Device, threshold: f32) -> Result<Vec<f32>, String> {
        let (salience, _) = self.forward_pcm(audio_16k, device)?;
        let values = salience.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        salience_to_f0(&values, threshold)
    }
}
