use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use std::{f64::consts::PI, sync::Arc};

use crate::{
    checkpoint::RvcCheckpoint,
    rmvpe::{COMPACT_F0_WIDTH, F0_BINS, compact_salience, compact_salience_to_f0, salience_to_f0},
    rmvpe_gru::RmvpeNetwork,
};
use burn::tensor::{Device, Tensor, TensorData};

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
    spans: Vec<(usize, usize)>,
}

impl Default for RmvpeMelSpectrogram {
    fn default() -> Self {
        Self::new()
    }
}

impl RmvpeMelSpectrogram {
    pub fn new() -> Self {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        let window = (0..FFT_SIZE)
            .map(|index| (0.5 - 0.5 * (2.0 * PI * index as f64 / FFT_SIZE as f64).cos()) as f32)
            .collect();
        let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
        let min_mel = hz_to_mel(MEL_FMIN);
        let max_mel = hz_to_mel(MEL_FMAX);
        let frequencies: Vec<f64> = (0..MEL_BINS + 2)
            .map(|index| {
                mel_to_hz(min_mel + (max_mel - min_mel) * index as f64 / (MEL_BINS + 1) as f64)
            })
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
        let spans = basis
            .chunks_exact(FFT_SIZE / 2 + 1)
            .map(|band| {
                let start = band.iter().position(|weight| *weight != 0.0).unwrap_or(0);
                let end = band
                    .iter()
                    .rposition(|weight| *weight != 0.0)
                    .map_or(start, |i| i + 1);
                (start, end)
            })
            .collect();
        Self {
            fft,
            window,
            basis,
            spans,
        }
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
                let (start, end) = self.spans[mel];
                let energy: f32 = basis[start..end]
                    .iter()
                    .zip(&magnitude[start..end])
                    .map(|(a, b)| a * b)
                    .sum();
                output[mel * frames + frame] = energy.max(LOG_CLAMP).ln();
            }
        }
        Ok((output, frames))
    }
}

fn reflect_index(mut index: isize, len: usize) -> usize {
    let limit = len as isize;
    while index < 0 || index >= limit {
        if index < 0 {
            index = -index;
        }
        if index >= limit {
            index = 2 * limit - 2 - index;
        }
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
        Ok(Self {
            mel: RmvpeMelSpectrogram::new(),
            network,
        })
    }

    pub fn forward_pcm_batch(
        &self,
        audios: &[Vec<f32>],
        device: &Device,
    ) -> Result<(Tensor<3>, usize), String> {
        let first_len = audios
            .first()
            .ok_or("RMVPE requires a nonempty batch")?
            .len();
        if audios.iter().any(|audio| audio.len() != first_len) {
            return Err("RMVPE tensor batches require equal audio lengths".into());
        }
        let frames = first_len / HOP_LENGTH + 1;
        let padded = frames.div_ceil(32) * 32;
        let mut features = Vec::with_capacity(audios.len() * MEL_BINS * padded);
        for audio in audios {
            let (mel, nframes) = self.mel.forward(audio)?;
            if nframes != frames {
                return Err("RMVPE frame count mismatch".into());
            }
            for band in mel.chunks_exact(frames) {
                features.extend_from_slice(band);
                features.resize(features.len() + padded - frames, 0.0);
            }
        }
        let batch = audios.len();
        let mel = Tensor::from_data(TensorData::new(features, [batch, MEL_BINS, padded]), device);
        let result = self.network.forward(mel);
        Ok((result.slice([0..batch, 0..frames, 0..F0_BINS]), frames))
    }

    /// Experimental GPU-resident FP32 F0; no host salience readback.
    #[cfg(any(feature = "cuda", feature = "cpu", feature = "wgpu"))]
    pub fn infer_f0_batch_resident(
        &self,
        audios: &[Vec<f32>],
        device: &Device,
        thresholds: &[f32],
    ) -> Result<Tensor<2>, String> {
        if audios.is_empty() || audios.len() != thresholds.len() {
            return Err("RMVPE thresholds must match input batch".into());
        }
        let (salience, _) = self.forward_pcm_batch(audios, device)?;
        let output = crate::rmvpe_gpu_f0::resident_f0(salience, thresholds)?;
        if std::env::var_os("RVC_PROFILE").is_some() {
            eprintln!("profile rmvpe_f0_readback gpu_resident B={} frames={} host_bytes=0", audios.len(), output.dims()[1]);
        }
        Ok(output)
    }

    pub fn infer_f0_batch(
        &self,
        audios: &[Vec<f32>],
        device: &Device,
        threshold: f32,
    ) -> Result<Vec<Vec<f32>>, String> {
        self.infer_f0_batch_thresholds(audios, device, &vec![threshold; audios.len()])
    }

    pub fn infer_f0_batch_thresholds(
        &self,
        audios: &[Vec<f32>],
        device: &Device,
        thresholds: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if audios.is_empty() || audios.len() != thresholds.len() {
            return Err("RMVPE thresholds must match the nonempty input batch".into());
        }
        let (salience, _frames) = self.forward_pcm_batch(audios, device)?;
        decode_salience_batch(salience, thresholds, compact_readback_enabled())
    }

    pub fn forward_pcm(
        &self,
        audio_16k: &[f32],
        device: &Device,
    ) -> Result<(Tensor<3>, usize), String> {
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

    pub fn infer_f0(
        &self,
        audio_16k: &[f32],
        device: &Device,
        threshold: f32,
    ) -> Result<Vec<f32>, String> {
        let (salience, _) = self.forward_pcm(audio_16k, device)?;
        decode_salience_batch(salience, &[threshold], compact_readback_enabled())?
            .into_iter()
            .next()
            .ok_or("RMVPE returned an empty F0 batch".into())
    }
}

/// Opt-in until RTX 4060 latency and full-route parity are verified.
fn compact_readback_enabled() -> bool {
    std::env::var_os("RVC_RMVPE_COMPACT_READBACK").is_some_and(|value| value == "1")
}

fn decode_salience_batch(
    salience: Tensor<3>,
    thresholds: &[f32],
    compact: bool,
) -> Result<Vec<Vec<f32>>, String> {
    let [batch, frames, bins] = salience.dims();
    if batch != thresholds.len() || bins != F0_BINS || frames == 0 {
        return Err("RMVPE salience/threshold shapes do not match".into());
    }
    if std::env::var_os("RVC_RMVPE_PLANE_READBACK").is_some_and(|value| value == "1") {
        #[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
        {
            if compact
                || std::env::var_os("RVC_RMVPE_FUSED_READBACK").is_some_and(|value| value == "1")
            {
                return Err("RMVPE salience readback modes are mutually exclusive".into());
            }
            let values = crate::rmvpe_fused::plane_compact_salience(salience)?;
            if std::env::var_os("RVC_PROFILE").is_some() {
                eprintln!(
                    "profile rmvpe_f0_readback plane B={batch} frames={frames} host_bytes={}",
                    values.len() * 4
                );
            }
            return values
                .chunks_exact(frames * COMPACT_F0_WIDTH)
                .zip(thresholds)
                .map(|(row, &threshold)| compact_salience_to_f0(row, threshold))
                .collect();
        }
        #[cfg(not(any(feature = "cpu", feature = "cuda", feature = "wgpu")))]
        {
            return Err("RMVPE plane readback requires a CubeCL-enabled build".into());
        }
    }
    if std::env::var_os("RVC_RMVPE_FUSED_READBACK").is_some_and(|value| value == "1") {
        #[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
        {
            let values = crate::rmvpe_fused::fused_compact_salience(salience)?;
            if std::env::var_os("RVC_PROFILE").is_some() {
                eprintln!(
                    "profile rmvpe_f0_readback fused B={batch} frames={frames} host_bytes={}",
                    values.len() * 4
                );
            }
            return values
                .chunks_exact(frames * COMPACT_F0_WIDTH)
                .zip(thresholds)
                .map(|(row, &threshold)| compact_salience_to_f0(row, threshold))
                .collect();
        }
        #[cfg(not(any(feature = "cpu", feature = "cuda", feature = "wgpu")))]
        {
            return Err("RMVPE fused readback requires a CubeCL-enabled build".into());
        }
    }
    if compact {
        let values = compact_salience(salience)
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|error| error.to_string())?;
        if std::env::var_os("RVC_PROFILE").is_some() {
            eprintln!(
                "profile rmvpe_f0_readback compact B={batch} frames={frames} host_bytes={}",
                values.len() * 4
            );
        }
        values
            .chunks_exact(frames * COMPACT_F0_WIDTH)
            .zip(thresholds)
            .map(|(row, &threshold)| compact_salience_to_f0(row, threshold))
            .collect()
    } else {
        let values = salience
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|error| error.to_string())?;
        if std::env::var_os("RVC_PROFILE").is_some() {
            eprintln!(
                "profile rmvpe_f0_readback dense B={batch} frames={frames} host_bytes={}",
                values.len() * 4
            );
        }
        values
            .chunks_exact(frames * F0_BINS)
            .zip(thresholds)
            .map(|(row, &threshold)| salience_to_f0(row, threshold))
            .collect()
    }
}

#[cfg(test)]
mod readback_tests {
    use super::*;

    #[test]
    fn mixed_thresholds_and_batches_agree_exactly_between_transfer_modes() {
        let device = Device::cpu();
        let batch = 4;
        let frames = 17;
        let mut values = vec![0.0f32; batch * frames * F0_BINS];
        for row in 0..batch * frames {
            let center = [0, 359, 4, 355, 179, 300, 12][row % 7];
            for bin in 0..F0_BINS {
                values[row * F0_BINS + bin] = if bin == center {
                    0.9
                } else {
                    0.1 + (bin % 7) as f32 / 1000.0
                };
            }
        }
        let thresholds = [0.03, 0.7, 0.9, 0.99];
        let input = || {
            Tensor::<3>::from_data(
                TensorData::new(values.clone(), [batch, frames, F0_BINS]),
                &device,
            )
        };
        let dense = decode_salience_batch(input(), &thresholds, false).unwrap();
        let compact = decode_salience_batch(input(), &thresholds, true).unwrap();
        assert_eq!(dense.len(), batch);
        for (full, small) in dense.iter().zip(compact.iter()) {
            assert!(
                full.iter()
                    .zip(small)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}

#[cfg(test)]
mod sparse_mel_tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn dense_reference(model: &RmvpeMelSpectrogram, audio: &[f32]) -> Vec<f32> {
        let frames = audio.len() / HOP_LENGTH + 1;
        let mut output = vec![0.0; MEL_BINS * frames];
        let mut spectrum = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
        let mut magnitude = vec![0.0_f32; FFT_SIZE / 2 + 1];
        for frame in 0..frames {
            let start = (frame * HOP_LENGTH) as isize - FFT_SIZE as isize / 2;
            for (i, value) in spectrum.iter_mut().enumerate() {
                let position = reflect_index(start + i as isize, audio.len());
                *value = Complex32::new(audio[position] * model.window[i], 0.0);
            }
            model.fft.process(&mut spectrum);
            for (bin, value) in magnitude.iter_mut().enumerate() {
                let number = spectrum[bin];
                *value = (number.re * number.re + number.im * number.im).sqrt();
            }
            for mel in 0..MEL_BINS {
                let basis = &model.basis[mel * magnitude.len()..(mel + 1) * magnitude.len()];
                let energy: f32 = basis.iter().zip(magnitude.iter()).map(|(a, b)| a * b).sum();
                output[mel * frames + frame] = energy.max(LOG_CLAMP).ln();
            }
        }
        output
    }

    fn sample_audio(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = i.wrapping_mul(129 + seed * 7) % 107_843;
                ((x as f32 / 107_843.0) * 2.0 - 1.0) * 0.83
            })
            .collect()
    }

    #[test]
    fn sparse_mel_matches_full_513_bin_filter_for_irregular_lengths() {
        let mel = RmvpeMelSpectrogram::new();
        assert_eq!(mel.spans.len(), MEL_BINS);
        for (index, &(start, end)) in mel.spans.iter().enumerate() {
            let band = &mel.basis[index * (FFT_SIZE / 2 + 1)..(index + 1) * (FFT_SIZE / 2 + 1)];
            assert!(band[..start].iter().all(|w| *w == 0.0));
            assert!(band[end..].iter().all(|w| *w == 0.0));
        }
        for seed in [0, 1, 9] {
            for len in [513, 801, 3_200, 5_600, 17_123, 32_017] {
                let audio = sample_audio(len, seed);
                let (sparse, frames) = mel.forward(&audio).unwrap();
                let dense = dense_reference(&mel, &audio);
                assert_eq!(frames, len / HOP_LENGTH + 1);
                assert_eq!(dense.len(), sparse.len());
                assert!(
                    dense
                        .iter()
                        .zip(&sparse)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "seed={seed} len={len} max_abs={}",
                    dense
                        .iter()
                        .zip(&sparse)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max)
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn benchmark_sparse_mel_against_dense() {
        let mel = RmvpeMelSpectrogram::new();
        let audio = sample_audio(160_000, 17);
        let (sparse, _) = mel.forward(&audio).unwrap();
        let dense = dense_reference(&mel, &audio);
        assert!(
            sparse
                .iter()
                .zip(&dense)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let mut sparse_times = Vec::new();
        let mut dense_times = Vec::new();
        for i in 0..16 {
            if i % 2 == 0 {
                let start = Instant::now();
                std::hint::black_box(dense_reference(&mel, &audio));
                dense_times.push(start.elapsed());
                let start = Instant::now();
                std::hint::black_box(mel.forward(&audio).unwrap());
                sparse_times.push(start.elapsed());
            } else {
                let start = Instant::now();
                std::hint::black_box(mel.forward(&audio).unwrap());
                sparse_times.push(start.elapsed());
                let start = Instant::now();
                std::hint::black_box(dense_reference(&mel, &audio));
                dense_times.push(start.elapsed());
            }
        }
        fn median(samples: &mut [Duration]) -> f64 {
            samples.sort_unstable();
            samples[samples.len() / 2].as_secs_f64() * 1000.0
        }
        println!(
            "stage57_mel len_samples={} median_dense_ms={:.4} median_sparse_ms={:.4} nonzero_coefficients={} full_coefficients={}",
            audio.len(),
            median(&mut dense_times),
            median(&mut sparse_times),
            mel.spans.iter().map(|(s, e)| e - s).sum::<usize>(),
            MEL_BINS * (FFT_SIZE / 2 + 1)
        );
    }
}
