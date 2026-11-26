use crate::{F5Error, Result, fft::transform};

pub const SAMPLE_RATE: usize = 24_000;
pub const FFT_SIZE: usize = 1024;
pub const HOP_SIZE: usize = 256;
pub const MEL_CHANNELS: usize = 100;
const FFT_BINS: usize = FFT_SIZE / 2 + 1;
const LOG_FLOOR: f32 = 1e-5;

#[derive(Debug, Clone)]
pub struct VocosMel {
    filters: Vec<f32>,
    window: [f32; FFT_SIZE],
}

impl Default for VocosMel {
    fn default() -> Self {
        Self::new()
    }
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn reflected_index(index: isize, length: usize) -> usize {
    let max = (length - 1) as isize;
    let period = max * 2;
    let wrapped = index.rem_euclid(period);
    (if wrapped > max {
        period - wrapped
    } else {
        wrapped
    }) as usize
}

impl VocosMel {
    pub fn new() -> Self {
        let mut window = [0.0_f32; FFT_SIZE];
        for (index, value) in window.iter_mut().enumerate() {
            *value =
                0.5 - 0.5 * (2.0 * core::f32::consts::PI * index as f32 / FFT_SIZE as f32).cos();
        }
        let mel_start = hz_to_mel(0.0);
        let mel_end = hz_to_mel(SAMPLE_RATE as f32 / 2.0);
        let frequencies: Vec<f32> = (0..MEL_CHANNELS + 2)
            .map(|bin| {
                let mel =
                    mel_start + (mel_end - mel_start) * bin as f32 / (MEL_CHANNELS + 1) as f32;
                mel_to_hz(mel)
            })
            .collect();
        let mut filters = vec![0.0; MEL_CHANNELS * FFT_BINS];
        for channel in 0..MEL_CHANNELS {
            let lower = frequencies[channel];
            let center = frequencies[channel + 1];
            let upper = frequencies[channel + 2];
            for bin in 0..FFT_BINS {
                let frequency = bin as f32 * SAMPLE_RATE as f32 / FFT_SIZE as f32;
                let rise = (frequency - lower) / (center - lower);
                let fall = (upper - frequency) / (upper - center);
                filters[channel * FFT_BINS + bin] = rise.min(fall).max(0.0);
            }
        }
        Self { filters, window }
    }

    pub fn extract(&self, samples: &[f32]) -> Result<MelFrames> {
        if samples.len() <= FFT_SIZE / 2 {
            return Err(F5Error::InvalidShape(
                "Vocos mel requires more than 512 input samples for reflect padding",
            ));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(F5Error::InvalidConfiguration(
                "Reference audio contains non-finite samples",
            ));
        }
        let frames = samples.len() / HOP_SIZE + 1;
        let capacity = frames
            .checked_mul(MEL_CHANNELS)
            .ok_or(F5Error::InvalidShape("Mel output size overflow"))?;
        let mut values = vec![0.0; capacity];
        let mut real = [0.0_f32; FFT_SIZE];
        let mut imag = [0.0_f32; FFT_SIZE];
        let mut magnitudes = [0.0_f32; FFT_BINS];
        for frame in 0..frames {
            let center = frame * HOP_SIZE;
            for bin in 0..FFT_SIZE {
                let position = center as isize + bin as isize - (FFT_SIZE / 2) as isize;
                real[bin] = samples[reflected_index(position, samples.len())] * self.window[bin];
                imag[bin] = 0.0;
            }
            transform(&mut real, &mut imag, false);
            for bin in 0..FFT_BINS {
                magnitudes[bin] = real[bin].hypot(imag[bin]);
            }
            for channel in 0..MEL_CHANNELS {
                let mut energy = 0.0;
                for bin in 0..FFT_BINS {
                    energy += self.filters[channel * FFT_BINS + bin] * magnitudes[bin];
                }
                values[frame * MEL_CHANNELS + channel] = energy.max(LOG_FLOOR).ln();
            }
        }
        Ok(MelFrames { frames, values })
    }
}

/// Slaney-normalized, center-free mel frontend used by BigVGAN v2 and the
/// corresponding F5-TTS checkpoint. The Vocos frontend is not interchangeable.
#[derive(Debug, Clone)]
pub struct BigVganMel {
    filters: Vec<f32>,
    window: [f32; FFT_SIZE],
}

#[derive(Debug, Clone)]
pub enum MelBackend {
    Vocos(VocosMel),
    BigVgan(BigVganMel),
}

impl MelExtractor for MelBackend {
    fn extract(&self, samples: &[f32]) -> Result<MelFrames> {
        match self {
            Self::Vocos(v) => v.extract(samples),
            Self::BigVgan(v) => v.extract(samples),
        }
    }
}

pub trait MelExtractor {
    fn extract(&self, samples: &[f32]) -> Result<MelFrames>;
}

impl MelExtractor for VocosMel {
    fn extract(&self, samples: &[f32]) -> Result<MelFrames> {
        VocosMel::extract(self, samples)
    }
}

impl Default for BigVganMel {
    fn default() -> Self {
        Self::new()
    }
}

fn slaney_mel(hz: f64) -> f64 {
    const LINEAR_HZ_PER_MEL: f64 = 200.0 / 3.0;
    if hz < 1000.0 {
        hz / LINEAR_HZ_PER_MEL
    } else {
        15.0 + (hz / 1000.0).ln() / (6.4_f64.ln() / 27.0)
    }
}

fn slaney_hz(mel: f64) -> f64 {
    const LINEAR_HZ_PER_MEL: f64 = 200.0 / 3.0;
    if mel < 15.0 {
        mel * LINEAR_HZ_PER_MEL
    } else {
        1000.0 * ((mel - 15.0) * (6.4_f64.ln() / 27.0)).exp()
    }
}

impl BigVganMel {
    pub fn new() -> Self {
        let mut window = [0.0; FFT_SIZE];
        for (index, value) in window.iter_mut().enumerate() {
            *value = 0.5
                - 0.5 * (2.0 * std::f64::consts::PI * index as f64 / FFT_SIZE as f64).cos() as f32;
        }
        let max_mel = slaney_mel((SAMPLE_RATE / 2) as f64);
        let mel_hz: Vec<f64> = (0..MEL_CHANNELS + 2)
            .map(|i| slaney_hz(i as f64 * max_mel / (MEL_CHANNELS + 1) as f64))
            .collect();
        let mut filters = vec![0.0; MEL_CHANNELS * FFT_BINS];
        for channel in 0..MEL_CHANNELS {
            let low = mel_hz[channel];
            let center = mel_hz[channel + 1];
            let high = mel_hz[channel + 2];
            let slaney_norm = 2.0 / (high - low);
            for bin in 0..FFT_BINS {
                let hz = bin as f64 * SAMPLE_RATE as f64 / FFT_SIZE as f64;
                let up = (hz - low) / (center - low);
                let down = (high - hz) / (high - center);
                filters[channel * FFT_BINS + bin] = (up.min(down).max(0.0) * slaney_norm) as f32;
            }
        }
        Self { filters, window }
    }

    pub fn extract(&self, samples: &[f32]) -> Result<MelFrames> {
        const PAD: usize = (FFT_SIZE - HOP_SIZE) / 2;
        if samples.len() <= PAD {
            return Err(F5Error::InvalidShape(
                "BigVGAN mel reflect padding requires >384 samples",
            ));
        }
        if samples.iter().any(|v| !v.is_finite()) {
            return Err(F5Error::InvalidConfiguration(
                "Reference audio contains non-finite samples",
            ));
        }
        let padded_len = samples
            .len()
            .checked_add(PAD * 2)
            .ok_or(F5Error::InvalidShape("BigVGAN mel frame count overflow"))?;
        let frames = 1 + (padded_len - FFT_SIZE) / HOP_SIZE;
        let output_len = frames
            .checked_mul(MEL_CHANNELS)
            .ok_or(F5Error::InvalidShape("BigVGAN mel size overflow"))?;
        let mut values = vec![0.0_f32; output_len];
        let mut real = [0.0_f32; FFT_SIZE];
        let mut imag = [0.0_f32; FFT_SIZE];
        let mut magnitudes = [0.0_f32; FFT_BINS];
        for frame in 0..frames {
            for i in 0..FFT_SIZE {
                let position = frame as isize * HOP_SIZE as isize + i as isize - PAD as isize;
                real[i] = samples[reflected_index(position, samples.len())] * self.window[i];
                imag[i] = 0.0;
            }
            transform(&mut real, &mut imag, false);
            for k in 0..FFT_BINS {
                magnitudes[k] = (real[k] * real[k] + imag[k] * imag[k] + 1e-9).sqrt();
            }
            for channel in 0..MEL_CHANNELS {
                let mut energy = 0.0_f32;
                for k in 0..FFT_BINS {
                    energy += self.filters[channel * FFT_BINS + k] * magnitudes[k];
                }
                values[frame * MEL_CHANNELS + channel] = energy.max(LOG_FLOOR).ln();
            }
        }
        Ok(MelFrames { frames, values })
    }
}

impl MelExtractor for BigVganMel {
    fn extract(&self, samples: &[f32]) -> Result<MelFrames> {
        BigVganMel::extract(self, samples)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MelFrames {
    pub frames: usize,
    pub values: Vec<f32>,
}

impl MelFrames {
    pub fn as_flat_frames(&self) -> &[f32] {
        &self.values
    }

    pub fn into_padded(self, duration: usize) -> Result<Vec<f32>> {
        if duration < self.frames {
            return Err(F5Error::InvalidShape(
                "Generation duration cannot be shorter than reference mel",
            ));
        }
        let mut result = self.values;
        let target = duration
            .checked_mul(MEL_CHANNELS)
            .ok_or(F5Error::InvalidShape("Mel pad size overflow"))?;
        result.resize(target, 0.0);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bigvgan_uses_center_free_frames() {
        let extractor = BigVganMel::new();
        let mel = extractor.extract(&vec![0.0_f32; 2048]).unwrap();
        assert_eq!(mel.frames, 8);
        assert_eq!(mel.values.len(), 800);
        assert!(mel.values.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn bigvgan_matches_original_torch_librosa_reference() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/bigvgan_mel_expected.json"))
                .unwrap();
        let extractor = BigVganMel::new();
        for fixture in fixture.as_array().unwrap() {
            let length = fixture["length"].as_u64().unwrap() as usize;
            let samples: Vec<f32> = (0..length)
                .map(|i| ((i * 37 % 101) as i32 - 50) as f32 / 128.0)
                .collect();
            let expected: Vec<f32> = fixture["mel"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let found = extractor.extract(&samples).unwrap();
            assert_eq!(found.frames, fixture["frames"].as_u64().unwrap() as usize);
            assert_eq!(found.values.len(), expected.len());
            let maximum = found
                .values
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                maximum < 5e-3,
                "BigVGAN mel PyTorch differential max_abs={maximum}"
            );
        }
    }

    #[test]
    fn mel_frame_count_and_padded_audio_shape() {
        let mel = VocosMel::new();
        let data = vec![0.0; 2048];
        let frames = mel.extract(&data).unwrap();
        assert_eq!(frames.frames, 9);
        assert_eq!(frames.values.len(), 900);
        assert!(frames.values.iter().all(|value| value.is_finite()));
        let padded = frames.into_padded(12).unwrap();
        assert_eq!(padded.len(), 1200);
        assert!(padded[900..].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn invalid_waveform_rejected() {
        let mel = VocosMel::new();
        assert!(mel.extract(&vec![0.0; 512]).is_err());
        let mut invalid = vec![0.0; 2048];
        invalid[1] = f32::NAN;
        assert!(mel.extract(&invalid).is_err());
    }

    #[test]
    fn torch_vocos_melspec_numerical_reference() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/vocos_mel_expected.json"))
                .unwrap();
        let input: Vec<f32> = data["wave"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expected: Vec<f32> = data["mel"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let result = VocosMel::new().extract(&input).unwrap();
        assert_eq!(result.frames, data["frames"].as_u64().unwrap() as usize);
        assert_eq!(result.values.len(), expected.len());
        let worst = result
            .values
            .iter()
            .zip(&expected)
            .map(|(&actual, &reference)| (actual - reference).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 2e-3, "Torch Vocos mel parity max_abs={worst}");
    }
}
