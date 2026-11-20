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
