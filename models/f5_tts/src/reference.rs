use crate::{
    F5Error, Result,
    mel::{SAMPLE_RATE, VocosMel},
    model::MelFrames,
    resample::resample_to_24khz,
};

const TARGET_RMS: f32 = 0.1;
const MAX_REFERENCE_SECONDS: usize = 12;

#[derive(Debug, Clone)]
pub struct ReferenceAudio {
    pub samples: Vec<f32>,
    pub original_rms: f32,
    pub normalization_gain: f32,
}

pub struct PreparedCondition {
    pub frames: MelFrames,
    pub conditioning_mel: Vec<f32>,
    pub original_rms: f32,
}

impl ReferenceAudio {
    pub fn from_channels(channels: &[&[f32]], sample_rate: usize) -> Result<Self> {
        if sample_rate == 0 || sample_rate > 384_000 {
            return Err(F5Error::InvalidConfiguration(
                "Invalid reference sample rate",
            ));
        }
        let Some(first) = channels.first() else {
            return Err(F5Error::InvalidShape(
                "Reference audio contains no channels",
            ));
        };
        let length = first.len().min(MAX_REFERENCE_SECONDS * sample_rate);
        if length <= 512 || channels.iter().any(|channel| channel.len() != first.len()) {
            return Err(F5Error::InvalidShape(
                "Reference audio must have aligned channels and at least 513 samples",
            ));
        }
        if channels
            .iter()
            .any(|channel| channel[..length].iter().any(|sample| !sample.is_finite()))
        {
            return Err(F5Error::InvalidConfiguration(
                "Reference audio samples must be finite",
            ));
        }
        let count = channels.len() as f32;
        let mut samples = Vec::with_capacity(length);
        for index in 0..length {
            samples.push(channels.iter().map(|channel| channel[index]).sum::<f32>() / count);
        }
        let square_sum = samples
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        let original_rms = (square_sum / length as f64).sqrt() as f32;
        let normalization_gain = if original_rms < TARGET_RMS {
            TARGET_RMS / original_rms.max(1e-8)
        } else {
            1.0
        };
        if normalization_gain > 1.0 {
            for sample in &mut samples {
                *sample *= normalization_gain;
            }
        }
        let samples = resample_to_24khz(&samples, sample_rate)?;
        if samples.len() <= 512 {
            return Err(F5Error::InvalidShape(
                "Resampled reference is too short for the mel frontend",
            ));
        }
        Ok(Self {
            samples,
            original_rms,
            normalization_gain,
        })
    }

    pub fn prepare(
        &self,
        extractor: &VocosMel,
        requested_frames: usize,
        token_count: usize,
    ) -> Result<PreparedCondition> {
        let mel = extractor.extract(&self.samples)?;
        let plan = MelFrames::new(mel.frames, requested_frames, token_count)?;
        let conditioning_mel = plan.pad_reference(mel.as_flat_frames())?;
        Ok(PreparedCondition {
            frames: plan,
            conditioning_mel,
            original_rms: self.original_rms,
        })
    }
}

impl PreparedCondition {
    pub fn generated_only<'a>(&self, full_mel: &'a [f32]) -> Result<&'a [f32]> {
        let expected = self
            .frames
            .target
            .checked_mul(100)
            .ok_or(F5Error::InvalidShape("Mel output length overflow"))?;
        if full_mel.len() != expected {
            return Err(F5Error::InvalidShape(
                "Output mel does not match the planned duration",
            ));
        }
        Ok(&full_mel[self.frames.reference * 100..])
    }

    pub fn restoration_gain(&self) -> f32 {
        if self.original_rms < TARGET_RMS {
            self.original_rms / TARGET_RMS
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_accepts_resampled_audio_and_preserves_source_rms() {
        for rate in [16_000_usize, 44_100, 48_000] {
            let left = vec![0.02; rate];
            let right = vec![0.06; rate];
            let audio = ReferenceAudio::from_channels(&[&left, &right], rate).unwrap();
            assert_eq!(audio.samples.len(), SAMPLE_RATE);
            assert!((audio.original_rms - 0.04).abs() < 1e-6);
            assert!((audio.normalization_gain - 2.5).abs() < 1e-5);
            assert!(audio.samples.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn stereo_downmix_rms_and_generated_mel() {
        let left = vec![0.05; 2048];
        let right = vec![-0.01; 2048];
        let ref_audio = ReferenceAudio::from_channels(&[&left, &right], SAMPLE_RATE).unwrap();
        assert!((ref_audio.original_rms - 0.02).abs() < 1e-6);
        assert!((ref_audio.normalization_gain - 5.0).abs() < 1e-5);
        assert!((ref_audio.samples[0] - 0.1).abs() < 1e-6);
        let plan = ref_audio.prepare(&VocosMel::new(), 18, 6).unwrap();
        assert_eq!(plan.frames.reference, 9);
        assert_eq!(plan.frames.target, 18);
        assert!(
            plan.conditioning_mel[900..]
                .iter()
                .all(|value| *value == 0.0)
        );
        let output = vec![2.0; 1800];
        assert_eq!(plan.generated_only(&output).unwrap(), &output[900..]);
        assert!((plan.restoration_gain() - 0.2).abs() < 1e-5);
    }

    #[test]
    fn invalid_channels_and_sample_rate_are_rejected() {
        let a = vec![0.5; 1024];
        let b = vec![0.2; 512];
        assert!(ReferenceAudio::from_channels(&[], SAMPLE_RATE).is_err());
        assert!(ReferenceAudio::from_channels(&[&a, &b], SAMPLE_RATE).is_err());
        assert!(ReferenceAudio::from_channels(&[&a], 16000).is_err());
    }
}
