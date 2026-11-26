use crate::{F5Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedSpectrogram {
    pub channels: usize,
    pub frames: usize,
    pub values: Vec<f32>,
}

impl GeneratedSpectrogram {
    pub fn from_frame_major(frame_major: &[f32], channels: usize) -> Result<Self> {
        if channels == 0 || frame_major.is_empty() || frame_major.len() % channels != 0 {
            return Err(F5Error::InvalidShape("Invalid generated mel dimensions"));
        }
        let frames = frame_major.len() / channels;
        let mut values = vec![0.0; frame_major.len()];
        for frame in 0..frames {
            for channel in 0..channels {
                values[channel * frames + frame] = frame_major[frame * channels + channel];
            }
        }
        Ok(Self {
            channels,
            frames,
            values,
        })
    }

    pub fn concatenate(chunks: &[Self]) -> Result<Self> {
        let first = chunks
            .first()
            .ok_or(F5Error::InvalidShape("No generated mel chunks"))?;
        let channels = first.channels;
        if channels == 0
            || chunks.iter().any(|c| {
                c.channels != channels
                    || c.frames == 0
                    || c.frames.checked_mul(channels) != Some(c.values.len())
            })
        {
            return Err(F5Error::InvalidShape("Incompatible generated mel chunks"));
        }
        let frames = chunks.iter().try_fold(0usize, |acc, c| {
            acc.checked_add(c.frames)
                .ok_or(F5Error::InvalidShape("Generated mel length overflow"))
        })?;
        let length = frames
            .checked_mul(channels)
            .ok_or(F5Error::InvalidShape("Generated mel size overflow"))?;
        let mut values = vec![0.0; length];
        for channel in 0..channels {
            let mut cursor = channel * frames;
            for chunk in chunks {
                let span = &chunk.values[channel * chunk.frames..(channel + 1) * chunk.frames];
                values[cursor..cursor + span.len()].copy_from_slice(span);
                cursor += span.len();
            }
        }
        Ok(Self {
            channels,
            frames,
            values,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceOutput {
    pub waveform: Vec<f32>,
    pub sample_rate: usize,
    pub spectrogram: GeneratedSpectrogram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F5Variant {
    Standard,
    CrossLingual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiTConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub text_dim: usize,
    pub ff_mult: usize,
    pub conv_layers: usize,
    pub mel_channels: usize,
    pub sample_rate: usize,
    pub hop_length: usize,
}

impl DiTConfig {
    pub const V1_BASE: Self = Self {
        dim: 1024,
        depth: 22,
        heads: 16,
        head_dim: 64,
        text_dim: 512,
        ff_mult: 2,
        conv_layers: 4,
        mel_channels: 100,
        sample_rate: 24_000,
        hop_length: 256,
    };

    pub fn verify_v1(self) -> Result<()> {
        if self != Self::V1_BASE {
            return Err(F5Error::InvalidConfiguration(
                "Only the F5TTS_v1_Base DiT architecture is supported",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MelFrames {
    pub reference: usize,
    pub target: usize,
    pub text_tokens: usize,
}

impl MelFrames {
    pub fn new(reference: usize, requested_target: usize, text_tokens: usize) -> Result<Self> {
        if reference == 0 {
            return Err(F5Error::InvalidConfiguration(
                "Reference audio must contain mel frames",
            ));
        }
        let min_target = reference
            .max(text_tokens)
            .checked_add(1)
            .ok_or(F5Error::InvalidConfiguration("Frame count overflow"))?;
        if min_target > 65_536 {
            return Err(F5Error::InvalidConfiguration(
                "Reference or token sequence exceeds the frame limit",
            ));
        }
        let target = requested_target.max(min_target).min(65_536);
        if target <= reference {
            return Err(F5Error::InvalidConfiguration(
                "Reference audio exceeds the frame limit",
            ));
        }
        Ok(Self {
            reference,
            target,
            text_tokens,
        })
    }

    pub fn generated(self) -> usize {
        self.target - self.reference
    }

    pub fn pad_reference(self, mel: &[f32]) -> Result<Vec<f32>> {
        let expected = self
            .reference
            .checked_mul(DiTConfig::V1_BASE.mel_channels)
            .ok_or(F5Error::InvalidShape("Reference mel shape overflow"))?;
        if mel.len() != expected || mel.iter().any(|v| !v.is_finite()) {
            return Err(F5Error::InvalidShape(
                "Reference mel must contain finite [reference, 100] samples",
            ));
        }
        let mut condition = vec![0.0; self.target * DiTConfig::V1_BASE.mel_channels];
        condition[..mel.len()].copy_from_slice(mel);
        Ok(condition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_spectrogram_transposition_and_python_style_batch_concat() {
        let left = GeneratedSpectrogram::from_frame_major(&[1.0, 10.0, 2.0, 20.0], 2).unwrap();
        let right = GeneratedSpectrogram::from_frame_major(&[3.0, 30.0], 2).unwrap();
        assert_eq!(left.values, [1.0, 2.0, 10.0, 20.0]);
        let joined = GeneratedSpectrogram::concatenate(&[left, right]).unwrap();
        assert_eq!(joined.frames, 3);
        assert_eq!(joined.values, [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        assert!(GeneratedSpectrogram::from_frame_major(&[1.0], 2).is_err());
        assert!(GeneratedSpectrogram::concatenate(&[]).is_err());
    }

    #[test]
    fn version_one_model_shape() {
        assert!(DiTConfig::V1_BASE.verify_v1().is_ok());
        assert_eq!(
            DiTConfig::V1_BASE.dim,
            DiTConfig::V1_BASE.heads * DiTConfig::V1_BASE.head_dim
        );
    }

    #[test]
    fn reference_is_preserved_and_target_has_one_extra_frame() {
        let plan = MelFrames::new(200, 50, 250).unwrap();
        assert_eq!(plan.target, 251);
        assert_eq!(plan.generated(), 51);
    }

    #[test]
    fn reference_audio_is_preserved_only_in_prefix() {
        let shape = MelFrames::new(2, 4, 1).unwrap();
        let padded = shape.pad_reference(&vec![1.0; 200]).unwrap();
        assert_eq!(padded.len(), 400);
        assert!(padded[..200].iter().all(|&x| x == 1.0));
        assert!(padded[200..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn no_zero_reference() {
        assert!(MelFrames::new(0, 100, 3).is_err());
    }
}
