use crate::{F5Error, Result};

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
