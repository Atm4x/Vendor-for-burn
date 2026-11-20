use std::path::Path;

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::{
    F5Error, Result,
    backend::cubecl::integrate,
    cfm::SamplingConfig,
    checkpoint::Checkpoint,
    dit_model::F5DiT,
    mel::VocosMel,
    reference::ReferenceAudio,
    vocab::Vocabulary,
    vocos::{VocosConfig, VocosDecoder},
};

pub struct StandardF5Runtime {
    device: Device,
    mel: VocosMel,
    dit: F5DiT,
    vocoder: VocosDecoder,
    vocabulary: Vocabulary,
}

#[derive(Debug, Clone, Copy)]
pub struct GenerateOptions {
    pub total_frames: usize,
    pub seed: u64,
    pub sampling: SamplingConfig,
}

struct StandardNormal {
    state: u64,
    spare: Option<f32>,
}

impl StandardNormal {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0x9e37_79b9_7f4a_7c15 {
                0x2545_f491_4f6c_dd1d
            } else {
                seed ^ 0x9e37_79b9_7f4a_7c15
            },
            spare: None,
        }
    }

    fn uniform(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let value = self.state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ((value >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    fn sample(&mut self) -> f32 {
        if let Some(sample) = self.spare.take() {
            return sample;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = 2.0 * core::f64::consts::PI * self.uniform();
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }
}

impl StandardF5Runtime {
    pub fn load(
        f5_checkpoint: impl AsRef<Path>,
        vocos_checkpoint: impl AsRef<Path>,
        vocabulary: impl AsRef<Path>,
        vocos_config: VocosConfig,
        device: Device,
    ) -> Result<Self> {
        let vocabulary = Vocabulary::from_path(vocabulary)?;
        let f5_checkpoint = Checkpoint::open(f5_checkpoint)?;
        let vocos_checkpoint = Checkpoint::open(vocos_checkpoint)?;
        let dit = F5DiT::from_checkpoint(&f5_checkpoint, vocabulary.size(), &device)?;
        let vocoder = VocosDecoder::from_checkpoint(&vocos_checkpoint, vocos_config, &device)?;
        Ok(Self {
            device,
            mel: VocosMel::new(),
            dit,
            vocoder,
            vocabulary,
        })
    }

    /// `text_tokens` must already contain the F5-normalized reference and generated text.
    /// Text-to-pinyin normalization is intentionally outside this inference runtime.
    pub fn synthesize_wav(
        &mut self,
        reference: &ReferenceAudio,
        text_tokens: &[i32],
        options: GenerateOptions,
    ) -> Result<Vec<u8>> {
        crate::wav::mono_pcm16(&self.synthesize(reference, text_tokens, options)?)
    }

    pub fn synthesize(
        &mut self,
        reference: &ReferenceAudio,
        text_tokens: &[i32],
        options: GenerateOptions,
    ) -> Result<Vec<f32>> {
        if text_tokens.is_empty() || text_tokens.len() > 8192 {
            return Err(F5Error::InvalidShape(
                "F5 token count is outside valid limits",
            ));
        }
        if text_tokens
            .iter()
            .any(|token| *token < 0 || *token as usize >= self.vocabulary.size())
        {
            return Err(F5Error::InvalidShape(
                "F5 token ID exceeds model vocabulary",
            ));
        }
        let prepared = reference.prepare(&self.mel, options.total_frames, text_tokens.len())?;
        let frames = prepared.frames;
        let restoration_gain = prepared.restoration_gain();
        let condition = Tensor::from_data(
            TensorData::new(prepared.conditioning_mel, [1, frames.target, 100]),
            &self.device,
        );
        let count = frames
            .target
            .checked_mul(100)
            .ok_or(F5Error::InvalidShape("F5 noise tensor size overflow"))?;
        let mut generator = StandardNormal::new(options.seed);
        let noise = Tensor::from_data(
            TensorData::new(
                (0..count).map(|_| generator.sample()).collect::<Vec<_>>(),
                [1, frames.target, 100],
            ),
            &self.device,
        );
        let text = Tensor::<2, Int>::from_data(
            TensorData::new(text_tokens.to_vec(), [1, text_tokens.len()]),
            &self.device,
        );
        let final_mel = integrate(
            &mut self.dit,
            noise,
            condition,
            text,
            frames.reference,
            options.sampling,
        )?;
        let generated = final_mel.slice([0..1, frames.reference..frames.target, 0..100]);
        self.vocoder.decode(generated, restoration_gain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_gaussian_is_deterministic_and_finite() {
        let mut a = StandardNormal::new(42);
        let mut b = StandardNormal::new(42);
        let mut different = StandardNormal::new(11);
        let samples: Vec<f32> = (0..2048).map(|_| a.sample()).collect();
        assert_eq!(samples, (0..2048).map(|_| b.sample()).collect::<Vec<_>>());
        assert_ne!(samples[0], different.sample());
        assert!(samples.iter().all(|x| x.is_finite()));
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.15);
    }
}
