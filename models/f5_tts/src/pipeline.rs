use std::{collections::BTreeMap, path::Path};

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::{
    F5Error, Result,
    backend::cubecl::integrate,
    cfm::SamplingConfig,
    checkpoint::Checkpoint,
    dit_model::F5DiT,
    mel::VocosMel,
    reference::ReferenceAudio,
    text_frontend::{F5TextFrontend, NoPinyin, PinyinProvider, standard_total_frames},
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

pub struct VoiceReference<'a> {
    pub audio: &'a ReferenceAudio,
    pub transcript: &'a str,
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

    /// Standard F5 synthesis from text and a transcribed reference. This path matches
    /// the original reference+generation conditioning and UTF-8 duration heuristic.
    pub fn synthesize_text(
        &mut self,
        reference: &ReferenceAudio,
        reference_text: &str,
        generated_text: &str,
        speed: f32,
        fixed_seconds: Option<f32>,
        seed: u64,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        self.synthesize_text_with_pinyin(
            reference,
            reference_text,
            generated_text,
            NoPinyin,
            speed,
            fixed_seconds,
            seed,
            sampling,
        )
    }

    pub fn synthesize_text_with_pinyin<P: PinyinProvider>(
        &mut self,
        reference: &ReferenceAudio,
        reference_text: &str,
        generated_text: &str,
        pinyin: P,
        speed: f32,
        fixed_seconds: Option<f32>,
        seed: u64,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        let tokens = F5TextFrontend::new(&self.vocabulary, pinyin)
            .standard_tokens(reference_text, generated_text)?;
        let reference_frames = reference.samples.len() / 256;
        let target = standard_total_frames(
            reference_frames,
            reference_text,
            generated_text,
            speed,
            fixed_seconds,
        )?;
        self.synthesize(
            reference,
            &tokens,
            GenerateOptions {
                total_frames: target,
                seed,
                sampling,
            },
        )
    }

    pub fn synthesize_script(
        &mut self,
        script: &str,
        voices: &BTreeMap<String, VoiceReference<'_>>,
        speed: f32,
        seed: u64,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        let Some(main) = voices.get("main") else {
            return Err(F5Error::InvalidConfiguration(
                "Multi-voice F5 requires a main reference",
            ));
        };
        let segments = crate::text_frontend::parse_voice_segments(script);
        if segments.is_empty() {
            return Err(F5Error::InvalidConfiguration("Voice script is empty"));
        }
        let mut rng = StandardNormal::new(seed);
        let mut result = Vec::new();
        for segment in segments {
            let voice = voices.get(&segment.voice).unwrap_or(main);
            let tokens = F5TextFrontend::new(&self.vocabulary, NoPinyin)
                .standard_tokens(voice.transcript, &segment.text)?;
            let total_frames = standard_total_frames(
                voice.audio.samples.len() / 256,
                voice.transcript,
                &segment.text,
                speed,
                None,
            )?;
            let wave =
                self.synthesize_with_rng(voice.audio, &tokens, total_frames, sampling, &mut rng)?;
            result.extend(wave);
        }
        Ok(result)
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
        let mut rng = StandardNormal::new(options.seed);
        self.synthesize_with_rng(
            reference,
            text_tokens,
            options.total_frames,
            options.sampling,
            &mut rng,
        )
    }

    fn synthesize_with_rng(
        &mut self,
        reference: &ReferenceAudio,
        text_tokens: &[i32],
        total_frames: usize,
        sampling: SamplingConfig,
        generator: &mut StandardNormal,
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
        let prepared = reference.prepare(&self.mel, total_frames, text_tokens.len())?;
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
            sampling,
        )?;
        let generated = final_mel.slice([0..1, frames.reference..frames.target, 0..100]);
        self.vocoder.decode(generated, restoration_gain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_rng_stream_serves_all_chunks_without_reseeding() {
        let mut sequential = StandardNormal::new(123);
        let first_chunk: Vec<_> = (0..7).map(|_| sequential.sample()).collect();
        let second_chunk: Vec<_> = (0..5).map(|_| sequential.sample()).collect();
        let mut uninterrupted = StandardNormal::new(123);
        let full: Vec<_> = (0..12).map(|_| uninterrupted.sample()).collect();
        assert_eq!(&full[..7], first_chunk);
        assert_eq!(&full[7..], second_chunk);
        assert_ne!(second_chunk[0], StandardNormal::new(123).sample());
    }

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

pub struct CrossLingualF5Runtime {
    base: StandardF5Runtime,
    speed: crate::speed_predictor::CrossLingualSpeedPredictor,
}

#[derive(Debug, Clone, Copy)]
pub struct CrossLingualGenerateOptions {
    pub speed: f32,
    pub fixed_seconds: Option<f32>,
    pub cross_fade_seconds: f32,
    pub seed: u64,
    pub sampling: SamplingConfig,
}

impl CrossLingualF5Runtime {
    pub fn load(
        f5_checkpoint: impl AsRef<Path>,
        vocos_checkpoint: impl AsRef<Path>,
        speed_checkpoint: impl AsRef<Path>,
        vocabulary: impl AsRef<Path>,
        vocos_config: VocosConfig,
        device: Device,
    ) -> Result<Self> {
        let base = StandardF5Runtime::load(
            f5_checkpoint,
            vocos_checkpoint,
            vocabulary,
            vocos_config,
            device.clone(),
        )?;
        let speed = crate::speed_predictor::CrossLingualSpeedPredictor::from_checkpoint(
            &Checkpoint::open(speed_checkpoint)?,
            &device,
        )?;
        Ok(Self { base, speed })
    }

    pub fn predict_speed(&self, reference: &ReferenceAudio) -> Result<f32> {
        self.speed
            .predict_audio(&reference.samples, &self.base.device)
    }

    pub fn synthesize_text<E>(
        &mut self,
        reference: &ReferenceAudio,
        text: &str,
        syllables: &impl crate::cross_lingual::EnglishSyllableCounter,
        mut tokenize: E,
        options: CrossLingualGenerateOptions,
    ) -> Result<Vec<f32>>
    where
        E: FnMut(&str) -> Result<Vec<i32>>,
    {
        let predicted_speed = self.predict_speed(reference)?;
        let chunks = crate::cross_lingual::plan_cross_lingual_chunks(
            text,
            reference.samples.len(),
            predicted_speed,
            options.speed,
            options.fixed_seconds,
            syllables,
        )?;
        if chunks.is_empty() {
            return Err(F5Error::InvalidShape(
                "Cross-Lingual generated text is empty",
            ));
        }
        let mut waves = Vec::with_capacity(chunks.len());
        let mut rng = StandardNormal::new(options.seed);
        for chunk in &chunks {
            let tokens = tokenize(&chunk.text)?;
            let audio = self.base.synthesize_with_rng(
                reference,
                &tokens,
                chunk.total_frames,
                options.sampling,
                &mut rng,
            )?;
            waves.push(audio);
        }
        crate::cross_lingual::cross_fade_waves(&waves, options.cross_fade_seconds)
    }

    /// Native English Cross-Lingual path without a caller-provided tokenizer.
    /// Han pronunciation must use a pinyin-aware frontend instead.
    pub fn synthesize_english(
        &mut self,
        reference: &ReferenceAudio,
        text: &str,
        syllables: &impl crate::cross_lingual::EnglishSyllableCounter,
        options: CrossLingualGenerateOptions,
    ) -> Result<Vec<f32>> {
        self.synthesize_with_pinyin(reference, text, syllables, NoPinyin, options)
    }

    pub fn synthesize_with_pinyin<P: PinyinProvider>(
        &mut self,
        reference: &ReferenceAudio,
        text: &str,
        syllables: &impl crate::cross_lingual::EnglishSyllableCounter,
        pinyin: P,
        options: CrossLingualGenerateOptions,
    ) -> Result<Vec<f32>> {
        let vocabulary = self.base.vocabulary.clone();
        let frontend = F5TextFrontend::new(&vocabulary, pinyin);
        self.synthesize_text(
            reference,
            text,
            syllables,
            |part| frontend.cross_lingual_tokens(part),
            options,
        )
    }

    pub fn synthesize_wav<E>(
        &mut self,
        reference: &ReferenceAudio,
        text: &str,
        syllables: &impl crate::cross_lingual::EnglishSyllableCounter,
        tokenize: E,
        options: CrossLingualGenerateOptions,
    ) -> Result<Vec<u8>>
    where
        E: FnMut(&str) -> Result<Vec<i32>>,
    {
        crate::wav::mono_pcm16(
            &self.synthesize_text(reference, text, syllables, tokenize, options)?,
        )
    }
}
