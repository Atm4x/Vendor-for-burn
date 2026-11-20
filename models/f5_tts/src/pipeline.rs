use std::{collections::BTreeMap, path::Path, sync::Arc};

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::{
    F5Error, Result,
    backend::cubecl::integrate,
    cfm::SamplingConfig,
    checkpoint::Checkpoint,
    dit_model::F5DiT,
    mel::VocosMel,
    pinyin::ChinesePinyin,
    reference::ReferenceAudio,
    standard_chunking::plan_standard_chunks,
    text_frontend::{F5TextFrontend, NoPinyin, PinyinProvider},
    vocab::Vocabulary,
    vocos::{VocosConfig, VocosDecoder},
};

pub struct StandardF5Runtime {
    device: Device,
    mel: VocosMel,
    dit: F5DiT,
    vocoder: VocosDecoder,
    vocabulary: Vocabulary,
    native_pinyin: Arc<ChinesePinyin>,
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

fn validate_initial_noise(noise: Vec<f32>, required: usize) -> Result<Vec<f32>> {
    if noise.len() != required || noise.iter().any(|value| !value.is_finite()) {
        return Err(F5Error::InvalidShape(
            "F5 initial noise must contain exactly [target_frames, 100] finite values",
        ));
    }
    Ok(noise)
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
        let native_pinyin = Arc::new(ChinesePinyin::bundled()?);
        let dit = F5DiT::from_checkpoint(&f5_checkpoint, vocabulary.size(), &device)?;
        let vocoder = VocosDecoder::from_checkpoint(&vocos_checkpoint, vocos_config, &device)?;
        Ok(Self {
            device,
            mel: VocosMel::new(),
            dit,
            vocoder,
            vocabulary,
            native_pinyin,
        })
    }

    /// Standard F5 synthesis from text and a transcribed reference. This path matches
    /// the original reference+generation conditioning and UTF-8 duration heuristic.
    pub fn replace_native_pinyin(&mut self, dictionary: ChinesePinyin) {
        self.native_pinyin = Arc::new(dictionary);
    }

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
        let native_pinyin = Arc::clone(&self.native_pinyin);
        self.synthesize_text_with_pinyin(
            reference,
            reference_text,
            generated_text,
            native_pinyin,
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
        self.synthesize_chunked_with_pinyin(
            reference,
            reference_text,
            generated_text,
            pinyin,
            speed,
            fixed_seconds,
            0.15,
            seed,
            sampling,
        )
    }

    pub fn synthesize_chunked_with_pinyin<P: PinyinProvider>(
        &mut self,
        reference: &ReferenceAudio,
        reference_text: &str,
        generated_text: &str,
        pinyin: P,
        speed: f32,
        fixed_seconds: Option<f32>,
        cross_fade_seconds: f32,
        seed: u64,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        let planned = plan_standard_chunks(
            generated_text,
            reference_text,
            reference.samples.len(),
            speed,
            fixed_seconds,
            cross_fade_seconds,
        )?;
        let vocabulary = self.vocabulary.clone();
        let frontend = F5TextFrontend::new(&vocabulary, pinyin);
        let mut generator = StandardNormal::new(seed);
        let mut waves = Vec::with_capacity(planned.len());
        for chunk in planned {
            let tokens = frontend.standard_tokens(reference_text, &chunk.text)?;
            let wave = self.synthesize_with_rng(
                reference,
                &tokens,
                chunk.total_frames,
                sampling,
                &mut generator,
            )?;
            waves.push(wave);
        }
        crate::cross_lingual::cross_fade_waves(&waves, cross_fade_seconds as f64)
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
            let chunks = plan_standard_chunks(
                &segment.text,
                voice.transcript,
                voice.audio.samples.len(),
                speed,
                None,
                0.15,
            )?;
            let vocabulary = self.vocabulary.clone();
            let frontend = F5TextFrontend::new(&vocabulary, Arc::clone(&self.native_pinyin));
            let mut chunk_waves = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                let tokens = frontend.standard_tokens(voice.transcript, &chunk.text)?;
                let wave = self.synthesize_with_rng(
                    voice.audio,
                    &tokens,
                    chunk.total_frames,
                    sampling,
                    &mut rng,
                )?;
                chunk_waves.push(wave);
            }
            result.extend(crate::cross_lingual::cross_fade_waves(&chunk_waves, 0.15)?);
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

    pub fn synthesize_with_noise(
        &mut self,
        reference: &ReferenceAudio,
        text_tokens: &[i32],
        total_frames: usize,
        sampling: SamplingConfig,
        initial_noise: &[f32],
    ) -> Result<Vec<f32>> {
        self.synthesize_from_noise_source(reference, text_tokens, total_frames, sampling, |count| {
            validate_initial_noise(initial_noise.to_vec(), count)
        })
    }

    fn synthesize_with_rng(
        &mut self,
        reference: &ReferenceAudio,
        text_tokens: &[i32],
        total_frames: usize,
        sampling: SamplingConfig,
        generator: &mut StandardNormal,
    ) -> Result<Vec<f32>> {
        self.synthesize_from_noise_source(reference, text_tokens, total_frames, sampling, |count| {
            Ok((0..count).map(|_| generator.sample()).collect())
        })
    }

    fn synthesize_from_noise_source(
        &mut self,
        reference: &ReferenceAudio,
        text_tokens: &[i32],
        total_frames: usize,
        sampling: SamplingConfig,
        create_noise: impl FnOnce(usize) -> Result<Vec<f32>>,
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
        let values = validate_initial_noise(create_noise(count)?, count)?;
        let noise = Tensor::from_data(
            TensorData::new(values, [1, frames.target, 100]),
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
    fn deterministic_noise_override_rejects_invalid_shape_and_nonfinite_samples() {
        assert_eq!(
            validate_initial_noise(vec![0.0; 300], 300).unwrap().len(),
            300
        );
        assert!(validate_initial_noise(vec![0.0; 299], 300).is_err());
        assert!(validate_initial_noise(vec![0.0, f32::NAN], 2).is_err());
        assert!(validate_initial_noise(vec![f32::INFINITY], 1).is_err());
    }

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
    pub speed: f64,
    pub fixed_seconds: Option<f64>,
    pub cross_fade_seconds: f64,
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

    /// Native mixed Chinese/English path with bundled lexicon and bounded tone sandhi.
    /// The dictionary can be overridden for precise polyphonic word coverage.
    pub fn synthesize_chinese(
        &mut self,
        reference: &ReferenceAudio,
        text: &str,
        syllables: &impl crate::cross_lingual::EnglishSyllableCounter,
        options: CrossLingualGenerateOptions,
    ) -> Result<Vec<f32>> {
        let pinyin = Arc::clone(&self.base.native_pinyin);
        self.synthesize_with_pinyin(reference, text, syllables, pinyin, options)
    }

    pub fn replace_native_pinyin(&mut self, dictionary: ChinesePinyin) {
        self.base.native_pinyin = Arc::new(dictionary);
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
