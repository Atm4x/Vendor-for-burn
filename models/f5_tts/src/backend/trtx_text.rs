//! F5-specific text and voice orchestration for TensorRT-RTX. Graph selection
//! is delegated to a prewarmed shape-specific kernel; there is no CubeCL fallback.

use std::collections::{BTreeMap, VecDeque};

use crate::{
    F5Error, Result,
    cfm::{CfmSampleMode, OdeMethod, SamplingConfig},
    cross_lingual::{EnglishSyllableCounter, cross_fade_waves, plan_cross_lingual_chunks},
    model::{GeneratedSpectrogram, InferenceOutput},
    noise::StandardNormal,
    reference::ReferenceAudio,
    speech_edit::SpeechEditPlan,
    standard_chunking::{emit_pcm_chunks, plan_standard_chunks},
    text_frontend::{F5TextFrontend, PinyinProvider, parse_voice_segments},
    vocab::Vocabulary,
};

use super::trtx_session::{CudaTransport, SynthesisInputs, TrtExecutor, TrtF5Session};

pub trait TrtKernelFactory {
    fn validate_model(&mut self) -> Result<()> {
        Ok(())
    }

    fn build(
        &mut self,
        reference_frames: usize,
        target_frames: usize,
    ) -> Result<Box<dyn TrtChunkKernel>>;
}

pub struct TrtWarmBank {
    shapes: BTreeMap<(usize, usize), Box<dyn TrtChunkKernel>>,
    speed_shape: Option<(usize, usize)>,
    factory: Option<Box<dyn TrtKernelFactory>>,
    capacity: usize,
    usage: VecDeque<(usize, usize)>,
}

impl TrtWarmBank {
    pub fn new() -> Self {
        Self {
            shapes: BTreeMap::new(),
            speed_shape: None,
            factory: None,
            capacity: usize::MAX,
            usage: VecDeque::new(),
        }
    }

    pub fn with_factory(factory: Box<dyn TrtKernelFactory>, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(F5Error::InvalidConfiguration(
                "F5 engine cache capacity must be nonzero",
            ));
        }
        Ok(Self {
            shapes: BTreeMap::new(),
            speed_shape: None,
            factory: Some(factory),
            capacity,
            usage: VecDeque::new(),
        })
    }

    fn touch(&mut self, key: (usize, usize)) {
        self.usage.retain(|item| *item != key);
        self.usage.push_back(key);
    }

    fn ensure(&mut self, key: (usize, usize)) -> Result<()> {
        if self.shapes.contains_key(&key) {
            self.touch(key);
            return Ok(());
        }
        let (reference_frames, target_frames) = key;
        if target_frames <= reference_frames || target_frames > 65_536 {
            return Err(F5Error::InvalidShape("Invalid F5 engine shape"));
        }
        let factory = self.factory.as_mut().ok_or(F5Error::InvalidShape(
            "No warmed F5 engine matches requested shape and no factory is installed",
        ))?;
        let engine = factory.build(reference_frames, target_frames)?;
        while self.shapes.len() >= self.capacity {
            if let Some(oldest) = self.usage.pop_front() {
                self.shapes.remove(&oldest);
                if self.speed_shape == Some(oldest) {
                    self.speed_shape = None;
                }
            } else {
                return Err(F5Error::ModelExecution(
                    "F5 engine cache lost LRU accounting".into(),
                ));
            }
        }
        self.shapes.insert(key, engine);
        self.touch(key);
        Ok(())
    }

    pub fn insert(
        &mut self,
        reference_frames: usize,
        target_frames: usize,
        engine: Box<dyn TrtChunkKernel>,
    ) -> Result<()> {
        if target_frames <= reference_frames
            || self.shapes.contains_key(&(reference_frames, target_frames))
        {
            return Err(F5Error::InvalidConfiguration(
                "Duplicate or invalid warmed TensorRT F5 shape",
            ));
        }
        if self.shapes.len() >= self.capacity {
            return Err(F5Error::InvalidConfiguration("F5 engine cache is full"));
        }
        let key = (reference_frames, target_frames);
        self.shapes.insert(key, engine);
        self.touch(key);
        Ok(())
    }

    pub fn choose_speed_engine(
        &mut self,
        reference_frames: usize,
        target_frames: usize,
    ) -> Result<()> {
        let key = (reference_frames, target_frames);
        if !self.shapes.contains_key(&key) {
            return Err(F5Error::InvalidShape(
                "Cross-Lingual speed engine shape is not warmed",
            ));
        }
        self.speed_shape = Some(key);
        Ok(())
    }

    pub fn remove(&mut self, reference_frames: usize, target_frames: usize) -> bool {
        let key = (reference_frames, target_frames);
        if self.speed_shape == Some(key) {
            self.speed_shape = None;
        }
        self.usage.retain(|item| *item != key);
        self.shapes.remove(&key).is_some()
    }

    pub fn clear(&mut self) {
        self.speed_shape = None;
        self.shapes.clear();
        self.usage.clear();
    }

    pub fn warmed_shapes(&self) -> usize {
        self.shapes.len()
    }
}

impl Default for TrtWarmBank {
    fn default() -> Self {
        Self::new()
    }
}

impl TrtChunkKernel for TrtWarmBank {
    fn edit_speech(
        &mut self,
        plan: &SpeechEditPlan,
        tokens: &[i32],
        noise: &[f32],
        restoration_gain: f32,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        let key = (0, plan.total_frames);
        self.ensure(key)?;
        self.shapes
            .get_mut(&key)
            .ok_or(F5Error::InvalidConfiguration(
                "Speech edit TRT engine is missing",
            ))?
            .edit_speech(plan, tokens, noise, restoration_gain, sampling)
    }

    fn edit_speech_with_method(
        &mut self,
        plan: &SpeechEditPlan,
        tokens: &[i32],
        noise: &[f32],
        restoration_gain: f32,
        sampling: SamplingConfig,
        method: OdeMethod,
    ) -> Result<Vec<f32>> {
        let key = (0, plan.total_frames);
        self.ensure(key)?;
        self.shapes
            .get_mut(&key)
            .ok_or(F5Error::InvalidConfiguration(
                "Missing warmed speech-edit engine",
            ))?
            .edit_speech_with_method(plan, tokens, noise, restoration_gain, sampling, method)
    }

    fn synthesize_chunk(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        let reference_frames = crate::mel::VocosMel::new()
            .extract(&reference.samples)?
            .frames;
        self.ensure((reference_frames, target_frames))?;
        let kernel = self
            .shapes
            .get_mut(&(reference_frames, target_frames))
            .ok_or(F5Error::InvalidShape(
                "No warmed TensorRT engine matches the requested reference and target frames",
            ))?;
        kernel.synthesize_chunk(reference, tokens, target_frames, noise, sampling)
    }

    fn synthesize_chunk_with_sample_mode(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
        method: OdeMethod,
        mode: CfmSampleMode,
    ) -> Result<Vec<f32>> {
        let reference_frames = crate::mel::VocosMel::new()
            .extract(&reference.samples)?
            .frames;
        self.ensure((reference_frames, target_frames))?;
        self.shapes
            .get_mut(&(reference_frames, target_frames))
            .ok_or(F5Error::InvalidConfiguration("Warmed F5 shape not found"))?
            .synthesize_chunk_with_sample_mode(
                reference,
                tokens,
                target_frames,
                noise,
                sampling,
                method,
                mode,
            )
    }

    fn synthesize_chunk_with_spectrogram(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
    ) -> Result<InferenceOutput> {
        let reference_frames = crate::mel::VocosMel::new()
            .extract(&reference.samples)?
            .frames;
        self.ensure((reference_frames, target_frames))?;
        self.shapes
            .get_mut(&(reference_frames, target_frames))
            .ok_or(F5Error::InvalidShape("Warmed TRT shape missing"))?
            .synthesize_chunk_with_spectrogram(reference, tokens, target_frames, noise, sampling)
    }

    fn synthesize_chunk_with_spectrogram_and_sample_mode(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
        method: OdeMethod,
        mode: CfmSampleMode,
    ) -> Result<InferenceOutput> {
        let reference_frames = crate::mel::VocosMel::new()
            .extract(&reference.samples)?
            .frames;
        self.ensure((reference_frames, target_frames))?;
        self.shapes
            .get_mut(&(reference_frames, target_frames))
            .ok_or(F5Error::InvalidShape("Warmed TRT shape missing"))?
            .synthesize_chunk_with_spectrogram_and_sample_mode(
                reference,
                tokens,
                target_frames,
                noise,
                sampling,
                method,
                mode,
            )
    }

    fn predict_reference_speed(&mut self, reference: &ReferenceAudio) -> Result<f32> {
        let reference_frames = crate::mel::VocosMel::new()
            .extract(&reference.samples)?
            .frames;
        let key = if let Some(key) = self.speed_shape.filter(|key| key.0 == reference_frames) {
            key
        } else if let Some(key) = self
            .shapes
            .keys()
            .find(|key| key.0 == reference_frames)
            .copied()
        {
            key
        } else {
            let target_frames = reference_frames
                .checked_add(1)
                .ok_or(F5Error::InvalidShape("F5 reference length overflow"))?;
            (reference_frames, target_frames)
        };
        self.ensure(key)?;
        self.speed_shape = Some(key);
        self.shapes
            .get_mut(&key)
            .ok_or(F5Error::InvalidConfiguration(
                "Warmed speed engine was unloaded",
            ))?
            .predict_reference_speed(reference)
    }
}

pub trait TrtChunkKernel {
    fn edit_speech(
        &mut self,
        _plan: &SpeechEditPlan,
        _tokens: &[i32],
        _noise: &[f32],
        _restoration_gain: f32,
        _sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        Err(F5Error::InvalidConfiguration(
            "TRT kernel does not support speech editing",
        ))
    }

    fn edit_speech_with_method(
        &mut self,
        plan: &SpeechEditPlan,
        tokens: &[i32],
        noise: &[f32],
        restoration_gain: f32,
        sampling: SamplingConfig,
        method: OdeMethod,
    ) -> Result<Vec<f32>> {
        if method == OdeMethod::Euler {
            self.edit_speech(plan, tokens, noise, restoration_gain, sampling)
        } else {
            Err(F5Error::InvalidConfiguration(
                "TRT kernel does not support midpoint editing",
            ))
        }
    }

    fn synthesize_chunk(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>>;

    fn synthesize_chunk_with_sample_mode(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
        method: OdeMethod,
        mode: CfmSampleMode,
    ) -> Result<Vec<f32>> {
        if method == OdeMethod::Euler && mode == CfmSampleMode::default() {
            self.synthesize_chunk(reference, tokens, target_frames, noise, sampling)
        } else {
            Err(F5Error::InvalidConfiguration(
                "TRT kernel does not support CFM sample modes",
            ))
        }
    }

    fn synthesize_chunk_with_spectrogram(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
    ) -> Result<InferenceOutput> {
        let _ = (reference, tokens, target_frames, noise, sampling);
        Err(F5Error::InvalidConfiguration(
            "TRT kernel does not expose generated mel spectrogram",
        ))
    }

    fn synthesize_chunk_with_spectrogram_and_sample_mode(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
        method: OdeMethod,
        mode: CfmSampleMode,
    ) -> Result<InferenceOutput> {
        if method == OdeMethod::Euler && mode == CfmSampleMode::default() {
            self.synthesize_chunk_with_spectrogram(
                reference,
                tokens,
                target_frames,
                noise,
                sampling,
            )
        } else {
            Err(F5Error::InvalidConfiguration(
                "TRT kernel does not support mel capture with CFM sample modes",
            ))
        }
    }

    fn predict_reference_speed(&mut self, reference: &ReferenceAudio) -> Result<f32>;
}

impl<I: CudaTransport, E: TrtExecutor> TrtChunkKernel for TrtF5Session<I, E> {
    fn edit_speech(
        &mut self,
        plan: &SpeechEditPlan,
        tokens: &[i32],
        noise: &[f32],
        restoration_gain: f32,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        TrtF5Session::edit_speech(self, plan, tokens, noise, restoration_gain, sampling)
    }

    fn edit_speech_with_method(
        &mut self,
        plan: &SpeechEditPlan,
        tokens: &[i32],
        noise: &[f32],
        restoration_gain: f32,
        sampling: SamplingConfig,
        method: OdeMethod,
    ) -> Result<Vec<f32>> {
        TrtF5Session::edit_speech_with_method(
            self,
            plan,
            tokens,
            noise,
            restoration_gain,
            sampling,
            method,
        )
    }

    fn synthesize_chunk(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        if target_frames != self.shape().cfm.backbone.frames {
            return Err(F5Error::InvalidShape(
                "No warmed TensorRT engine for the requested F5 duration",
            ));
        }
        let condition =
            reference.prepare(&crate::mel::VocosMel::new(), target_frames, tokens.len())?;
        if condition.frames.reference != self.shape().reference_frames
            || condition.frames.target != target_frames
        {
            return Err(F5Error::InvalidShape(
                "No warmed TensorRT engine for this reference duration",
            ));
        }
        self.synthesize(SynthesisInputs {
            condition: &condition.conditioning_mel,
            initial_noise: noise,
            tokens,
            restoration_gain: condition.restoration_gain(),
            sampling,
        })
    }

    fn synthesize_chunk_with_sample_mode(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
        method: OdeMethod,
        mode: CfmSampleMode,
    ) -> Result<Vec<f32>> {
        if target_frames != self.shape().cfm.backbone.frames {
            return Err(F5Error::InvalidShape("Wrong warmed duration"));
        }
        let condition =
            reference.prepare(&crate::mel::VocosMel::new(), target_frames, tokens.len())?;
        if condition.frames.reference != self.shape().reference_frames {
            return Err(F5Error::InvalidShape("Wrong warmed reference length"));
        }
        self.synthesize_with_sample_mode(
            SynthesisInputs {
                condition: &condition.conditioning_mel,
                initial_noise: noise,
                tokens,
                restoration_gain: condition.restoration_gain(),
                sampling,
            },
            method,
            mode,
        )
    }

    fn synthesize_chunk_with_spectrogram(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
    ) -> Result<InferenceOutput> {
        if target_frames != self.shape().cfm.backbone.frames {
            return Err(F5Error::InvalidShape("Wrong warmed F5 duration"));
        }
        let condition =
            reference.prepare(&crate::mel::VocosMel::new(), target_frames, tokens.len())?;
        if condition.frames.reference != self.shape().reference_frames {
            return Err(F5Error::InvalidShape("Wrong warmed F5 reference length"));
        }
        self.synthesize_with_spectrogram(
            SynthesisInputs {
                condition: &condition.conditioning_mel,
                initial_noise: noise,
                tokens,
                restoration_gain: condition.restoration_gain(),
                sampling,
            },
            OdeMethod::Euler,
            CfmSampleMode::default(),
        )
    }

    fn synthesize_chunk_with_spectrogram_and_sample_mode(
        &mut self,
        reference: &ReferenceAudio,
        tokens: &[i32],
        target_frames: usize,
        noise: &[f32],
        sampling: SamplingConfig,
        method: OdeMethod,
        mode: CfmSampleMode,
    ) -> Result<InferenceOutput> {
        if target_frames != self.shape().cfm.backbone.frames {
            return Err(F5Error::InvalidShape("Wrong warmed F5 duration"));
        }
        let condition =
            reference.prepare(&crate::mel::VocosMel::new(), target_frames, tokens.len())?;
        if condition.frames.reference != self.shape().reference_frames {
            return Err(F5Error::InvalidShape("Wrong warmed F5 reference length"));
        }
        self.synthesize_with_spectrogram(
            SynthesisInputs {
                condition: &condition.conditioning_mel,
                initial_noise: noise,
                tokens,
                restoration_gain: condition.restoration_gain(),
                sampling,
            },
            method,
            mode,
        )
    }

    fn predict_reference_speed(&mut self, reference: &ReferenceAudio) -> Result<f32> {
        TrtF5Session::predict_reference_speed(self, reference)
    }
}

#[derive(Clone, Copy)]
pub struct TrtVoice<'a> {
    pub audio: &'a ReferenceAudio,
    pub transcript: &'a str,
}

#[derive(Clone, Copy)]
pub struct StandardTextOptions {
    pub speed: f32,
    pub fixed_seconds: Option<f32>,
    pub cross_fade_seconds: f32,
    pub seed: u64,
    pub sampling: SamplingConfig,
}

#[derive(Clone, Copy)]
pub struct CrossLingualTextOptions {
    pub user_speed: f64,
    pub fixed_seconds: Option<f64>,
    pub cross_fade_seconds: f64,
    pub seed: u64,
    pub sampling: SamplingConfig,
}

fn generate_chunk<K: TrtChunkKernel>(
    kernel: &mut K,
    reference: &ReferenceAudio,
    tokens: &[i32],
    frames: usize,
    rng: &mut StandardNormal,
    sampling: SamplingConfig,
) -> Result<Vec<f32>> {
    let count = frames
        .checked_mul(100)
        .ok_or(F5Error::InvalidShape("F5 noise length overflow"))?;
    let noise = rng.sample_vec(count)?;
    kernel.synthesize_chunk(reference, tokens, frames, &noise, sampling)
}

pub fn synthesize_standard_text<K: TrtChunkKernel, P: PinyinProvider>(
    kernel: &mut K,
    reference: TrtVoice<'_>,
    generated_text: &str,
    vocabulary: &Vocabulary,
    pinyin: P,
    options: StandardTextOptions,
) -> Result<Vec<f32>> {
    synthesize_standard_text_with_sample_mode(
        kernel,
        reference,
        generated_text,
        vocabulary,
        pinyin,
        options,
        OdeMethod::Euler,
        CfmSampleMode::default(),
    )
}

pub fn synthesize_standard_text_with_sample_mode<K: TrtChunkKernel, P: PinyinProvider>(
    kernel: &mut K,
    reference: TrtVoice<'_>,
    generated_text: &str,
    vocabulary: &Vocabulary,
    pinyin: P,
    options: StandardTextOptions,
    method: OdeMethod,
    mode: CfmSampleMode,
) -> Result<Vec<f32>> {
    let plans = plan_standard_chunks(
        generated_text,
        reference.transcript,
        reference.audio.samples.len(),
        options.speed,
        options.fixed_seconds,
        options.cross_fade_seconds,
    )?;
    let frontend = F5TextFrontend::new(vocabulary, pinyin);
    let mut rng = StandardNormal::new(options.seed);
    let mut waves = Vec::with_capacity(plans.len());
    for chunk in plans {
        let tokens = frontend.standard_tokens(reference.transcript, &chunk.text)?;
        let count = chunk
            .total_frames
            .checked_mul(100)
            .ok_or(F5Error::InvalidShape("Noise size overflow"))?;
        let noise = rng.sample_vec(count)?;
        waves.push(kernel.synthesize_chunk_with_sample_mode(
            reference.audio,
            &tokens,
            chunk.total_frames,
            &noise,
            options.sampling,
            method,
            mode,
        )?);
    }
    cross_fade_waves(&waves, options.cross_fade_seconds as f64)
}

pub fn synthesize_standard_text_with_spectrogram<K: TrtChunkKernel, P: PinyinProvider>(
    kernel: &mut K,
    reference: TrtVoice<'_>,
    generated_text: &str,
    vocabulary: &Vocabulary,
    pinyin: P,
    options: StandardTextOptions,
) -> Result<InferenceOutput> {
    synthesize_standard_text_with_spectrogram_and_sample_mode(
        kernel,
        reference,
        generated_text,
        vocabulary,
        pinyin,
        options,
        OdeMethod::Euler,
        CfmSampleMode::default(),
    )
}

pub fn synthesize_standard_text_with_spectrogram_and_sample_mode<
    K: TrtChunkKernel,
    P: PinyinProvider,
>(
    kernel: &mut K,
    reference: TrtVoice<'_>,
    generated_text: &str,
    vocabulary: &Vocabulary,
    pinyin: P,
    options: StandardTextOptions,
    method: OdeMethod,
    mode: CfmSampleMode,
) -> Result<InferenceOutput> {
    mode.validate()?;
    let plans = plan_standard_chunks(
        generated_text,
        reference.transcript,
        reference.audio.samples.len(),
        options.speed,
        options.fixed_seconds,
        options.cross_fade_seconds,
    )?;
    let frontend = F5TextFrontend::new(vocabulary, pinyin);
    let mut rng = StandardNormal::new(options.seed);
    let mut waves = Vec::with_capacity(plans.len());
    let mut spectra = Vec::with_capacity(plans.len());
    for chunk in plans {
        let tokens = frontend.standard_tokens(reference.transcript, &chunk.text)?;
        let noise = rng.sample_vec(
            chunk
                .total_frames
                .checked_mul(100)
                .ok_or(F5Error::InvalidShape("Noise size overflow"))?,
        )?;
        let output = kernel.synthesize_chunk_with_spectrogram_and_sample_mode(
            reference.audio,
            &tokens,
            chunk.total_frames,
            &noise,
            options.sampling,
            method,
            mode,
        )?;
        waves.push(output.waveform);
        spectra.push(output.spectrogram);
    }
    Ok(InferenceOutput {
        waveform: cross_fade_waves(&waves, options.cross_fade_seconds as f64)?,
        sample_rate: crate::mel::SAMPLE_RATE,
        spectrogram: GeneratedSpectrogram::concatenate(&spectra)?,
    })
}

pub fn stream_standard_text<K: TrtChunkKernel, P: PinyinProvider>(
    kernel: &mut K,
    reference: TrtVoice<'_>,
    generated_text: &str,
    vocabulary: &Vocabulary,
    pinyin: P,
    options: StandardTextOptions,
    chunk_size: usize,
    emit: impl FnMut(&[f32], usize) -> Result<()>,
) -> Result<()> {
    stream_standard_text_with_sample_mode(
        kernel,
        reference,
        generated_text,
        vocabulary,
        pinyin,
        options,
        OdeMethod::Euler,
        CfmSampleMode::default(),
        chunk_size,
        emit,
    )
}

pub fn stream_standard_text_with_sample_mode<K: TrtChunkKernel, P: PinyinProvider>(
    kernel: &mut K,
    reference: TrtVoice<'_>,
    generated_text: &str,
    vocabulary: &Vocabulary,
    pinyin: P,
    options: StandardTextOptions,
    method: OdeMethod,
    mode: CfmSampleMode,
    chunk_size: usize,
    mut emit: impl FnMut(&[f32], usize) -> Result<()>,
) -> Result<()> {
    if chunk_size == 0 {
        return Err(F5Error::InvalidConfiguration(
            "Streaming chunk size must be nonzero",
        ));
    }
    mode.validate()?;
    let plans = plan_standard_chunks(
        generated_text,
        reference.transcript,
        reference.audio.samples.len(),
        options.speed,
        options.fixed_seconds,
        options.cross_fade_seconds,
    )?;
    let frontend = F5TextFrontend::new(vocabulary, pinyin);
    let mut rng = StandardNormal::new(options.seed);
    for chunk in plans {
        let tokens = frontend.standard_tokens(reference.transcript, &chunk.text)?;
        let count = chunk
            .total_frames
            .checked_mul(100)
            .ok_or(F5Error::InvalidShape("Noise size overflow"))?;
        let noise = rng.sample_vec(count)?;
        let wave = kernel.synthesize_chunk_with_sample_mode(
            reference.audio,
            &tokens,
            chunk.total_frames,
            &noise,
            options.sampling,
            method,
            mode,
        )?;
        emit_pcm_chunks(&wave, chunk_size, &mut emit)?;
    }
    Ok(())
}

pub fn synthesize_multi_voice<K: TrtChunkKernel, P: PinyinProvider>(
    kernel: &mut K,
    script: &str,
    voices: &BTreeMap<String, TrtVoice<'_>>,
    vocabulary: &Vocabulary,
    pinyin: P,
    speed: f32,
    seed: u64,
    sampling: SamplingConfig,
) -> Result<Vec<f32>> {
    let main = *voices.get("main").ok_or(F5Error::InvalidConfiguration(
        "Multi-voice F5 needs a main voice",
    ))?;
    let segments = parse_voice_segments(script);
    if segments.is_empty() {
        return Err(F5Error::InvalidShape("Empty multi-voice F5 script"));
    }
    let frontend = F5TextFrontend::new(vocabulary, pinyin);
    let mut rng = StandardNormal::new(seed);
    let mut output = Vec::new();
    for segment in segments {
        let voice = *voices.get(&segment.voice).unwrap_or(&main);
        let plans = plan_standard_chunks(
            &segment.text,
            voice.transcript,
            voice.audio.samples.len(),
            speed,
            None,
            0.15,
        )?;
        let mut waves = Vec::with_capacity(plans.len());
        for chunk in plans {
            let tokens = frontend.standard_tokens(voice.transcript, &chunk.text)?;
            waves.push(generate_chunk(
                kernel,
                voice.audio,
                &tokens,
                chunk.total_frames,
                &mut rng,
                sampling,
            )?);
        }
        output.extend(cross_fade_waves(&waves, 0.15)?);
    }
    Ok(output)
}

pub fn synthesize_cross_lingual<K: TrtChunkKernel, P: PinyinProvider>(
    kernel: &mut K,
    reference: &ReferenceAudio,
    generated_text: &str,
    vocabulary: &Vocabulary,
    pinyin: P,
    english_syllables: &impl EnglishSyllableCounter,
    options: CrossLingualTextOptions,
) -> Result<Vec<f32>> {
    let predicted_speed = kernel.predict_reference_speed(reference)?;
    let planned = plan_cross_lingual_chunks(
        generated_text,
        reference.samples.len(),
        predicted_speed,
        options.user_speed,
        options.fixed_seconds,
        english_syllables,
    )?;
    if planned.is_empty() {
        return Err(F5Error::InvalidShape(
            "Cross-Lingual text produced no chunks",
        ));
    }
    let frontend = F5TextFrontend::new(vocabulary, pinyin);
    let mut rng = StandardNormal::new(options.seed);
    let mut waves = Vec::with_capacity(planned.len());
    for chunk in planned {
        let tokens = frontend.cross_lingual_tokens(&chunk.text)?;
        waves.push(generate_chunk(
            kernel,
            reference,
            &tokens,
            chunk.total_frames,
            &mut rng,
            options.sampling,
        )?);
    }
    cross_fade_waves(&waves, options.cross_fade_seconds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{hyphenation::EnglishHyphenator, text_frontend::NoPinyin};

    #[derive(Default)]
    struct RecordingKernel {
        calls: Vec<(usize, usize, Vec<f32>, usize)>,
        speed_calls: usize,
    }

    impl TrtChunkKernel for RecordingKernel {
        fn synthesize_chunk(
            &mut self,
            reference: &ReferenceAudio,
            tokens: &[i32],
            target_frames: usize,
            noise: &[f32],
            _: SamplingConfig,
        ) -> Result<Vec<f32>> {
            assert_eq!(noise.len(), target_frames * 100);
            self.calls.push((
                target_frames,
                tokens.len(),
                noise[..4].to_vec(),
                reference.samples.len(),
            ));
            Ok(vec![target_frames as f32 / 10_000.0; 2048])
        }

        fn synthesize_chunk_with_spectrogram(
            &mut self,
            reference: &ReferenceAudio,
            tokens: &[i32],
            target_frames: usize,
            noise: &[f32],
            sampling: SamplingConfig,
        ) -> Result<InferenceOutput> {
            let waveform =
                self.synthesize_chunk(reference, tokens, target_frames, noise, sampling)?;
            let frames = target_frames
                - crate::mel::VocosMel::new()
                    .extract(&reference.samples)?
                    .frames;
            let mel = vec![target_frames as f32; frames * 100];
            Ok(InferenceOutput {
                waveform,
                sample_rate: 24_000,
                spectrogram: GeneratedSpectrogram::from_frame_major(&mel, 100)?,
            })
        }

        fn predict_reference_speed(&mut self, _: &ReferenceAudio) -> Result<f32> {
            self.speed_calls += 1;
            Ok(2.0)
        }
    }

    fn vocab() -> Vocabulary {
        Vocabulary::from_lines(
            " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?.,:;"
                .chars()
                .map(|c| c.to_string()),
        )
        .unwrap()
    }

    fn reference() -> ReferenceAudio {
        let samples = vec![0.04_f32; 24_000];
        ReferenceAudio::from_channels(&[&samples], 24_000).unwrap()
    }

    #[test]
    fn warm_bank_routes_reference_and_target_shapes_and_unloads() {
        let ref_audio = reference();
        let ref_frames = crate::mel::VocosMel::new()
            .extract(&ref_audio.samples)
            .unwrap()
            .frames;
        let mut bank = TrtWarmBank::new();
        assert!(
            bank.insert(
                ref_frames,
                ref_frames + 10,
                Box::new(RecordingKernel::default())
            )
            .is_ok()
        );
        assert!(
            bank.insert(
                ref_frames,
                ref_frames + 10,
                Box::new(RecordingKernel::default())
            )
            .is_err()
        );
        assert!(
            bank.synthesize_chunk(
                &ref_audio,
                &[1, 2],
                ref_frames + 10,
                &vec![0.0; (ref_frames + 10) * 100],
                SamplingConfig::default()
            )
            .is_ok()
        );
        assert!(
            bank.synthesize_chunk(
                &ref_audio,
                &[1, 2],
                ref_frames + 11,
                &vec![0.0; (ref_frames + 11) * 100],
                SamplingConfig::default()
            )
            .is_err()
        );
        bank.choose_speed_engine(ref_frames, ref_frames + 10)
            .unwrap();
        assert_eq!(bank.predict_reference_speed(&ref_audio).unwrap(), 2.0);
        assert!(bank.remove(ref_frames, ref_frames + 10));
        assert!(bank.predict_reference_speed(&ref_audio).is_err());
        assert_eq!(bank.warmed_shapes(), 0);
    }

    #[test]
    fn standard_chunks_preserve_python_duration_and_rng_continuity() {
        let reference = reference();
        let mut backend = RecordingKernel::default();
        let text = "Hello world. Another sentence. More words here!";
        let options = StandardTextOptions {
            speed: 1.0,
            fixed_seconds: None,
            cross_fade_seconds: 0.15,
            seed: 42,
            sampling: SamplingConfig::default(),
        };
        let _ = synthesize_standard_text(
            &mut backend,
            TrtVoice {
                audio: &reference,
                transcript: "Hi",
            },
            text,
            &vocab(),
            NoPinyin,
            options,
        )
        .unwrap();
        let planned =
            plan_standard_chunks(text, "Hi", reference.samples.len(), 1.0, None, 0.15).unwrap();
        assert_eq!(backend.calls.len(), planned.len());
        let mut rng = StandardNormal::new(42);
        for (index, call) in backend.calls.iter().enumerate() {
            assert_eq!(call.0, planned[index].total_frames);
            assert_eq!(call.2, rng.sample_vec(call.0 * 100).unwrap()[..4]);
        }
    }

    #[test]
    fn standard_streaming_emits_original_pcm_chunks_without_crossfade() {
        let reference = reference();
        let text = "Hello world. Another sentence. More words here!";
        let options = StandardTextOptions {
            speed: 1.0,
            fixed_seconds: Some(9.0),
            cross_fade_seconds: 0.15,
            seed: 42,
            sampling: SamplingConfig::default(),
        };
        let mut backend = RecordingKernel::default();
        let mut audio = Vec::new();
        let mut sizes = Vec::new();
        stream_standard_text(
            &mut backend,
            TrtVoice {
                audio: &reference,
                transcript: "Hi",
            },
            text,
            &vocab(),
            NoPinyin,
            options,
            700,
            |chunk, rate| {
                assert_eq!(rate, 24_000);
                sizes.push(chunk.len());
                audio.extend_from_slice(chunk);
                Ok(())
            },
        )
        .unwrap();
        let planned =
            plan_standard_chunks(text, "Hi", reference.samples.len(), 1.0, Some(9.0), 0.15)
                .unwrap();
        assert_eq!(backend.calls.len(), planned.len());
        assert_eq!(audio.len(), 2048 * planned.len());
        assert_eq!(sizes, [700, 700, 648].repeat(planned.len()));
        for (i, call) in backend.calls.iter().enumerate() {
            assert_eq!(call.0, planned[i].total_frames);
            assert!(
                audio[i * 2048..(i + 1) * 2048]
                    .iter()
                    .all(|v| *v == call.0 as f32 / 10_000.0)
            );
        }
        assert!(
            stream_standard_text(
                &mut RecordingKernel::default(),
                TrtVoice {
                    audio: &reference,
                    transcript: "Hi"
                },
                text,
                &vocab(),
                NoPinyin,
                options,
                0,
                |_, _| Ok(()),
            )
            .is_err()
        );
        assert_eq!(backend.calls.len(), planned.len());
    }

    #[test]
    fn streaming_sink_failure_prevents_generation_of_later_batches() {
        let reference = reference();
        let mut backend = RecordingKernel::default();
        let result = stream_standard_text(
            &mut backend,
            TrtVoice {
                audio: &reference,
                transcript: "Hi",
            },
            "Hello world. Another sentence. More words here!",
            &vocab(),
            NoPinyin,
            StandardTextOptions {
                speed: 1.0,
                fixed_seconds: None,
                cross_fade_seconds: 0.15,
                seed: 13,
                sampling: SamplingConfig::default(),
            },
            128,
            |_, _| Err(F5Error::ModelExecution("consumer closed".into())),
        );
        assert!(result.is_err());
        assert_eq!(backend.calls.len(), 1);
    }

    #[test]
    fn spectrogram_batches_are_concatenated_on_frame_axis() {
        let reference = reference();
        let mut backend = RecordingKernel::default();
        let text = "Hello world. Another sentence. More words here!";
        let options = StandardTextOptions {
            speed: 1.0,
            fixed_seconds: Some(9.0),
            cross_fade_seconds: 0.15,
            seed: 21,
            sampling: SamplingConfig::default(),
        };
        let output = synthesize_standard_text_with_spectrogram(
            &mut backend,
            TrtVoice {
                audio: &reference,
                transcript: "Hi",
            },
            text,
            &vocab(),
            NoPinyin,
            options,
        )
        .unwrap();
        let ref_frames = crate::mel::VocosMel::new()
            .extract(&reference.samples)
            .unwrap()
            .frames;
        let generated_frames = backend
            .calls
            .iter()
            .map(|x| x.0 - ref_frames)
            .sum::<usize>();
        assert_eq!(output.spectrogram.channels, 100);
        assert_eq!(output.spectrogram.frames, generated_frames);
        assert_eq!(output.spectrogram.values.len(), generated_frames * 100);
        assert_eq!(output.sample_rate, 24_000);
        assert_eq!(output.waveform.len(), 2048);
        let first = backend.calls[0].0 as f32;
        assert_eq!(output.spectrogram.values[0], first);
    }

    #[test]
    fn cross_lingual_uses_predicted_reference_speed_once() {
        let reference = reference();
        let en = EnglishHyphenator::us_english().unwrap();
        let mut backend = RecordingKernel::default();
        let options = CrossLingualTextOptions {
            user_speed: 1.0,
            fixed_seconds: None,
            cross_fade_seconds: 0.15,
            seed: 55,
            sampling: SamplingConfig::default(),
        };
        let wave = synthesize_cross_lingual(
            &mut backend,
            &reference,
            "Hello!",
            &vocab(),
            NoPinyin,
            &en,
            options,
        )
        .unwrap();
        assert!(!wave.is_empty());
        assert_eq!(backend.speed_calls, 1);
        assert_eq!(backend.calls.len(), 1);
        let plans =
            plan_cross_lingual_chunks("Hello!", reference.samples.len(), 2.0, 1.0, None, &en)
                .unwrap();
        assert_eq!(backend.calls[0].0, plans[0].total_frames);
    }

    #[test]
    fn voice_script_falls_back_to_main_without_reseeding() {
        let reference = reference();
        let mut backend = RecordingKernel::default();
        let mut voices = BTreeMap::new();
        voices.insert(
            "main".to_string(),
            TrtVoice {
                audio: &reference,
                transcript: "Hi",
            },
        );
        let output = synthesize_multi_voice(
            &mut backend,
            "[main] Hello! [unknown] World!",
            &voices,
            &vocab(),
            NoPinyin,
            1.0,
            51,
            SamplingConfig::default(),
        )
        .unwrap();
        assert!(!output.is_empty());
        assert!(backend.calls.len() >= 2);
        let mut rng = StandardNormal::new(51);
        for call in &backend.calls {
            assert_eq!(call.2, rng.sample_vec(call.0 * 100).unwrap()[..4]);
        }
    }
}
