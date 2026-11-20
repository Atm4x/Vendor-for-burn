//! Single-model F5-TTS worker with lazy shape selection and hot-swappable reference voices.

use std::collections::BTreeMap;

use crate::{
    F5Error, Result,
    cfm::SamplingConfig,
    cross_lingual::EnglishSyllableCounter,
    mel::VocosMel,
    noise::StandardNormal,
    reference::ReferenceAudio,
    speech_edit::{EditSpan, SpeechEditPlan},
    text_frontend::{PinyinProvider, normalized_reference_transcript},
    vocab::Vocabulary,
};

use super::trtx_text::{
    CrossLingualTextOptions, StandardTextOptions, TrtChunkKernel, TrtKernelFactory, TrtVoice,
    TrtWarmBank, synthesize_cross_lingual, synthesize_multi_voice, synthesize_standard_text,
};

#[derive(Debug, Clone)]
pub struct ReferenceVoice {
    pub audio: ReferenceAudio,
    pub transcript: String,
}

impl ReferenceVoice {
    pub fn new(audio: ReferenceAudio, transcript: impl Into<String>) -> Result<Self> {
        let transcript = transcript.into();
        normalized_reference_transcript(&transcript)?;
        Ok(Self { audio, transcript })
    }

    fn borrowed(&self) -> TrtVoice<'_> {
        TrtVoice {
            audio: &self.audio,
            transcript: &self.transcript,
        }
    }
}

pub struct ModelActivation<P> {
    pub model_id: String,
    pub vocabulary: Vocabulary,
    pub pinyin: P,
    pub main_voice: ReferenceVoice,
    pub factory: Box<dyn TrtKernelFactory>,
    pub cache_capacity: usize,
}

struct ActiveModel<P> {
    id: String,
    vocabulary: Vocabulary,
    pinyin: P,
    engines: TrtWarmBank,
    voices: BTreeMap<String, ReferenceVoice>,
    selected_voice: String,
}

pub struct TrtModelWorker<P: PinyinProvider + Clone> {
    active: Option<ActiveModel<P>>,
}

impl<P: PinyinProvider + Clone> Default for TrtModelWorker<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: PinyinProvider + Clone> TrtModelWorker<P> {
    pub fn new() -> Self {
        Self { active: None }
    }

    pub fn model_id(&self) -> Option<&str> {
        self.active.as_ref().map(|model| model.id.as_str())
    }

    pub fn active_voice(&self) -> Option<&str> {
        self.active
            .as_ref()
            .map(|model| model.selected_voice.as_str())
    }

    pub fn cached_shapes(&self) -> usize {
        self.active
            .as_ref()
            .map_or(0, |model| model.engines.warmed_shapes())
    }

    pub fn load(&mut self, mut config: ModelActivation<P>) -> Result<()> {
        if config.model_id.trim().is_empty() {
            return Err(F5Error::InvalidConfiguration("F5 model ID is empty"));
        }
        if config.cache_capacity == 0 {
            return Err(F5Error::InvalidConfiguration(
                "F5 engine cache capacity must be nonzero",
            ));
        }
        config.factory.validate_model()?;
        let engines = TrtWarmBank::with_factory(config.factory, config.cache_capacity)?;
        let mut voices = BTreeMap::new();
        voices.insert("main".to_owned(), config.main_voice);
        let replacement = ActiveModel {
            id: config.model_id,
            vocabulary: config.vocabulary,
            pinyin: config.pinyin,
            engines,
            voices,
            selected_voice: "main".to_owned(),
        };
        self.active = Some(replacement);
        Ok(())
    }

    pub fn unload(&mut self) -> bool {
        self.active.take().is_some()
    }

    pub fn set_voice(&mut self, voice_id: impl Into<String>, voice: ReferenceVoice) -> Result<()> {
        let id = voice_id.into();
        if id.is_empty() || id.contains(['[', ']']) || id.trim() != id {
            return Err(F5Error::InvalidConfiguration("Invalid F5 voice ID"));
        }
        let active = self
            .active
            .as_mut()
            .ok_or(F5Error::InvalidConfiguration("No F5 model loaded"))?;
        active.voices.insert(id, voice);
        Ok(())
    }

    pub fn choose_voice(&mut self, voice_id: &str) -> Result<()> {
        let active = self
            .active
            .as_mut()
            .ok_or(F5Error::InvalidConfiguration("No F5 model loaded"))?;
        if !active.voices.contains_key(voice_id) {
            return Err(F5Error::InvalidConfiguration("Unknown F5 voice"));
        }
        active.selected_voice = voice_id.to_owned();
        Ok(())
    }

    pub fn remove_voice(&mut self, voice_id: &str) -> Result<bool> {
        if voice_id == "main" {
            return Err(F5Error::InvalidConfiguration(
                "Cannot remove the main F5 voice",
            ));
        }
        let active = self
            .active
            .as_mut()
            .ok_or(F5Error::InvalidConfiguration("No F5 model loaded"))?;
        let removed = active.voices.remove(voice_id).is_some();
        if removed && active.selected_voice == voice_id {
            active.selected_voice = "main".to_owned();
        }
        Ok(removed)
    }

    pub fn edit_speech(
        &mut self,
        recording: &ReferenceAudio,
        target_text: &str,
        spans: &[EditSpan],
        seed: u64,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        let active = self
            .active
            .as_mut()
            .ok_or(F5Error::InvalidConfiguration("No F5 model loaded"))?;
        let tokens =
            crate::text_frontend::F5TextFrontend::new(&active.vocabulary, active.pinyin.clone())
                .cross_lingual_tokens(target_text)?;
        let original = VocosMel::new().extract(&recording.samples)?;
        let plan = SpeechEditPlan::new(&original.values, original.frames, spans, tokens.len())?;
        let mut rng = StandardNormal::new(seed);
        let noise = rng.sample_vec(plan.total_frames * 100)?;
        let gain = if recording.normalization_gain > 1.0 {
            1.0 / recording.normalization_gain
        } else {
            1.0
        };
        active
            .engines
            .edit_speech(&plan, &tokens, &noise, gain, sampling)
    }

    pub fn synthesize(&mut self, text: &str, options: StandardTextOptions) -> Result<Vec<f32>> {
        let active = self
            .active
            .as_mut()
            .ok_or(F5Error::InvalidConfiguration("No F5 model loaded"))?;
        let voice = active
            .voices
            .get(&active.selected_voice)
            .ok_or(F5Error::InvalidConfiguration("Active F5 voice is missing"))?;
        synthesize_standard_text(
            &mut active.engines,
            voice.borrowed(),
            text,
            &active.vocabulary,
            active.pinyin.clone(),
            options,
        )
    }

    pub fn synthesize_voices(
        &mut self,
        script: &str,
        speed: f32,
        seed: u64,
        sampling: SamplingConfig,
    ) -> Result<Vec<f32>> {
        let active = self
            .active
            .as_mut()
            .ok_or(F5Error::InvalidConfiguration("No F5 model loaded"))?;
        let voices = active
            .voices
            .iter()
            .map(|(id, v)| (id.clone(), v.borrowed()))
            .collect();
        synthesize_multi_voice(
            &mut active.engines,
            script,
            &voices,
            &active.vocabulary,
            active.pinyin.clone(),
            speed,
            seed,
            sampling,
        )
    }

    pub fn synthesize_cross_lingual(
        &mut self,
        text: &str,
        english: &impl EnglishSyllableCounter,
        options: CrossLingualTextOptions,
    ) -> Result<Vec<f32>> {
        let active = self
            .active
            .as_mut()
            .ok_or(F5Error::InvalidConfiguration("No F5 model loaded"))?;
        let voice = active
            .voices
            .get(&active.selected_voice)
            .ok_or(F5Error::InvalidConfiguration("Active F5 voice is missing"))?;
        synthesize_cross_lingual(
            &mut active.engines,
            &voice.audio,
            text,
            &active.vocabulary,
            active.pinyin.clone(),
            english,
            options,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::trtx_text::TrtChunkKernel;
    use super::*;
    use crate::mel::VocosMel;
    use crate::text_frontend::NoPinyin;
    use std::{cell::RefCell, rc::Rc};

    #[derive(Default)]
    struct Trace {
        shapes: Vec<(usize, usize)>,
        audio: Vec<f32>,
        noises: Vec<Vec<f32>>,
    }
    struct Factory(Rc<RefCell<Trace>>);
    struct Kernel(Rc<RefCell<Trace>>);

    impl TrtKernelFactory for Factory {
        fn build(
            &mut self,
            reference_frames: usize,
            target_frames: usize,
        ) -> Result<Box<dyn TrtChunkKernel>> {
            self.0
                .borrow_mut()
                .shapes
                .push((reference_frames, target_frames));
            Ok(Box::new(Kernel(self.0.clone())))
        }
    }
    impl TrtChunkKernel for Kernel {
        fn synthesize_chunk(
            &mut self,
            reference: &ReferenceAudio,
            _: &[i32],
            target_frames: usize,
            noise: &[f32],
            _: SamplingConfig,
        ) -> Result<Vec<f32>> {
            let mut trace = self.0.borrow_mut();
            trace.audio.push(reference.samples[0]);
            trace.noises.push(noise[..4].to_vec());
            Ok(vec![target_frames as f32 / 100.0; 2048])
        }
        fn predict_reference_speed(&mut self, _: &ReferenceAudio) -> Result<f32> {
            Ok(2.0)
        }
    }
    fn audio(value: f32) -> ReferenceAudio {
        ReferenceAudio::from_channels(&[&vec![value; 24_000]], 24_000).unwrap()
    }
    fn vocabulary() -> Vocabulary {
        Vocabulary::from_lines(
            " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?.,:;"
                .chars()
                .map(|c| c.to_string()),
        )
        .unwrap()
    }
    fn activation(
        id: &str,
        trace: Rc<RefCell<Trace>>,
        capacity: usize,
    ) -> ModelActivation<NoPinyin> {
        ModelActivation {
            model_id: id.to_owned(),
            vocabulary: vocabulary(),
            pinyin: NoPinyin,
            main_voice: ReferenceVoice::new(audio(0.03), "Hi").unwrap(),
            factory: Box::new(Factory(trace)),
            cache_capacity: capacity,
        }
    }
    fn options(seed: u64) -> StandardTextOptions {
        StandardTextOptions {
            speed: 1.0,
            fixed_seconds: Some(1.5),
            cross_fade_seconds: 0.0,
            seed,
            sampling: SamplingConfig::default(),
        }
    }
    #[test]
    fn lazy_shape_factory_reuses_and_evicts_and_failed_reload_keeps_old_model() {
        let trace = Rc::new(RefCell::new(Trace::default()));
        let mut worker = TrtModelWorker::new();
        worker
            .load(activation("model-a", trace.clone(), 1))
            .unwrap();
        worker.synthesize("Hello!", options(17)).unwrap();
        worker.synthesize("Hello!", options(17)).unwrap();
        assert_eq!(trace.borrow().shapes.len(), 1);
        assert_eq!(worker.cached_shapes(), 1);
        assert!(worker.load(activation("bad", trace.clone(), 0)).is_err());
        assert_eq!(worker.model_id(), Some("model-a"));
        worker.synthesize("Hello!", options(17)).unwrap();
        assert_eq!(trace.borrow().shapes.len(), 1);
        worker.synthesize("Hello!", options(18)).unwrap();
        let old = trace.borrow().shapes[0];
        let other = (old.0, old.1 + 1);
        let mut direct =
            super::super::trtx_text::TrtWarmBank::with_factory(Box::new(Factory(trace.clone())), 1)
                .unwrap();
        let ref_audio = audio(0.03);
        let frames = VocosMel::new().extract(&ref_audio.samples).unwrap().frames;
        assert_eq!(frames, old.0);
        direct
            .synthesize_chunk(
                &ref_audio,
                &[1],
                other.1,
                &vec![0.0; other.1 * 100],
                SamplingConfig::default(),
            )
            .unwrap();
        direct
            .synthesize_chunk(
                &ref_audio,
                &[1],
                old.1,
                &vec![0.0; old.1 * 100],
                SamplingConfig::default(),
            )
            .unwrap();
        assert_eq!(direct.warmed_shapes(), 1);
        assert_eq!(trace.borrow().shapes.len(), 3);
        assert!(worker.unload());
        assert_eq!(worker.cached_shapes(), 0);
    }
    #[test]
    fn voice_hot_swap_and_model_replacement() {
        let trace = Rc::new(RefCell::new(Trace::default()));
        let mut worker = TrtModelWorker::new();
        worker.load(activation("f5", trace.clone(), 2)).unwrap();
        worker
            .set_voice("other", ReferenceVoice::new(audio(0.12), "Hello").unwrap())
            .unwrap();
        worker.choose_voice("other").unwrap();
        worker.synthesize("World!", options(19)).unwrap();
        assert!((trace.borrow().audio.last().unwrap() - 0.12).abs() < 0.001);
        worker
            .set_voice("other", ReferenceVoice::new(audio(0.16), "Hello").unwrap())
            .unwrap();
        worker.synthesize("World!", options(19)).unwrap();
        assert!((trace.borrow().audio.last().unwrap() - 0.16).abs() < 0.001);
        worker
            .synthesize_voices(
                "[main] Hi! [other] Hello!",
                1.0,
                21,
                SamplingConfig::default(),
            )
            .unwrap();
        assert!(trace.borrow().audio.len() >= 4);
        worker.remove_voice("other").unwrap();
        assert_eq!(worker.active_voice(), Some("main"));
        worker
            .load(activation("crosslingual", trace.clone(), 2))
            .unwrap();
        assert_eq!(worker.model_id(), Some("crosslingual"));
        assert_eq!(worker.active_voice(), Some("main"));
    }

    struct UnloadableFactory;
    impl TrtKernelFactory for UnloadableFactory {
        fn validate_model(&mut self) -> Result<()> {
            Err(F5Error::InvalidCheckpoint(
                "Cannot read new model".to_owned(),
            ))
        }
        fn build(&mut self, _: usize, _: usize) -> Result<Box<dyn TrtChunkKernel>> {
            panic!("Factory must not be invoked after validation failure")
        }
    }

    #[test]
    fn invalid_checkpoint_replacement_preserves_current_warm_model_and_voice() {
        let trace = Rc::new(RefCell::new(Trace::default()));
        let mut worker = TrtModelWorker::new();
        worker.load(activation("old", trace.clone(), 2)).unwrap();
        worker
            .set_voice("other", ReferenceVoice::new(audio(0.2), "Hello").unwrap())
            .unwrap();
        worker.choose_voice("other").unwrap();
        worker.synthesize("Hello!", options(3)).unwrap();
        let warmed = worker.cached_shapes();
        let bad = ModelActivation {
            model_id: "new".to_owned(),
            vocabulary: vocabulary(),
            pinyin: NoPinyin,
            main_voice: ReferenceVoice::new(audio(0.12), "Hi").unwrap(),
            factory: Box::new(UnloadableFactory),
            cache_capacity: 2,
        };
        assert!(matches!(
            worker.load(bad),
            Err(F5Error::InvalidCheckpoint(_))
        ));
        assert_eq!(worker.model_id(), Some("old"));
        assert_eq!(worker.active_voice(), Some("other"));
        assert_eq!(worker.cached_shapes(), warmed);
        worker.synthesize("Hello!", options(3)).unwrap();
        assert_eq!(trace.borrow().shapes.len(), 1);
    }

    #[derive(Default)]
    struct OracleLog {
        serial: usize,
        calls: Vec<(usize, String)>,
    }
    struct OracleFactory(Rc<RefCell<OracleLog>>);
    struct OracleKernel(Rc<RefCell<OracleLog>>);
    impl TrtKernelFactory for OracleFactory {
        fn build(&mut self, _: usize, _: usize) -> Result<Box<dyn TrtChunkKernel>> {
            Ok(Box::new(OracleKernel(self.0.clone())))
        }
    }
    impl TrtChunkKernel for OracleKernel {
        fn synthesize_chunk(
            &mut self,
            audio: &ReferenceAudio,
            _: &[i32],
            frames: usize,
            _: &[f32],
            _: SamplingConfig,
        ) -> Result<Vec<f32>> {
            let mut log = self.0.borrow_mut();
            let voice = if audio.samples[0] > 0.14 {
                "other"
            } else {
                "main"
            };
            log.calls.push((frames, voice.to_owned()));
            log.serial += 1;
            Ok(vec![log.serial as f32 / 8.0; 8192])
        }
        fn predict_reference_speed(&mut self, _: &ReferenceAudio) -> Result<f32> {
            Ok(2.0)
        }
    }

    #[test]
    fn multi_chunk_multi_voice_pcm_assembly_matches_original_python_numpy() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/trtx_worker_python_expected.json"
        ))
        .unwrap();
        let expected_pcm = include_bytes!("../../tests/fixtures/trtx_worker_python_expected.f32le");
        for case in fixture["cases"].as_array().unwrap() {
            let log = Rc::new(RefCell::new(OracleLog::default()));
            let mut worker = TrtModelWorker::new();
            worker
                .load(ModelActivation {
                    model_id: "numerical-oracle".to_owned(),
                    vocabulary: vocabulary(),
                    pinyin: NoPinyin,
                    main_voice: ReferenceVoice::new(
                        ReferenceAudio::from_channels(&[&vec![0.12; 120_000]], 24_000).unwrap(),
                        "Hi",
                    )
                    .unwrap(),
                    factory: Box::new(OracleFactory(log.clone())),
                    cache_capacity: 3,
                })
                .unwrap();
            let input = &case["input"];
            let samples = if case["kind"] == "standard" {
                worker
                    .synthesize(
                        input["text"].as_str().unwrap(),
                        StandardTextOptions {
                            speed: 1.0,
                            fixed_seconds: input["fixed"].as_f64().map(|value| value as f32),
                            cross_fade_seconds: input["fade"].as_f64().unwrap() as f32,
                            seed: 50,
                            sampling: SamplingConfig::default(),
                        },
                    )
                    .unwrap()
            } else {
                worker
                    .set_voice(
                        "other",
                        ReferenceVoice::new(
                            ReferenceAudio::from_channels(&[&vec![0.16; 120_000]], 24_000).unwrap(),
                            "Hey",
                        )
                        .unwrap(),
                    )
                    .unwrap();
                worker
                    .synthesize_voices(
                        input["script"].as_str().unwrap(),
                        1.0,
                        50,
                        SamplingConfig::default(),
                    )
                    .unwrap()
            };
            let first = case["offset_samples"].as_u64().unwrap() as usize * 4;
            let count = case["sample_count"].as_u64().unwrap() as usize;
            let expected = &expected_pcm[first..first + count * 4];
            assert_eq!(samples.len(), count);
            let maximum = samples
                .iter()
                .zip(expected.chunks_exact(4))
                .map(|(got, bytes)| {
                    let want = f32::from_le_bytes(bytes.try_into().unwrap());
                    (f64::from(*got) - f64::from(want)).abs()
                })
                .fold(0.0_f64, f64::max);
            assert!(
                maximum <= 6.0e-8,
                "Python/NumPy PCM assembly mismatch: {maximum}"
            );
            let expected_calls = case["calls"].as_array().unwrap();
            let actual = &log.borrow().calls;
            assert_eq!(actual.len(), expected_calls.len());
            for (got, want) in actual.iter().zip(expected_calls) {
                assert_eq!(got.0, want["frames"].as_u64().unwrap() as usize);
                assert_eq!(got.1, want["voice"].as_str().unwrap());
            }
        }
    }

    struct RejectShapeFactory {
        log: Rc<RefCell<Trace>>,
        rejected: usize,
    }
    impl TrtKernelFactory for RejectShapeFactory {
        fn build(
            &mut self,
            reference_frames: usize,
            target_frames: usize,
        ) -> Result<Box<dyn TrtChunkKernel>> {
            if target_frames == self.rejected {
                return Err(F5Error::ModelExecution(
                    "Injected shape build failure".into(),
                ));
            }
            self.log
                .borrow_mut()
                .shapes
                .push((reference_frames, target_frames));
            Ok(Box::new(Kernel(self.log.clone())))
        }
    }
    #[test]
    fn failed_shape_compilation_does_not_evict_the_existing_warmed_engine() {
        let log = Rc::new(RefCell::new(Trace::default()));
        let reference = audio(0.12);
        let count = VocosMel::new().extract(&reference.samples).unwrap().frames;
        let mut bank = super::super::trtx_text::TrtWarmBank::with_factory(
            Box::new(RejectShapeFactory {
                log: log.clone(),
                rejected: count + 3,
            }),
            1,
        )
        .unwrap();
        let mut execute = |frames: usize| {
            bank.synthesize_chunk(
                &reference,
                &[0, 1, 2],
                frames,
                &vec![0.5; frames * 100],
                SamplingConfig::default(),
            )
        };
        assert!(execute(count + 2).is_ok());
        assert!(execute(count + 3).is_err());
        assert!(execute(count + 2).is_ok());
        assert_eq!(bank.warmed_shapes(), 1);
        assert_eq!(log.borrow().shapes, vec![(count, count + 2)]);
    }
}
