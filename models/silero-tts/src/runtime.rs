use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use burn::tensor::{Device, Tensor};
use std::{collections::{HashSet, VecDeque}, fs, path::Path};

use crate::{longform, acoustic::TextToMel, accentor::{Accentor, AccentOptions}, homosolver::HomoSolver, frontend::ModelConfig, store::TensorStore, synthesis::Synthesizer, ssml::{self, SileroControlConstants}};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TextOptions {
    pub put_accent: bool,
    pub put_stress_homo: bool,
    pub put_yo: bool,
    pub put_yo_homo: bool,
    pub stress_single_vowel: bool,
}

impl Default for TextOptions {
    fn default() -> Self {
        Self {
            put_accent: true,
            put_stress_homo: true,
            put_yo: true,
            put_yo_homo: true,
            stress_single_vowel: true,
        }
    }
}

/// An independently loaded, persistent Silero TTS inference session.
/// The source PyTorch package is used only by the developer-side export tool.
pub struct SileroTts {
    config: ModelConfig,
    acoustic: TextToMel,
    accentor: Accentor,
    homographs: HashSet<String>,
    homosolver: HomoSolver,
    synthesizer: Synthesizer,
    control_constants: Option<SileroControlConstants>,
}

impl SileroTts {
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let config: ModelConfig = serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)
            .context("Invalid Silero checkpoint configuration")?;
        let tts = TensorStore::open(&model_dir.join("tts.safetensors"))?;
        let filters = TensorStore::open(&model_dir.join("pqmf.safetensors"))?;
        let homographs: serde_json::Value = serde_json::from_slice(
            &fs::read(model_dir.join("homographs.json"))?)?;
        let homographs = homographs.as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid homograph dictionary"))?
            .keys().cloned().collect();
        let control_constants = match fs::read(model_dir.join("ssml-controls.json")) {
            Ok(data) => Some(serde_json::from_slice(&data).context("Invalid Silero SSML control constants")?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        Ok(Self {
            control_constants,
            accentor: Accentor::load(model_dir, device)?,
            homosolver: HomoSolver::load(model_dir, device)?,
            homographs,
            config,
            acoustic: TextToMel::load(&tts, device)?,
            synthesizer: Synthesizer::load(&tts, &filters, device)?,
        })
    }

    /// Full SileroStress frontend, including independently controlled homograph and ngram stages.
    pub fn accentuate_with_options(&self, text: &str, flags: &TextOptions) -> Result<String> {
        let cleaned = self.config.prepare_text(text)?;
        self.accentuate_prepared(&cleaned.sentence, flags)
    }

    fn accentuate_prepared(&self, text: &str, flags: &TextOptions) -> Result<String> {
        let resolved = self.homosolver.accentuate(text,
            flags.put_stress_homo, flags.put_yo_homo, flags.stress_single_vowel)?;
        let mut accent_flags = AccentOptions {
            put_stress: flags.put_accent,
            put_yo: flags.put_yo,
            stress_single_vowel: flags.stress_single_vowel,
            ..Default::default()
        };
        if !flags.put_stress_homo {
            accent_flags.skip_stress_words = self.homosolver.variant_words()
                .map(|(key, _)| key.clone()).collect();
        }
        if !flags.put_yo_homo {
            accent_flags.skip_yo_words = self.homosolver.variant_words()
                .filter(|(_, variants)| variants.iter().any(|v| v.contains('ё')))
                .map(|(key, _)| key.clone()).collect();
        }
        self.accentor.accentuate(&resolved, &accent_flags)
    }

    /// Full default SileroStress frontend: contextual BERT homographs followed by Accentor.
    pub fn accentuate(&self, text: &str) -> Result<String> {
        self.accentuate_with_options(text, &TextOptions::default())
    }

    /// Default non-SSML text-to-PCM synthesis with automatic stress and homograph resolution.
    pub fn infer_text(&self, text: &str, speaker: &str, sample_rate: u32) -> Result<Vec<f32>> {
        self.infer_text_with_options(text, speaker, sample_rate, &TextOptions::default())
    }

    pub fn infer_text_with_options(&self, text: &str, speaker: &str, sample_rate: u32,
        flags: &TextOptions) -> Result<Vec<f32>> {
        let speaker_id = self.config.validate_speaker(speaker)? as i64;
        self.config.validate_sample_rate(sample_rate)?;
        longform::infer_text_chunks(text, sample_rate, |fragment| {
            let accented = self.accentuate_with_options(fragment, flags)?;
            self.infer_accented_single(&accented, speaker_id, sample_rate)
        })
    }

    /// Run the same Silero acoustic model with explicitly selected RUAccent text preparation.
    /// Normal `text` requests always retain SileroStress; this method is opt-in.
    #[cfg(feature = "ruaccent")]
    pub fn infer_text_with_ruaccent(
        &self,
        text: &str,
        speaker: &str,
        sample_rate: u32,
        ruaccent: &mut ruaccent_rs::RuAccent,
    ) -> Result<Vec<f32>> {
        let speaker_id = self.config.validate_speaker(speaker)? as i64;
        self.config.validate_sample_rate(sample_rate)?;
        longform::infer_text_chunks(text, sample_rate, |fragment| {
            let prepared = self.config.prepare_text(fragment)?;
            let accented = ruaccent.process_all(&prepared.sentence);
            self.infer_accented_single(&accented, speaker_id, sample_rate)
        })
    }

    /// Original `<speak>` SSML entrypoint with nested prosody, sentence breaks,
    /// pause overrides, and voice-specific F0 adjustment. The session stays loaded.
    pub fn infer_ssml(&self, ssml: &str, speaker: &str, sample_rate: u32) -> Result<Vec<f32>> {
        self.infer_ssml_with_options(ssml, speaker, sample_rate, &TextOptions::default())
    }

    pub fn infer_ssml_with_options(&self, ssml: &str, speaker: &str, sample_rate: u32,
        flags: &TextOptions) -> Result<Vec<f32>> {
        Ok(self.infer_ssml_with_durations(ssml, speaker, sample_rate, flags)?.0)
    }

    fn infer_ssml_with_durations(&self, ssml: &str, speaker: &str, sample_rate: u32,
        flags: &TextOptions) -> Result<(Vec<f32>, Vec<f32>)> {
        let speaker_id = self.config.validate_speaker(speaker)? as i64;
        self.config.validate_sample_rate(sample_rate)?;
        let constants = self.control_constants.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Missing ssml-controls.json; export model weights again"))?;
        let segments = ssml::process_ssml(&self.config, ssml)?;
        let total_chars = segments.iter().map(|s| s.text.chars().count()).sum::<usize>();
        anyhow::ensure!(total_chars <= longform::MAX_INPUT_CHARS,
            "Silero SSML text exceeds long-form input limit");
        let mut pending: VecDeque<Vec<ssml::Segment>> = longform::group_ssml_segments(&self.config, segments)?.into();
        let mut output = Vec::new();
        let mut durations = Vec::new();
        let mut completed = 0;
        while let Some(group) = pending.pop_front() {
            match self.infer_ssml_group(&group, speaker_id, sample_rate, flags, constants) {
                Ok((pcm, timing)) => {
                    longform::append_audio(&mut output, pcm, sample_rate)?;
                    durations.extend(timing);
                    completed += 1;
                }
                Err(error) if longform::is_acoustic_limit(&error) => {
                    let halves = if group.len() > 1 {
                        let midpoint = group.len() / 2;
                        vec![group[..midpoint].to_vec(), group[midpoint..].to_vec()]
                    } else {
                        let item = &group[0];
                        let max = (item.text.chars().count() / 2).max(2);
                        longform::split_ssml_segment(&self.config, item, max)?.into_iter()
                            .map(|single| vec![single]).collect::<Vec<_>>()
                    };
                    if halves.len() <= 1 {
                        return Err(error.context("Cannot split SSML segment below acoustic limit"));
                    }
                    anyhow::ensure!(completed + pending.len() + halves.len() <= longform::MAX_CHUNKS,
                        "Silero SSML segment count exceeds limit");
                    for half in halves.into_iter().rev() {
                        pending.push_front(half);
                    }
                }
                Err(error) => return Err(error.context(format!(
                    "Silero SSML long-form inference failed in segment {}", completed + 1,
                ))),
            }
        }
        Ok((output, durations))
    }

    fn infer_ssml_group(&self, segments: &[ssml::Segment], speaker_id: i64,
        sample_rate: u32, flags: &TextOptions,
        constants: &SileroControlConstants) -> Result<(Vec<f32>, Vec<f32>)> {
        let prepared = segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
        let accented = self.accentuate_prepared(&prepared, flags)?;
        let merged = ssml::merge_segments(segments, &accented)?;
        let (ids, _) = self.config.encode_accented(&merged.text)?;
        let (mel, durations) = self.acoustic.infer_with_controls(&ids, speaker_id, &merged, constants)?;
        let pcm = self.synthesizer.forward(mel, sample_rate)?.into_data().try_to_vec::<f32>()?;
        Ok((pcm, durations))
    }

    /// Reproduces Silero's original `symbol_durs` API by transforming matched
    /// substrings into SSML breaks before normalization and accentuation.
    pub fn infer_text_with_symbol_durations(&self, text: &str, speaker: &str,
        sample_rate: u32, symbol_durs: &[(String, u32)], flags: &TextOptions) -> Result<Vec<f32>> {
        let ssml = ssml::symbol_durations_to_ssml(text, symbol_durs)?;
        self.infer_ssml_with_options(&ssml, speaker, sample_rate, flags)
    }

    /// Recreates the original Russian frontend for input without unresolved homographs.
    /// No silent guessing is allowed until the original HomoSolver/BERT is ported.
    pub fn accentuate_unambiguous(&self, text: &str) -> Result<String> {
        let cleaned = self.config.prepare_text(text)?;
        let mut word = String::new();
        for ch in cleaned.sentence.chars().chain(std::iter::once(' ')) {
            if ('а'..='я').contains(&ch) || ch == 'ё' || ch == '+' {
                word.push(ch);
            } else {
                if self.homographs.contains(&word) {
                    anyhow::bail!("Unresolved Silero homograph {word:?}; native BERT HomoSolver is required");
                }
                word.clear();
            }
        }
        self.accentor.accentuate(&cleaned.sentence, &AccentOptions::original_defaults())
    }

    /// Full native frontend and text-to-PCM inference when no homographs are present.
    pub fn infer_text_unambiguous(&self, text: &str, speaker: &str, sample_rate: u32) -> Result<Vec<f32>> {
        let accented = self.accentuate_unambiguous(text)?;
        self.infer_accented(&accented, speaker, sample_rate)
    }

    /// Converts an explicitly stressed Silero-symbol string to PCM.
    /// A separate Accentor/HomoSolver pass is required for unstressed text.
    pub fn infer_accented(&self, text: &str, speaker: &str, sample_rate: u32) -> Result<Vec<f32>> {
        let speaker_id = self.config.validate_speaker(speaker)? as i64;
        self.config.validate_sample_rate(sample_rate)?;
        longform::infer_text_chunks(text, sample_rate, |fragment| {
            self.infer_accented_single(fragment, speaker_id, sample_rate)
        })
    }

    fn infer_accented_single(&self, text: &str, speaker_id: i64, sample_rate: u32) -> Result<Vec<f32>> {
        let (ids, _) = self.config.encode_accented(text)?;
        let pcm = self.infer_token_ids(&ids, speaker_id, sample_rate)?;
        Ok(pcm.into_data().try_to_vec::<f32>()?)
    }

    /// Complete network inference from symbol ids through the vocoder.
    /// PCM has shape `[1, samples]`. The loaded session remains reusable.
    pub fn infer_token_ids(&self, symbols: &[i64], speaker: i64, sample_rate: u32) -> Result<Tensor<2>> {
        self.config.validate_sample_rate(sample_rate)?;
        let (mel, _) = self.acoustic.infer(symbols, speaker)?;
        self.synthesizer.forward(mel, sample_rate)
    }

    /// Produce Silero's 16-bit mono WAV representation from the native text pipeline.
    pub fn save_wav(&self, text: &str, path: &Path, speaker: &str,
        sample_rate: u32, flags: &TextOptions) -> Result<()> {
        let pcm = self.infer_text_with_options(text, speaker, sample_rate, flags)?;
        crate::audio::write_wav_pcm16(path, &pcm, sample_rate)
    }

    pub fn save_ssml_wav(&self, ssml: &str, path: &Path, speaker: &str,
        sample_rate: u32, flags: &TextOptions) -> Result<()> {
        let pcm = self.infer_ssml_with_options(ssml, speaker, sample_rate, flags)?;
        crate::audio::write_wav_pcm16(path, &pcm, sample_rate)
    }

    pub fn speakers(&self) -> impl Iterator<Item = &str> {
        self.config.speakers.keys().map(String::as_str)
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn reference_full_tts_pcm_8khz() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = Path::new(&dir);
        let device = Device::flex();
        let model = SileroTts::load(dir, &device).unwrap();
        let input: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("acoustic-input.json")).unwrap()).unwrap();
        let ids: Vec<i64> = serde_json::from_value(input["tokens"].clone()).unwrap();
        let prediction = model.infer_token_ids(&ids, 0, 8000).unwrap().into_data().try_to_vec::<f32>().unwrap();
        let reference = TensorStore::open(&dir.join("acoustic-parity.safetensors")).unwrap();
        let actual = reference.float_data("full_pcm8000").unwrap();
        assert_eq!(prediction.len(), actual.len());
        let error = prediction.iter().zip(actual.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        println!("full token-to-PCM error: max_abs={error:.9}; samples={}", prediction.len());
        assert!(error < 2.0e-3);
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod text_parity {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Reference {
        text: String,
        accented: String,
        speaker: String,
        sample_rate: u32,
        key: String,
    }

    #[test]
    fn original_text_to_pcm() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = Path::new(&dir);
        let cases: Vec<Reference> = serde_json::from_slice(&fs::read(dir.join("full-text-parity.json")).unwrap()).unwrap();
        let references = TensorStore::open(&dir.join("full-text-parity.safetensors")).unwrap();
        let model = SileroTts::load(dir, &Device::flex()).unwrap();
        for sample in cases {
            let accented = model.accentuate(&sample.text).unwrap();
            assert_eq!(accented, sample.accented, "Text frontend mismatch");
            let pcm = model.infer_text(&sample.text, &sample.speaker, sample.sample_rate).unwrap();
            let expected = references.float_data(&sample.key).unwrap();
            assert_eq!(pcm.len(), expected.len(), "Audio duration differs for {:?}", sample.text);
            let error = pcm.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
            println!("Full text-to-PCM {:?}: samples={}, max_abs={error:.8}", sample.text, pcm.len());
            assert!(error < 3e-3, "Full text-to-PCM parity failed");
        }
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod flags_parity {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Case {
        text: String,
        options: serde_json::Map<String, serde_json::Value>,
        accented: String,
    }

    #[test]
    fn original_optional_stress_flags() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = Path::new(&dir);
        let cases: Vec<Case> = serde_json::from_slice(&fs::read(dir.join("stress-flags-parity.json")).unwrap()).unwrap();
        let model = SileroTts::load(dir, &Device::flex()).unwrap();
        for case in cases {
            let mut flags = TextOptions::default();
            for (key, value) in &case.options {
                let value = value.as_bool().unwrap();
                match key.as_str() {
                    "put_accent" => flags.put_accent = value,
                    "put_stress_homo" => flags.put_stress_homo = value,
                    "put_yo" => flags.put_yo = value,
                    "put_yo_homo" => flags.put_yo_homo = value,
                    "stress_single_vowel" => flags.stress_single_vowel = value,
                    _ => panic!("Unknown flag: {key}"),
                }
            }
            let actual = model.accentuate_with_options(&case.text, &flags).unwrap();
            assert_eq!(actual, case.accented, "Mismatch for {:?} with {:?}", case.text, case.options);
        }
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod ssml_parity {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Case {
        ssml: String,
        key: String,
        durations: Vec<f32>,
    }

    #[test]
    fn original_ssml_mel_with_prosody_and_breaks() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = std::path::Path::new(&dir);
        let cases: Vec<Case> = serde_json::from_slice(&fs::read(dir.join("ssml-audio-parity.json")).unwrap()).unwrap();
        let mel_reference = TensorStore::open(&dir.join("ssml-mel-parity.safetensors")).unwrap();
        let model = SileroTts::load(dir, &Device::flex()).unwrap();
        let constants = model.control_constants.as_ref().unwrap();
        for case in cases {
            let segments = ssml::process_ssml(&model.config, &case.ssml).unwrap();
            let prepared = segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
            let accented = model.accentuate_prepared(&prepared, &TextOptions::default()).unwrap();
            let merged = ssml::merge_segments(&segments, &accented).unwrap();
            let (ids, _) = model.config.encode_accented(&merged.text).unwrap();
            let (mel, durations) = model.acoustic.infer_with_controls(&ids, 0, &merged, constants).unwrap();
            assert_eq!(durations.len(), case.durations.len());
            let duration_error = durations.iter().zip(&case.durations).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
            println!("SSML {}: duration max_abs={duration_error:.8}; mismatches={:?}", case.key,
                durations.iter().zip(&case.durations).enumerate()
                    .filter(|(_, (a, b))| (*a - *b).abs() > 0.1)
                    .map(|(i, (a, b))| (i, *a, *b)).collect::<Vec<_>>());
            assert!(duration_error < 0.1, "Duration parity failed: {duration_error}");
            let key = case.key.replace("pcm_", "mel_");
            let reference = mel_reference.float_data(&key).unwrap();
            let got = mel.into_data().try_to_vec::<f32>().unwrap();
            assert_eq!(got.len(), reference.len(), "Mel frame count differs: {}", case.key);
            let mel_error = got.iter().zip(&reference).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
            println!("SSML {}: mel max_abs={mel_error:.8}, frames={}", case.key, got.len()/192);
            assert!(mel_error < 0.005, "SSML mel parity failed for {}: {mel_error}", case.key);
        }
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod ssml_pcm_parity {
    use super::*;

    #[test]
    fn short_text_with_pause_produces_pytorch_pcm() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = std::path::Path::new(&dir);
        let references: Vec<serde_json::Value> = serde_json::from_slice(
            &fs::read(dir.join("ssml-audio-parity.json")).unwrap()).unwrap();
        let case = references.iter().find(|case| case["key"] == "pcm_12").unwrap();
        let audio = TensorStore::open(&dir.join("ssml-audio-parity.safetensors")).unwrap();
        let expected = audio.float_data("pcm_12").unwrap();
        let model = SileroTts::load(dir, &Device::flex()).unwrap();
        let got = model.infer_ssml(case["ssml"].as_str().unwrap(), "aidar", 8000).unwrap();
        assert_eq!(got.len(), expected.len());
        let error = got.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        println!("short SSML pcm_12 max_abs={error:.8}; samples={}", got.len());
        assert!(error < 3e-3);
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod symbol_duration_pcm_parity {
    use super::*;

    #[test]
    fn original_symbol_durations_pcm_all_sample_rates() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = std::path::Path::new(&dir);
        let expected = TensorStore::open(&dir.join("symbol-durations-audio.safetensors")).unwrap();
        let model = SileroTts::load(dir, &Device::flex()).unwrap();
        let controls = [("а".to_string(), 75)];
        for sample_rate in [8000, 24000, 48000] {
            let actual = model.infer_text_with_symbol_durations(
                "Да.", "aidar", sample_rate, &controls, &TextOptions::default()
            ).unwrap();
            let reference = expected.float_data(&format!("pcm_{sample_rate}")).unwrap();
            assert_eq!(actual.len(), reference.len(), "Sample count mismatch at {sample_rate}");
            let max_abs = actual.iter().zip(reference.iter())
                .map(|(x, y)| (x-y).abs()).fold(0.0_f32, f32::max);
            println!("symbol_durs {sample_rate}: samples={}, max_abs={max_abs:.8}", actual.len());
            assert!(max_abs < 0.003, "symbol_durs PCM parity failed at {sample_rate}");
        }
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod multispeaker_ssml_parity {
    use super::*;

    #[test]
    fn original_all_five_speakers_with_duration_and_pitch_controls() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = std::path::Path::new(&dir);
        let source: Vec<serde_json::Value> = serde_json::from_slice(
            &fs::read(dir.join("symbol-durations-parity.json")).unwrap()).unwrap();
        let ids: Vec<i64> = serde_json::from_value(source[0]["ids"].clone()).unwrap();
        let merged = ssml::MergeInput {
            text: String::new(),
            pauses: serde_json::from_value(source[0]["pauses"].clone()).unwrap(),
            rates: serde_json::from_value(source[0]["rates"].clone()).unwrap(),
            pitches: serde_json::from_value(source[0]["pitches"].clone()).unwrap(),
        };
        let reference = TensorStore::open(&dir.join("multispeaker-ssml-parity.safetensors")).unwrap();
        let manifest: Vec<serde_json::Value> = serde_json::from_slice(
            &fs::read(dir.join("multispeaker-ssml-parity.json")).unwrap()).unwrap();
        let model = SileroTts::load(dir, &Device::flex()).unwrap();
        let constants = model.control_constants.as_ref().unwrap();
        for voice in manifest {
            let name = voice["speaker"].as_str().unwrap();
            let speaker_id = model.config.validate_speaker(name).unwrap();
            assert_eq!(speaker_id, voice["speaker_id"].as_u64().unwrap() as usize);
            let (mel, durations) = model.acoustic.infer_with_controls(
                &ids, speaker_id as i64, &merged, constants
            ).unwrap();
            let expected_durations: Vec<f32> = serde_json::from_value(voice["durations"].clone()).unwrap();
            assert_eq!(durations, expected_durations, "Duration mismatch: {name}");
            let got = mel.into_data().try_to_vec::<f32>().unwrap();
            let expected = reference.float_data(name).unwrap();
            assert_eq!(got.len(), expected.len(), "Mel length mismatch: {name}");
            let max_abs = got.iter().zip(expected.iter())
                .map(|(x,y)| (x-y).abs()).fold(0.0_f32, f32::max);
            println!("voice {name}: mel max_abs={max_abs:.8}; frames={}", got.len()/192);
            assert!(max_abs < 0.003, "SSML mel parity failed for speaker {name}");
        }
    }
}
