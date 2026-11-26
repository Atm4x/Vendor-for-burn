use anyhow::{Result, ensure};
use burn::tensor::{Device, Tensor};
use silero_package::{EnglishCheckpoint, encode_english, normalize_english};
use std::collections::BTreeMap;
use crate::{acoustic::TextToMel, english_vocoder::EnglishVocoder, frontend::ModelConfig, longform, store::TensorStore};

pub(crate) struct EnglishTts {
    pub(crate) config: ModelConfig,
    ascii_symbols: BTreeMap<char, String>,
    acoustic: TextToMel,
    vocoder: EnglishVocoder,
}

impl EnglishTts {
    pub(crate) fn load(mut checkpoint: EnglishCheckpoint, device: &Device) -> Result<Self> {
        let config = ModelConfig { format_version: 1, symbols: std::mem::take(&mut checkpoint.symbols), alphabet: "abcdefghijklmnopqrstuvwxyz".into(),
            sos: "|".into(), eos: "~".into(), speakers: std::mem::take(&mut checkpoint.speakers), sample_rates: vec![8000, 24000, 48000] };
        let ascii_symbols = std::mem::take(&mut checkpoint.ascii_symbols);
        let weights = TensorStore::native(checkpoint.into_weights()?);
        let acoustic = TextToMel::load_variant(&weights, device, false)?;
        let vocoder = EnglishVocoder::load(&weights, device)?;
        Ok(Self { config, ascii_symbols, acoustic, vocoder })
    }

    pub(crate) fn infer_text(&self, text: &str, speaker: &str, rate: u32) -> Result<Vec<f32>> {
        let speaker_id = self.config.validate_speaker(speaker)? as i64;
        self.config.validate_sample_rate(rate)?;
        longform::infer_text_chunks(text, rate, |fragment| {
            let normalized = normalize_english(&self.config.symbols, &self.ascii_symbols, fragment)?;
            let tokens = encode_english(&self.config.symbols, &normalized)?;
            self.infer_token_ids(&tokens, speaker_id, rate)?.into_data().try_to_vec::<f32>().map_err(Into::into)
        })
    }

    pub(crate) fn infer_token_ids(&self, tokens: &[i64], speaker: i64, rate: u32) -> Result<Tensor<2>> {
        self.config.validate_sample_rate(rate)?;
        ensure!(self.config.speakers.values().any(|&id| id as i64 == speaker), "Unknown English speaker ID");
        let start = self.vocoder.profile_enabled().then(std::time::Instant::now);
        let (mel, _) = self.acoustic.infer_english(tokens, speaker)?;
        if let Some(start) = start { eprintln!("silero_en_profile stage=acoustic index=- ms={:.6}", start.elapsed().as_secs_f64() * 1000.0); }
        self.vocoder.forward(mel, rate)
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod tests {
    use super::*;
    use std::{fs, path::PathBuf};
    use serde::Deserialize;
    use burn::tensor::{Int, TensorData};

    #[derive(Deserialize)]
    struct Case { prefix: String, text: String, speaker: String, speaker_id: i64, tokens: Vec<i64> }

    fn compare(label: &str, got: &[f32], reference: &[f32], tolerance: f32) {
        assert_eq!(got.len(), reference.len(), "{label} shape");
        let max = got.iter().zip(reference).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        let error = got.iter().zip(reference).map(|(a,b)| ((a-b) as f64).powi(2)).sum::<f64>();
        let signal = reference.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
        let snr = 10.0 * (signal / error).log10();
        println!("{label}: count={} max_abs={max:.9} SNR={snr:.3} dB", got.len());
        assert!(got.iter().all(|v| v.is_finite()) && max <= tolerance, "{label} max_abs={max}");
    }

    #[test]
    #[ignore = "requires original v3_en and independent reference fixture"]
    fn official_english_acoustic_and_vocoder_parity() {
        let checkpoint = PathBuf::from(std::env::var_os("SILERO_V3_EN_CHECKPOINT").expect("checkpoint path"));
        let root = PathBuf::from(std::env::var_os("SILERO_EN_PARITY_DIR").expect("reference fixture"));
        let cases: Vec<Case> = serde_json::from_slice(&fs::read(root.join("english.json")).unwrap()).unwrap();
        let reference = TensorStore::open(&root.join("english.safetensors")).unwrap();
        let device = Device::flex();
        let model = EnglishTts::load(EnglishCheckpoint::open(checkpoint).unwrap(), &device).unwrap();
        assert_eq!(cases.len(), 2);
        for case in cases {
            let name = &case.prefix;
            let normalized = normalize_english(&model.config.symbols, &model.ascii_symbols, &case.text).unwrap();
            let ids = encode_english(&model.config.symbols, &normalized).unwrap();
            assert_eq!(ids, case.tokens);
            assert_eq!(model.config.validate_speaker(&case.speaker).unwrap() as i64, case.speaker_id);
            let len = ids.len();
            let tokens = Tensor::<2, Int>::from_data(TensorData::new(ids.clone(), [1, len]), &device);
            let speaker = Tensor::<2, Int>::from_data(TensorData::new(vec![case.speaker_id], [1, 1]), &device);
            for (label, actual) in [
                ("log_duration", model.acoustic.duration.forward(tokens.clone(), speaker.clone(), None, 1.0)),
                ("pitch", model.acoustic.pitch.forward(tokens, speaker, None, 1.0)),
            ] {
                compare(&format!("{name}_{label}"), &actual.into_data().try_to_vec::<f32>().unwrap(), &reference.float_data(&format!("{name}_{label}")).unwrap(), 2e-4);
            }
            let (mel, duration) = model.acoustic.infer_english(&ids, case.speaker_id).unwrap();
            assert_eq!(duration, reference.float_data(&format!("{name}_duration")).unwrap());
            compare(&format!("{name}_mel"), &mel.clone().into_data().try_to_vec::<f32>().unwrap(), &reference.float_data(&format!("{name}_mel")).unwrap(), 5e-4);
            let reference_mel = reference.float::<3>(&format!("{name}_mel"), &device).unwrap();
            for rate in [8000, 24000, 48000] {
                let expected = reference.float_data(&format!("{name}_pcm{rate}")).unwrap();
                let isolated = model.vocoder.forward(reference_mel.clone(), rate).unwrap().into_data().try_to_vec::<f32>().unwrap();
                compare(&format!("{name}_vocoder{rate}"), &isolated, &expected, 1e-3);
                let pcm = model.vocoder.forward(mel.clone(), rate).unwrap().into_data().try_to_vec::<f32>().unwrap();
                compare(&format!("{name}_pcm{rate}"), &pcm, &expected, 1e-3);
            }
        }
    }
}
