use anyhow::{Context, Result};
use burn::tensor::{Device, Tensor};
use std::{fs, path::Path};

use crate::{acoustic::TextToMel, frontend::ModelConfig, store::TensorStore, synthesis::Synthesizer};

/// An independently loaded, persistent Silero TTS inference session.
/// The source PyTorch package is used only by the developer-side export tool.
pub struct SileroTts {
    config: ModelConfig,
    acoustic: TextToMel,
    synthesizer: Synthesizer,
}

impl SileroTts {
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self> {
        let config: ModelConfig = serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)
            .context("Invalid Silero checkpoint configuration")?;
        let tts = TensorStore::open(&model_dir.join("tts.safetensors"))?;
        let filters = TensorStore::open(&model_dir.join("pqmf.safetensors"))?;
        Ok(Self {
            config,
            acoustic: TextToMel::load(&tts, device)?,
            synthesizer: Synthesizer::load(&tts, &filters, device)?,
        })
    }

    /// Converts an explicitly stressed Silero-symbol string to PCM.
    /// A separate Accentor/HomoSolver pass is required for unstressed text.
    pub fn infer_accented(&self, text: &str, speaker: &str, sample_rate: u32) -> Result<Vec<f32>> {
        let speaker = self.config.validate_speaker(speaker)?;
        let (ids, _) = self.config.encode_accented(text)?;
        let pcm = self.infer_token_ids(&ids, speaker as i64, sample_rate)?;
        Ok(pcm.into_data().try_to_vec::<f32>()?)
    }

    /// Complete network inference from symbol ids through the vocoder.
    /// PCM has shape `[1, samples]`. The loaded session remains reusable.
    pub fn infer_token_ids(&self, symbols: &[i64], speaker: i64, sample_rate: u32) -> Result<Tensor<2>> {
        self.config.validate_sample_rate(sample_rate)?;
        let (mel, _) = self.acoustic.infer(symbols, speaker)?;
        self.synthesizer.forward(mel, sample_rate)
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
