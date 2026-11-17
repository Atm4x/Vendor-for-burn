pub mod audio;
pub mod decode;
pub mod model;
pub mod weights;

use anyhow::{Context, Result, ensure};
use audio::{SAMPLE_RATE, log_mel, speech_chunks};
use burn::tensor::Device;
use decode::{Transcription, ctc_argmax, ctc_greedy};
use model::{Config, CtcModel};
use serde::Deserialize;
use std::path::Path;
use weights::Weights;

#[derive(Deserialize)]
struct ModelManifest {
    #[serde(flatten)]
    config: Config,
    vocabulary: Vec<String>,
}

pub struct GigaAmCtc {
    model: CtcModel,
    vocabulary: Vec<String>,
    device: Device,
}
impl GigaAmCtc {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let cfg_path = model_dir.join("config.json");
        let manifest: ModelManifest = serde_json::from_slice(
            &std::fs::read(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?,
        )?;
        manifest.config.validate()?;
        ensure!(
            manifest.vocabulary.len() + 1 == manifest.config.num_classes,
            "vocabulary must have num_classes-1 entries"
        );
        let candidates = [
            "v3_ctc.ckpt",
            "model.ckpt",
            "model.safetensors",
            "v3_ctc.safetensors",
        ];
        let weights_path = candidates
            .iter()
            .map(|f| model_dir.join(f))
            .find(|p| p.exists())
            .context("missing v3_ctc.ckpt or model.safetensors")?;
        let mut weights = Weights::open(&weights_path)?;
        let model = CtcModel::load(&mut weights, manifest.config, &device)?;
        Ok(Self {
            model,
            vocabulary: manifest.vocabulary,
            device,
        })
    }
    pub fn infer_logits_16k(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let features = log_mel(pcm, self.model_n_mels())?;
        self.model.infer_logits(&features, &self.device)
    }
    pub fn transcribe_16k(&self, pcm: &[f32]) -> Result<Transcription> {
        let (logits, frames) = self.infer_logits_16k(pcm)?;
        let ids = ctc_argmax(&logits, frames, self.vocabulary.len() + 1)?;
        ctc_greedy(&ids, frames, &self.vocabulary)
    }
    pub fn n_mels(&self) -> usize {
        self.model.n_mels()
    }
    fn model_n_mels(&self) -> usize {
        self.n_mels()
    }
    pub fn transcribe_long_16k(&self, samples: &[f32]) -> Result<Vec<Transcription>> {
        speech_chunks(samples, (SAMPLE_RATE as usize) * 24)
            .iter()
            .map(|chunk| self.transcribe_16k(chunk))
            .collect()
    }
}
