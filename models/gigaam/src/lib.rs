pub mod audio;
pub mod decode;
pub mod model;
pub mod rnnt;
pub mod tokenizer;
pub mod weights;

use anyhow::{Context, Result, bail, ensure};
use audio::{SAMPLE_RATE, log_mel, speech_chunks};
use burn::tensor::Device;
use decode::{Transcription, ctc_argmax};
use model::{Config, CtcModel, EncoderModel};
use rnnt::{RnntConfig, RnntModel};
use serde::Deserialize;
use std::path::Path;
use tokenizer::Tokenizer;
use weights::Weights;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Ctc,
    Rnnt,
    E2eCtc,
    E2eRnnt,
    Ssl,
}
impl ModelKind {
    pub fn is_ctc(self) -> bool {
        matches!(self, Self::Ctc | Self::E2eCtc)
    }
    pub fn is_rnnt(self) -> bool {
        matches!(self, Self::Rnnt | Self::E2eRnnt)
    }
    pub fn is_ssl(self) -> bool {
        self == Self::Ssl
    }
    pub fn from_model_name(name: &str) -> Self {
        match (
            name.contains("e2e"),
            name.ends_with("_rnnt"),
            name.ends_with("_ssl"),
        ) {
            (true, true, _) => Self::E2eRnnt,
            (true, false, _) => Self::E2eCtc,
            (_, true, _) => Self::Rnnt,
            (_, _, true) => Self::Ssl,
            _ => Self::Ctc,
        }
    }
}

#[derive(Deserialize)]
struct ModelManifest {
    #[serde(flatten)]
    config: Config,
    #[serde(default)]
    vocabulary: Vec<String>,
    #[serde(default)]
    model_name: Option<String>,
    #[serde(default)]
    model_type: Option<ModelKind>,
    #[serde(default)]
    sentencepiece_model: Option<String>,
    #[serde(default)]
    rnnt: Option<RnntConfig>,
}

enum Network {
    Ctc(CtcModel),
    Rnnt(RnntModel),
    Ssl(EncoderModel),
}

pub struct GigaAm {
    network: Network,
    tokenizer: Option<Tokenizer>,
    device: Device,
    n_mels: usize,
    kind: ModelKind,
}

pub type GigaAmCtc = GigaAm;

impl GigaAm {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let cfg_path = model_dir.join("config.json");
        let manifest: ModelManifest = serde_json::from_slice(
            &std::fs::read(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?,
        )?;
        manifest.config.validate()?;
        let kind = manifest.model_type.unwrap_or_else(|| {
            manifest
                .model_name
                .as_deref()
                .map(ModelKind::from_model_name)
                .unwrap_or(ModelKind::Ctc)
        });
        let num_classes = manifest.config.num_classes;
        let n_mels = manifest.config.n_mels;
        let tokenizer = if kind.is_ssl() {
            None
        } else {
            let sp_path = if let Some(path) = manifest.sentencepiece_model {
                ensure!(
                    !path.contains('/') && !path.contains('\\') && !path.contains(".."),
                    "sentencepiece_model must be a filename in model_dir"
                );
                Some(model_dir.join(path))
            } else if matches!(kind, ModelKind::E2eCtc | ModelKind::E2eRnnt) {
                Some(model_dir.join(match kind {
                    ModelKind::E2eCtc => "v3_e2e_ctc_tokenizer.model",
                    _ => "v3_e2e_rnnt_tokenizer.model",
                }))
            } else {
                None
            };
            let tok = Tokenizer::load(manifest.vocabulary, sp_path.as_deref())?;
            ensure!(
                tok.len() + 1 == num_classes,
                "tokenizer size {} + blank != configured num_classes {num_classes}",
                tok.len()
            );
            Some(tok)
        };
        let rnnt_config = if kind.is_rnnt() {
            let rnnt = manifest.rnnt.context(
                "RNN-T requires rnnt config (pred_hidden, pred_rnn_layers, joint_hidden)",
            )?;
            rnnt.validate()?;
            Some(rnnt)
        } else {
            None
        };
        let mut weight_names = Vec::new();
        if let Some(name) = &manifest.model_name {
            ensure!(
                name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'),
                "model_name must contain only letters, numbers or underscores"
            );
            weight_names.push(format!("{name}.ckpt"));
            weight_names.push(format!("{name}.safetensors"));
        }
        weight_names.extend(["model.ckpt".to_owned(), "model.safetensors".to_owned()]);
        if kind == ModelKind::Ctc {
            weight_names.extend(["v3_ctc.ckpt".to_owned(), "v3_ctc.safetensors".to_owned()]);
        }
        let weights_path = weight_names
            .iter()
            .map(|n| model_dir.join(n))
            .find(|p| p.is_file())
            .context("GigaAM model weights not found")?;
        let mut weights = Weights::open(&weights_path)?;
        let network = if kind.is_ctc() {
            Network::Ctc(CtcModel::load(
                &mut weights,
                manifest.config.clone(),
                &device,
            )?)
        } else if let Some(rnnt) = rnnt_config {
            Network::Rnnt(RnntModel::load(
                &mut weights,
                manifest.config.clone(),
                rnnt,
                &device,
            )?)
        } else {
            Network::Ssl(EncoderModel::load(
                &mut weights,
                manifest.config.clone(),
                &device,
            )?)
        };
        Ok(Self {
            network,
            tokenizer,
            device,
            n_mels,
            kind,
        })
    }

    pub fn model_kind(&self) -> ModelKind {
        self.kind
    }
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    pub fn infer_logits_16k(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        ensure!(
            pcm.len() <= SAMPLE_RATE as usize * 25,
            "audio exceeds 25 seconds: split before inference"
        );
        let features = log_mel(pcm, self.n_mels)?;
        match &self.network {
            Network::Ctc(m) => m.infer_logits(&features, &self.device),
            _ => bail!("infer_logits_16k is only available for CTC variants"),
        }
    }

    pub fn infer_embeddings_16k(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        ensure!(
            pcm.len() <= SAMPLE_RATE as usize * 25,
            "audio exceeds 25 seconds: split before inference"
        );
        let features = log_mel(pcm, self.n_mels)?;
        let net = match &self.network {
            Network::Ssl(net) => net,
            _ => bail!("infer_embeddings_16k is only available for SSL models"),
        };
        let encoded = net.encode(&features, &self.device)?;
        let [batch, frames, channels] = encoded.dims();
        ensure!(batch == 1, "only single-sample embeddings supported");
        let values = encoded.into_data().try_to_vec::<f32>()?;
        Ok((values, frames, channels))
    }

    pub fn transcribe_16k(&self, pcm: &[f32]) -> Result<Transcription> {
        ensure!(
            pcm.len() <= SAMPLE_RATE as usize * 25,
            "audio exceeds 25 seconds: use transcribe_long_16k"
        );
        let tokenizer = self
            .tokenizer
            .as_ref()
            .context("SSL models do not produce transcripts")?;
        let features = log_mel(pcm, self.n_mels)?;
        let (ids, frames) = match &self.network {
            Network::Ctc(m) => {
                let (logits, time) = m.infer_logits(&features, &self.device)?;
                let labels = ctc_argmax(&logits, time, tokenizer.len() + 1)?;
                let blank = tokenizer.len();
                let mut ids = Vec::new();
                let mut frames = Vec::new();
                for (t, &id) in labels.iter().enumerate() {
                    if id != blank && (t == 0 || id != labels[t - 1]) {
                        ids.push(id);
                        frames.push(t);
                    }
                }
                (ids, frames)
            }
            Network::Rnnt(m) => m.infer_ids(&features, &self.device)?,
            Network::Ssl(_) => bail!("SSL models do not produce transcripts"),
        };
        let text = tokenizer.decode(&ids)?;
        Ok(Transcription {
            text,
            token_ids: ids,
            frames,
        })
    }

    pub fn transcribe_long_16k(&self, samples: &[f32]) -> Result<Vec<Transcription>> {
        speech_chunks(samples, (SAMPLE_RATE as usize) * 24)
            .iter()
            .map(|chunk| self.transcribe_16k(chunk))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_name_selects_correct_architecture() {
        assert_eq!(ModelKind::from_model_name("v3_ctc"), ModelKind::Ctc);
        assert_eq!(
            ModelKind::from_model_name("v3_e2e_rnnt"),
            ModelKind::E2eRnnt
        );
        assert_eq!(
            ModelKind::from_model_name("multilingual_large_ssl"),
            ModelKind::Ssl
        );
    }
}
