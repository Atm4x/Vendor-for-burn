pub mod audio;
pub mod decode;
pub mod model;
pub mod rnnt;
pub mod tokenizer;
pub mod weights;

use anyhow::{Context, Result, bail, ensure};
use audio::{SAMPLE_RATE, log_mel, speech_chunks, speech_chunks_with_offsets};
use burn::tensor::Device;
use decode::{Transcription, Word, ctc_argmax, words_from_frames};
use model::{Config, CtcModel, EmotionModel, EncoderModel};
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
    Emo,
}
impl ModelKind {
    pub fn is_ctc(self) -> bool {
        matches!(self, Self::Ctc | Self::E2eCtc)
    }
    pub fn is_rnnt(self) -> bool {
        matches!(self, Self::Rnnt | Self::E2eRnnt)
    }
    pub fn is_emo(self) -> bool {
        self == Self::Emo
    }
    pub fn is_ssl(self) -> bool {
        self == Self::Ssl
    }
    pub fn from_model_name(name: &str) -> Option<Self> {
        const MODELS: &[(&str, ModelKind)] = &[
            ("ctc", ModelKind::Ctc),
            ("rnnt", ModelKind::Rnnt),
            ("e2e_ctc", ModelKind::E2eCtc),
            ("e2e_rnnt", ModelKind::E2eRnnt),
            ("ssl", ModelKind::Ssl),
            ("emo", ModelKind::Emo),
            ("v1_ctc", ModelKind::Ctc),
            ("v1_rnnt", ModelKind::Rnnt),
            ("v1_ssl", ModelKind::Ssl),
            ("v2_ctc", ModelKind::Ctc),
            ("v2_rnnt", ModelKind::Rnnt),
            ("v2_ssl", ModelKind::Ssl),
            ("v3_ctc", ModelKind::Ctc),
            ("v3_rnnt", ModelKind::Rnnt),
            ("v3_e2e_ctc", ModelKind::E2eCtc),
            ("v3_e2e_rnnt", ModelKind::E2eRnnt),
            ("v3_ssl", ModelKind::Ssl),
            ("multilingual_ctc", ModelKind::Ctc),
            ("multilingual_ssl", ModelKind::Ssl),
            ("multilingual_large_ctc", ModelKind::Ctc),
            ("multilingual_large_ssl", ModelKind::Ssl),
        ];
        MODELS
            .iter()
            .find(|(model_name, _)| *model_name == name)
            .map(|(_, kind)| *kind)
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
    #[serde(default)]
    id2name: Option<EmotionLabels>,
}

impl ModelManifest {
    fn resolve_kind(&self) -> Result<ModelKind> {
        match (self.model_type, self.model_name.as_deref()) {
            (Some(explicit), Some(name)) => {
                if let Some(registered) = ModelKind::from_model_name(name) {
                    ensure!(
                        explicit == registered,
                        "model_type {explicit:?} conflicts with registered model {name} ({registered:?})"
                    );
                }
                Ok(explicit)
            }
            (Some(explicit), None) => Ok(explicit),
            (None, Some(name)) => ModelKind::from_model_name(name).with_context(|| {
                format!(
                    "unknown model_name {name:?}: specify model_type explicitly for custom models"
                )
            }),
            (None, None) => bail!("config.json must specify model_name or model_type"),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmotionLabels {
    List(Vec<String>),
    Map(std::collections::BTreeMap<String, String>),
}

impl EmotionLabels {
    fn into_names(self) -> Result<Vec<String>> {
        let values = match self {
            Self::List(values) => values,
            Self::Map(values) => {
                let mut indexed = values
                    .into_iter()
                    .map(|(index, name)| Ok((index.parse::<usize>()?, name)))
                    .collect::<Result<Vec<_>>>()?;
                indexed.sort_by_key(|(index, _)| *index);
                for (expected, (index, _)) in indexed.iter().enumerate() {
                    ensure!(
                        *index == expected,
                        "emotion label IDs must be consecutive from zero"
                    );
                }
                indexed.into_iter().map(|(_, name)| name).collect()
            }
        };
        ensure!(values.len() >= 2, "at least two emotion labels required");
        ensure!(values.iter().all(|s| !s.is_empty()), "empty emotion label");
        Ok(values)
    }
}

enum Network {
    Ctc(CtcModel),
    Rnnt(RnntModel),
    Ssl(EncoderModel),
    Emo(EmotionModel),
}

pub struct GigaAm {
    network: Network,
    tokenizer: Option<Tokenizer>,
    device: Device,
    n_mels: usize,
    kind: ModelKind,
    emotion_names: Vec<String>,
}

pub type GigaAmCtc = GigaAm;

#[derive(Debug, Clone, PartialEq)]
pub struct TimedTranscription {
    pub transcription: Transcription,
    pub words: Vec<Word>,
    pub start: f64,
    pub end: f64,
}

impl GigaAm {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let cfg_path = model_dir.join("config.json");
        let manifest: ModelManifest = serde_json::from_slice(
            &std::fs::read(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?,
        )?;
        manifest.config.validate()?;
        let kind = manifest.resolve_kind()?;
        let num_classes = manifest.config.num_classes;
        let n_mels = manifest.config.n_mels;
        let emotion_names = if kind.is_emo() {
            manifest
                .id2name
                .context("emotion model requires id2name labels")?
                .into_names()?
        } else {
            Vec::new()
        };
        let tokenizer = if kind.is_ssl() || kind.is_emo() {
            None
        } else {
            let sp_path = if let Some(path) = manifest.sentencepiece_model {
                ensure!(
                    !path.contains('/') && !path.contains('\\') && !path.contains(".."),
                    "sentencepiece_model must be a filename in model_dir"
                );
                Some(model_dir.join(path))
            } else if matches!(kind, ModelKind::E2eCtc | ModelKind::E2eRnnt)
                || manifest.model_name.as_deref() == Some("v1_rnnt")
            {
                Some(model_dir.join(match manifest.model_name.as_deref() {
                    Some("v1_rnnt") => "v1_rnnt_tokenizer.model",
                    Some("v3_e2e_ctc") => "v3_e2e_ctc_tokenizer.model",
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
        let network = match kind {
            ModelKind::Emo => Network::Emo(EmotionModel::load(
                &mut weights,
                manifest.config.clone(),
                emotion_names.len(),
                &device,
            )?),
            ModelKind::Ctc | ModelKind::E2eCtc => Network::Ctc(CtcModel::load(
                &mut weights,
                manifest.config.clone(),
                &device,
            )?),
            ModelKind::Rnnt | ModelKind::E2eRnnt => {
                let rnnt = manifest.rnnt.context(
                    "RNN-T requires rnnt config (pred_hidden, pred_rnn_layers, joint_hidden)",
                )?;
                rnnt.validate()?;
                Network::Rnnt(RnntModel::load(
                    &mut weights,
                    manifest.config.clone(),
                    rnnt,
                    &device,
                )?)
            }
            ModelKind::Ssl => Network::Ssl(EncoderModel::load(
                &mut weights,
                manifest.config.clone(),
                &device,
            )?),
        };
        Ok(Self {
            network,
            tokenizer,
            device,
            n_mels,
            kind,
            emotion_names,
        })
    }

    pub fn model_kind(&self) -> ModelKind {
        self.kind
    }
    pub fn emotion_labels(&self) -> &[String] {
        &self.emotion_names
    }

    pub fn emotion_probs_16k(&self, pcm: &[f32]) -> Result<Vec<(String, f32)>> {
        ensure!(
            !pcm.is_empty() && pcm.len() <= SAMPLE_RATE as usize * 25,
            "emotion inference requires 0 < audio duration <= 25s"
        );
        let model = match &self.network {
            Network::Emo(model) => model,
            _ => bail!("emotion_probs_16k requires an emotion model"),
        };
        let features = log_mel(pcm, self.n_mels)?;
        let values = model.infer_probs(&features, &self.device)?;
        ensure!(
            values.len() == self.emotion_names.len(),
            "emotion label count mismatch"
        );
        Ok(self.emotion_names.iter().cloned().zip(values).collect())
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

    fn decode_with_length_16k(&self, pcm: &[f32]) -> Result<(Transcription, usize)> {
        ensure!(
            !pcm.is_empty() && pcm.len() <= SAMPLE_RATE as usize * 25,
            "transcription requires 0 < audio duration <= 25s"
        );
        let tokenizer = self
            .tokenizer
            .as_ref()
            .context("non-ASR models do not produce transcripts")?;
        let features = log_mel(pcm, self.n_mels)?;
        let (ids, frames, encoded_len) = match &self.network {
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
                (ids, frames, time)
            }
            Network::Rnnt(m) => m.infer_ids_with_length(&features, &self.device)?,
            Network::Ssl(_) | Network::Emo(_) => bail!("non-ASR models do not produce transcripts"),
        };
        let text = tokenizer.decode(&ids)?;
        Ok((
            Transcription {
                text,
                token_ids: ids,
                frames,
            },
            encoded_len,
        ))
    }

    pub fn transcribe_16k(&self, pcm: &[f32]) -> Result<Transcription> {
        Ok(self.decode_with_length_16k(pcm)?.0)
    }

    pub fn transcribe_with_timestamps_16k(&self, pcm: &[f32]) -> Result<TimedTranscription> {
        let (transcription, encoded_len) = self.decode_with_length_16k(pcm)?;
        let tokenizer = self.tokenizer.as_ref().context("ASR tokenizer missing")?;
        let words = words_from_frames(
            tokenizer,
            &transcription.token_ids,
            &transcription.frames,
            pcm.len(),
            encoded_len,
        )?;
        Ok(TimedTranscription {
            transcription,
            words,
            start: 0.0,
            end: pcm.len() as f64 / SAMPLE_RATE as f64,
        })
    }

    pub fn transcribe_long_with_timestamps_16k(
        &self,
        samples: &[f32],
    ) -> Result<Vec<TimedTranscription>> {
        let mut segments = Vec::new();
        for chunk in speech_chunks_with_offsets(samples, SAMPLE_RATE as usize * 24) {
            let mut timed = self.transcribe_with_timestamps_16k(chunk.samples)?;
            let offset = chunk.start_sample as f64 / SAMPLE_RATE as f64;
            timed.start = offset;
            timed.end += offset;
            for word in &mut timed.words {
                word.start += offset;
                word.end += offset;
            }
            segments.push(timed);
        }
        Ok(segments)
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
        assert_eq!(ModelKind::from_model_name("v3_ctc"), Some(ModelKind::Ctc));
        assert_eq!(ModelKind::from_model_name("emo"), Some(ModelKind::Emo));
        assert_eq!(
            ModelKind::from_model_name("v3_e2e_rnnt"),
            Some(ModelKind::E2eRnnt)
        );
        assert_eq!(
            ModelKind::from_model_name("multilingual_large_ssl"),
            Some(ModelKind::Ssl)
        );
    }
    #[test]
    fn model_selection_rejects_unknown_names_and_conflicting_types() {
        assert_eq!(ModelKind::from_model_name("unknown_ctc"), None);
        assert_eq!(ModelKind::from_model_name("v3_e2e_rnnt_typo"), None);
        let cfg = |name: Option<&str>, model_type: Option<ModelKind>| ModelManifest {
            config: Config::default(),
            vocabulary: Vec::new(),
            model_name: name.map(str::to_owned),
            model_type,
            sentencepiece_model: None,
            rnnt: None,
            id2name: None,
        };
        assert!(
            cfg(Some("v3_rnnt"), Some(ModelKind::Ctc))
                .resolve_kind()
                .is_err()
        );
        assert!(cfg(Some("unknown_ctc"), None).resolve_kind().is_err());
        assert!(cfg(None, None).resolve_kind().is_err());
        assert_eq!(
            cfg(Some("custom_encoder"), Some(ModelKind::Ssl))
                .resolve_kind()
                .unwrap(),
            ModelKind::Ssl
        );
    }
    #[test]
    fn emotion_names_preserve_numeric_order() {
        let names =
            serde_json::from_str::<EmotionLabels>(r#"{"10":"extra", "0":"a", "1":"b"}"#).unwrap();
        assert!(names.into_names().is_err());
        let names =
            serde_json::from_str::<EmotionLabels>(r#"{"2":"c", "0":"a", "1":"b"}"#).unwrap();
        assert_eq!(names.into_names().unwrap(), ["a", "b", "c"]);
    }
    #[test]
    fn rejects_unimplemented_encoder_variants() {
        let mut cfg = Config::default();
        cfg.self_attention_model = "unsupported".into();
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("attention")
        );
        cfg.self_attention_model = "rel_pos".into();
        cfg.subsampling = "unknown".into();
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("subsampling")
        );
        cfg.subsampling = "conv1d".into();
        cfg.conv_norm_type = "layer_norm".into();
        assert!(cfg.validate().is_ok());
    }
}
