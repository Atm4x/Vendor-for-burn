pub mod audio;
pub mod checkpoint_config;
pub mod decode;
pub mod model;
pub mod rnnt;
pub mod text;
pub mod tokenizer;
pub mod weights;

use anyhow::{Context, Result, bail, ensure};
use audio::{
    MelConfig, SAMPLE_RATE, SpeechRegion, VadGrouping, log_mel_with_config, speech_chunks,
    speech_chunks_grouped_vad, speech_chunks_in_regions, speech_chunks_with_offsets,
};
use burn::tensor::Device;
use decode::{Transcription, Word, ctc_argmax, words_from_frames};
use model::{Config, CtcModel, EmotionModel, EncoderModel};
use rnnt::{RnntConfig, RnntModel};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizer::Tokenizer;
use weights::Weights;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
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
    /// Python's load_model accepts short aliases for the v3 family.
    pub fn canonical_model_name(name: &str) -> &str {
        match name {
            "ctc" => "v3_ctc",
            "rnnt" => "v3_rnnt",
            "e2e_ctc" => "v3_e2e_ctc",
            "e2e_rnnt" => "v3_e2e_rnnt",
            "ssl" => "v3_ssl",
            other => other,
        }
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

#[derive(Deserialize, Serialize)]
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
    #[serde(default)]
    mel: MelConfig,
}

impl ModelManifest {
    fn from_checkpoint(path: &Path) -> Result<Self> {
        let metadata = checkpoint_config::read_checkpoint_metadata(path).with_context(|| {
            format!(
                "extracting native GigaAM configuration from {}",
                path.display()
            )
        })?;
        Self::from_upstream_values(metadata.config, path, metadata.base_model_name.as_deref())
    }

    fn from_upstream_values(
        cfg: serde_json::Value,
        path: &Path,
        base_model_name: Option<&str>,
    ) -> Result<Self> {
        let encoder = cfg
            .get("encoder")
            .context("checkpoint cfg.encoder is missing")?;
        let pre = cfg
            .get("preprocessor")
            .context("checkpoint cfg.preprocessor is missing")?;
        let kind_name = cfg
            .get("model_name")
            .and_then(|v| v.as_str())
            .or(base_model_name)
            .or_else(|| path.file_stem().and_then(|name| name.to_str()))
            .context("checkpoint has no model identity")?;
        let model_name = ModelKind::canonical_model_name(kind_name);
        ensure!(
            ModelKind::from_model_name(model_name).is_some(),
            "cannot infer architecture for checkpoint model {kind_name:?}; provide config.json"
        );
        let kind = ModelKind::from_model_name(model_name).unwrap();

        let field = |root: &serde_json::Value, key: &str, fallback: usize| -> Result<usize> {
            root.get(key).map_or(Ok(fallback), |v| {
                v.as_u64()
                    .and_then(|x| usize::try_from(x).ok())
                    .with_context(|| format!("invalid nonnegative integer for {key}"))
            })
        };
        let string = |root: &serde_json::Value, key: &str, fallback: &str| -> Result<String> {
            root.get(key).map_or(Ok(fallback.to_owned()), |v| {
                v.as_str()
                    .map(str::to_owned)
                    .with_context(|| format!("invalid string for {key}"))
            })
        };
        let sample_rate = field(pre, "sample_rate", 16000)?;
        ensure!(
            sample_rate == audio::SAMPLE_RATE as usize,
            "preprocessor sample_rate={sample_rate} not supported (only 16000 Hz)"
        );
        let mel = MelConfig {
            n_fft: field(pre, "n_fft", 400)?,
            hop_length: field(pre, "hop_length", 160)?,
            win_length: field(pre, "win_length", 400)?,
            center: pre.get("center").map_or(Ok(true), |v| {
                v.as_bool().context("preprocessor.center must be a boolean")
            })?,
        };
        mel.validate()?;
        ensure!(
            field(pre, "pad", 0)? == 0,
            "preprocessor.pad is not supported"
        );
        for (name, expected) in [("f_min", 0.0), ("f_max", 8000.0), ("power", 2.0)] {
            if let Some(value) = pre.get(name) {
                ensure!(
                    (value.is_null() && name == "f_max") || value.as_f64() == Some(expected),
                    "preprocessor.{name} is not supported by native log-mel"
                );
            }
        }
        for name in ["norm", "mel_norm"] {
            ensure!(
                pre.get(name).is_none_or(serde_json::Value::is_null),
                "preprocessor.{name} not supported by native log-mel"
            );
        }
        ensure!(
            pre.get("mel_scale")
                .is_none_or(|v| v.as_str() == Some("htk")),
            "preprocessor.mel_scale must be HTK"
        );
        if let Some(target) = pre.get("_target_").and_then(|v| v.as_str()) {
            ensure!(
                target == "gigaam.preprocess.FeatureExtractor",
                "unsupported preprocessor target: {target}"
            );
        }
        if let Some(target) = encoder.get("_target_").and_then(|v| v.as_str()) {
            ensure!(
                target == "gigaam.encoder.ConformerEncoder",
                "unsupported encoder target: {target}"
            );
        }
        let n_mels = field(pre, "features", field(encoder, "feat_in", 64)?)?;
        ensure!(
            n_mels == field(encoder, "feat_in", 64)?,
            "preprocessor.features and encoder.feat_in differ"
        );
        let decoding = cfg.get("decoding").unwrap_or(&serde_json::Value::Null);
        let head = cfg.get("head").unwrap_or(&serde_json::Value::Null);
        let id2name = cfg.get("id2name").filter(|v| !v.is_null()).cloned();
        let classes = if kind.is_emo() {
            let names = id2name
                .as_ref()
                .context("emotion checkpoint lacks id2name")?;
            match names {
                serde_json::Value::Object(map) => map.len(),
                serde_json::Value::Array(a) => a.len(),
                _ => bail!("invalid emotion labels"),
            }
        } else if kind.is_ssl() {
            2
        } else if kind.is_rnnt() {
            field(&head["joint"], "num_classes", 0)?
        } else {
            field(head, "num_classes", 0)?
        };
        ensure!(classes > 1, "checkpoint does not specify num_classes");
        let config = Config {
            n_mels,
            n_layers: field(encoder, "n_layers", 16)?,
            d_model: field(encoder, "d_model", 768)?,
            n_heads: field(encoder, "n_heads", 16)?,
            ff_expansion: field(encoder, "ff_expansion_factor", 4)?,
            conv_kernel: field(encoder, "conv_kernel_size", 31)?,
            subsample_kernel: field(encoder, "subs_kernel_size", 3)?,
            subsample_channels: field(encoder, "d_model", 768)?,
            num_classes: classes,
            self_attention_model: string(encoder, "self_attention_model", "rotary")?,
            conv_norm_type: string(encoder, "conv_norm_type", "batch_norm")?,
            subsampling: string(encoder, "subsampling", "conv2d")?,
            subsampling_factor: field(encoder, "subsampling_factor", 4)?,
            pos_emb_max_len: field(encoder, "pos_emb_max_len", 5000)?,
        };
        config.validate()?;
        let rnnt = if kind.is_rnnt() {
            let decoder = head.get("decoder").context("RNN-T head.decoder missing")?;
            let joint = head.get("joint").context("RNN-T head.joint missing")?;
            let rnnt = RnntConfig {
                pred_hidden: field(decoder, "pred_hidden", 0)?,
                pred_rnn_layers: field(decoder, "pred_rnn_layers", 0)?,
                joint_hidden: field(joint, "joint_hidden", 0)?,
                max_symbols_per_step: field(decoding, "max_symbols_per_step", 10)?,
            };
            ensure!(
                field(decoder, "num_classes", classes)? == classes,
                "RNN-T decoder and joint num_classes disagree"
            );
            ensure!(
                field(joint, "enc_hidden", config.d_model)? == config.d_model,
                "RNN-T joint enc_hidden disagrees with encoder"
            );
            ensure!(
                field(joint, "pred_hidden", rnnt.pred_hidden)? == rnnt.pred_hidden,
                "RNN-T joint pred_hidden disagrees with predictor"
            );
            rnnt.validate()?;
            Some(rnnt)
        } else {
            None
        };
        let vocab =
            decoding
                .get("vocabulary")
                .filter(|v| !v.is_null())
                .map_or(Ok(Vec::new()), |v| {
                    serde_json::from_value::<Vec<String>>(v.clone())
                        .context("invalid decoding.vocabulary")
                })?;
        let sp_name = decoding
            .get("model_path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .and_then(|s| Path::new(s).file_name().and_then(|s| s.to_str()))
            .filter(|basename| {
                path.parent()
                    .is_some_and(|dir| dir.join(basename).is_file())
            })
            .map(str::to_owned);
        let id2name = id2name
            .map(serde_json::from_value)
            .transpose()
            .context("invalid emotion id2name")?;
        Ok(Self {
            config,
            vocabulary: vocab,
            model_name: Some(model_name.to_owned()),
            model_type: Some(kind),
            sentencepiece_model: sp_name,
            rnnt,
            id2name,
            mel,
        })
    }
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

#[derive(Debug, Deserialize, Serialize)]
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
    mel: MelConfig,
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

impl TimedTranscription {
    fn with_sample_offset(mut self, start_sample: usize) -> Self {
        let offset = start_sample as f64 / SAMPLE_RATE as f64;
        self.start = offset;
        self.end += offset;
        for word in &mut self.words {
            word.start = ((word.start + offset) * 1000.0).round_ties_even() / 1000.0;
            word.end = ((word.end + offset) * 1000.0).round_ties_even() / 1000.0;
        }
        self
    }
}

/// Build a native inference manifest from an original PyTorch checkpoint's
/// OmegaConf metadata without launching Python or writing a sidecar file.
pub fn inspect_native_manifest(path: &Path) -> Result<serde_json::Value> {
    serde_json::to_value(ModelManifest::from_checkpoint(path)?).map_err(Into::into)
}

impl GigaAm {
    /// Load from a model directory or a specific checkpoint file. A sibling
    /// `config.json` supplies the architecture and tokenizer configuration.
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let direct_checkpoint = match model_dir.extension().and_then(|ext| ext.to_str()) {
            Some("ckpt" | "pth" | "pt" | "safetensors") => {
                ensure!(
                    model_dir.is_file(),
                    "checkpoint not found: {}",
                    model_dir.display()
                );
                Some(model_dir)
            }
            _ => None,
        };
        let model_root = match direct_checkpoint {
            Some(path) => path
                .parent()
                .context("checkpoint has no parent directory")?,
            None => model_dir,
        };
        let cfg_path = model_root.join("config.json");
        let inferred_checkpoint = if cfg_path.is_file() || direct_checkpoint.is_some() {
            direct_checkpoint.map(Path::to_path_buf)
        } else {
            let available = std::fs::read_dir(model_root)
                .with_context(|| format!("scanning {}", model_root.display()))?
                .filter_map(|entry| entry.ok().map(|x| x.path()))
                .filter(|path| {
                    path.is_file()
                        && matches!(
                            path.extension().and_then(|ext| ext.to_str()),
                            Some("ckpt" | "pt" | "pth")
                        )
                })
                .collect::<Vec<_>>();
            ensure!(
                available.len() == 1,
                "without config.json the model directory must contain exactly one PyTorch checkpoint; found {}",
                available.len()
            );
            available.into_iter().next()
        };
        let manifest: ModelManifest = if cfg_path.is_file() {
            serde_json::from_slice(
                &std::fs::read(&cfg_path)
                    .with_context(|| format!("reading {}", cfg_path.display()))?,
            )?
        } else {
            let path=inferred_checkpoint.as_deref().context("automatic configuration requires a PyTorch .ckpt/.pt/.pth file; use config.json with Safetensors")?;
            ModelManifest::from_checkpoint(path)?
        };
        manifest.config.validate()?;
        manifest.mel.validate()?;
        let mel = manifest.mel;
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
        let canonical_name = manifest
            .model_name
            .as_deref()
            .map(ModelKind::canonical_model_name);
        let tokenizer = if kind.is_ssl() || kind.is_emo() {
            None
        } else {
            let sp_path = if let Some(path) = manifest.sentencepiece_model {
                ensure!(
                    !path.contains('/') && !path.contains('\\') && !path.contains(".."),
                    "sentencepiece_model must be a filename in model_dir"
                );
                Some(model_root.join(path))
            } else if matches!(kind, ModelKind::E2eCtc | ModelKind::E2eRnnt)
                || canonical_name == Some("v1_rnnt")
            {
                Some(model_root.join(match canonical_name {
                    Some("v1_rnnt") => "v1_rnnt_tokenizer.model",
                    Some("v3_e2e_ctc") => "v3_e2e_ctc_tokenizer.model",
                    Some("v3_e2e_rnnt") => "v3_e2e_rnnt_tokenizer.model",
                    _ => {
                        anyhow::bail!("custom E2E model requires sentencepiece_model filename")
                    }
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
            let canonical = ModelKind::canonical_model_name(name);
            weight_names.push(format!("{canonical}.ckpt"));
            weight_names.push(format!("{canonical}.safetensors"));
        }
        weight_names.extend(["model.ckpt".to_owned(), "model.safetensors".to_owned()]);
        if kind == ModelKind::Ctc {
            weight_names.extend(["v3_ctc.ckpt".to_owned(), "v3_ctc.safetensors".to_owned()]);
        }
        let weights_path = match direct_checkpoint {
            Some(path) => path.to_owned(),
            None if !cfg_path.is_file() => {
                inferred_checkpoint.context("checkpoint required for auto configuration")?
            }
            None => weight_names
                .iter()
                .map(|n| model_root.join(n))
                .find(|p| p.is_file())
                .context("GigaAM model weights not found")?,
        };
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
            mel,
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
        ensure!(!pcm.is_empty(), "emotion inference requires nonempty audio");
        let model = match &self.network {
            Network::Emo(model) => model,
            _ => bail!("emotion_probs_16k requires an emotion model"),
        };
        let features = log_mel_with_config(pcm, self.n_mels, self.mel)?;
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

    pub fn extract_features_16k(&self, pcm: &[f32]) -> Result<audio::Features> {
        log_mel_with_config(pcm, self.n_mels, self.mel)
    }

    pub fn mel_config(&self) -> MelConfig {
        self.mel
    }

    pub fn infer_logits_16k(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let features = log_mel_with_config(pcm, self.n_mels, self.mel)?;
        match &self.network {
            Network::Ctc(m) => m.infer_logits(&features, &self.device),
            _ => bail!("infer_logits_16k is only available for CTC variants"),
        }
    }

    pub fn infer_embeddings_16k(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        let features = log_mel_with_config(pcm, self.n_mels, self.mel)?;
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
        ensure!(!pcm.is_empty(), "transcription requires nonempty audio");
        let tokenizer = self
            .tokenizer
            .as_ref()
            .context("non-ASR models do not produce transcripts")?;
        let features = log_mel_with_config(pcm, self.n_mels, self.mel)?;
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

    /// Ordered inference for multiple recordings. Each sample is executed
    /// independently to avoid padding-dependent output changes in Conformer.
    /// This is a functional batch API, not GPU-batched execution.
    pub fn transcribe_batch_16k<T: AsRef<[f32]>>(
        &self,
        recordings: &[T],
    ) -> Result<Vec<Transcription>> {
        ensure!(
            self.tokenizer.is_some(),
            "batch transcription requires an ASR model"
        );
        recordings
            .iter()
            .enumerate()
            .map(|(index, pcm)| {
                self.transcribe_16k(pcm.as_ref())
                    .with_context(|| format!("transcribing batch element {index}"))
            })
            .collect()
    }

    /// Ordered batch transcription with word timestamps relative to each sample.
    pub fn transcribe_batch_with_timestamps_16k<T: AsRef<[f32]>>(
        &self,
        recordings: &[T],
    ) -> Result<Vec<TimedTranscription>> {
        ensure!(
            self.tokenizer.is_some(),
            "batch transcription requires an ASR model"
        );
        recordings
            .iter()
            .enumerate()
            .map(|(index, pcm)| {
                self.transcribe_with_timestamps_16k(pcm.as_ref())
                    .with_context(|| format!("transcribing timed batch element {index}"))
            })
            .collect()
    }

    /// Ordered batch encoder outputs as (row-major frames, time, channels).
    pub fn infer_embeddings_batch_16k<T: AsRef<[f32]>>(
        &self,
        recordings: &[T],
    ) -> Result<Vec<(Vec<f32>, usize, usize)>> {
        ensure!(self.kind.is_ssl(), "batch embeddings require an SSL model");
        recordings
            .iter()
            .enumerate()
            .map(|(index, pcm)| {
                self.infer_embeddings_16k(pcm.as_ref())
                    .with_context(|| format!("embedding batch element {index}"))
            })
            .collect()
    }

    /// Ordered utterance-level emotion probability distributions.
    pub fn emotion_probs_batch_16k<T: AsRef<[f32]>>(
        &self,
        recordings: &[T],
    ) -> Result<Vec<Vec<(String, f32)>>> {
        ensure!(
            self.kind.is_emo(),
            "batch emotions require an emotion model"
        );
        recordings
            .iter()
            .enumerate()
            .map(|(index, pcm)| {
                self.emotion_probs_16k(pcm.as_ref())
                    .with_context(|| format!("classifying batch element {index}"))
            })
            .collect()
    }

    fn check_short_transcription(pcm: &[f32]) -> Result<()> {
        ensure!(
            !pcm.is_empty() && pcm.len() <= SAMPLE_RATE as usize * 25,
            "transcription requires 0 < audio duration <= 25s"
        );
        Ok(())
    }

    pub fn transcribe_16k(&self, pcm: &[f32]) -> Result<Transcription> {
        Self::check_short_transcription(pcm)?;
        Ok(self.decode_with_length_16k(pcm)?.0)
    }

    pub fn transcribe_with_timestamps_16k(&self, pcm: &[f32]) -> Result<TimedTranscription> {
        Self::check_short_transcription(pcm)?;
        self.transcribe_segment_with_timestamps_16k(pcm)
    }

    fn transcribe_segment_with_timestamps_16k(&self, pcm: &[f32]) -> Result<TimedTranscription> {
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

    /// Transcribe VAD-provided half-open intervals (16 kHz sample indices),
    /// preserving their absolute position in the source recording.
    /// Regions must be ordered, non-overlapping and within PCM boundaries.
    /// The model itself does not instantiate or duplicate a VAD engine.
    pub fn transcribe_regions_with_timestamps_16k(
        &self,
        samples: &[f32],
        regions: &[SpeechRegion],
    ) -> Result<Vec<TimedTranscription>> {
        let chunks = speech_chunks_in_regions(samples, regions, SAMPLE_RATE as usize * 24)?;
        let mut output = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            output.push(
                self.transcribe_with_timestamps_16k(chunk.samples)?
                    .with_sample_offset(chunk.start_sample),
            );
        }
        Ok(output)
    }

    /// Match the Python `transcribe_longform` VAD grouping after receiving
    /// precomputed speech regions from the application's VAD service.
    /// Longform runs may contain chunks up to 30 seconds, unlike short ASR.
    pub fn transcribe_grouped_vad_with_timestamps_16k(
        &self,
        samples: &[f32],
        regions: &[SpeechRegion],
        options: VadGrouping,
    ) -> Result<Vec<TimedTranscription>> {
        speech_chunks_grouped_vad(samples, regions, options)?
            .iter()
            .map(|chunk| {
                self.transcribe_segment_with_timestamps_16k(chunk.samples)
                    .map(|result| result.with_sample_offset(chunk.start_sample))
            })
            .collect()
    }

    pub fn transcribe_grouped_vad_16k(
        &self,
        samples: &[f32],
        regions: &[SpeechRegion],
        options: VadGrouping,
    ) -> Result<Vec<Transcription>> {
        speech_chunks_grouped_vad(samples, regions, options)?
            .iter()
            .map(|chunk| {
                self.decode_with_length_16k(chunk.samples)
                    .map(|(text, _)| text)
            })
            .collect()
    }

    /// Transcribe VAD-provided intervals without the overhead of word alignment.
    pub fn transcribe_regions_16k(
        &self,
        samples: &[f32],
        regions: &[SpeechRegion],
    ) -> Result<Vec<Transcription>> {
        speech_chunks_in_regions(samples, regions, SAMPLE_RATE as usize * 24)?
            .iter()
            .map(|chunk| self.transcribe_16k(chunk.samples))
            .collect()
    }

    pub fn transcribe_long_with_timestamps_16k(
        &self,
        samples: &[f32],
    ) -> Result<Vec<TimedTranscription>> {
        let mut segments = Vec::new();
        for chunk in speech_chunks_with_offsets(samples, SAMPLE_RATE as usize * 24) {
            segments.push(
                self.transcribe_with_timestamps_16k(chunk.samples)?
                    .with_sample_offset(chunk.start_sample),
            );
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
    fn longform_word_timestamps_use_absolute_millisecond_grid() {
        let local = TimedTranscription {
            transcription: Transcription {
                text: "слово".to_owned(),
                token_ids: vec![0],
                frames: vec![1],
            },
            words: vec![Word {
                text: "слово".to_owned(),
                start: 0.10137,
                end: 0.48976,
            }],
            start: 0.0,
            end: 0.5,
        };
        let shifted = local.with_sample_offset(20_805);
        assert!((shifted.start - 1.3003125).abs() < 1e-12);
        assert!((shifted.end - 1.8003125).abs() < 1e-12);
        assert!((shifted.words[0].start - 1.402).abs() < 1e-12);
        assert!((shifted.words[0].end - 1.790).abs() < 1e-12);
    }
    #[test]
    fn python_short_model_names_resolve_to_v3_artifacts() {
        for (alias, canonical, kind) in [
            ("ctc", "v3_ctc", ModelKind::Ctc),
            ("rnnt", "v3_rnnt", ModelKind::Rnnt),
            ("e2e_ctc", "v3_e2e_ctc", ModelKind::E2eCtc),
            ("e2e_rnnt", "v3_e2e_rnnt", ModelKind::E2eRnnt),
            ("ssl", "v3_ssl", ModelKind::Ssl),
        ] {
            assert_eq!(ModelKind::canonical_model_name(alias), canonical);
            assert_eq!(ModelKind::from_model_name(alias), Some(kind));
            assert_eq!(ModelKind::canonical_model_name(canonical), canonical);
        }
        assert_eq!(ModelKind::canonical_model_name("v1_rnnt"), "v1_rnnt");
        assert_eq!(ModelKind::canonical_model_name("custom"), "custom");
    }

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
            mel: MelConfig::default(),
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
    #[test]
    fn omegaconf_checkpoint_maps_all_required_conformer_fields() {
        let original = serde_json::json!({
            "model_name":"v3_ctc",
            "preprocessor":{"sample_rate":16000,"features":64,"center":true},
            "encoder":{"n_layers":22,"d_model":512,"n_heads":8,"feat_in":64,
                "ff_expansion_factor":6,"conv_kernel_size":15,"subs_kernel_size":5,
                "subsampling":"conv1d","subsampling_factor":8,"self_attention_model":"rel_pos",
                "conv_norm_type":"layer_norm","pos_emb_max_len":1024},
            "head":{"num_classes":34},
            "decoding":{"vocabulary":(0..33).map(|n|n.to_string()).collect::<Vec<_>>()}
        });
        let path = Path::new("v3_ctc.ckpt");
        let manifest = ModelManifest::from_upstream_values(original, path, None).unwrap();
        assert_eq!(manifest.mel, MelConfig::default());
        assert_eq!(manifest.config.n_layers, 22);
        assert_eq!(manifest.config.ff_expansion, 6);
        assert_eq!(manifest.config.conv_kernel, 15);
        assert_eq!(manifest.config.subsample_kernel, 5);
        assert_eq!(manifest.config.subsampling_factor, 8);
        assert_eq!(manifest.config.self_attention_model, "rel_pos");
        assert_eq!(manifest.config.conv_norm_type, "layer_norm");
        assert_eq!(manifest.config.pos_emb_max_len, 1024);
        assert_eq!(manifest.resolve_kind().unwrap(), ModelKind::Ctc);
    }

    #[test]
    fn omegaconf_checkpoint_maps_rnnt_and_e2e() {
        let original = serde_json::json!({
            "model_name":"v3_e2e_rnnt",
            "preprocessor":{"sample_rate":16000,"features":8},
            "encoder":{"n_layers":1,"d_model":8,"n_heads":2,"feat_in":8},
            "head":{"decoder":{"pred_hidden":16,"pred_rnn_layers":2,"num_classes":128},
                    "joint":{"enc_hidden":8,"pred_hidden":16,"joint_hidden":12,"num_classes":128}},
            "decoding":{"max_symbols_per_step":6,"model_path":"/unused/old/path/v3_e2e_rnnt_tokenizer.model"}
        });
        let model =
            ModelManifest::from_upstream_values(original, Path::new("v3_e2e_rnnt.ckpt"), None)
                .unwrap();
        assert_eq!(model.resolve_kind().unwrap(), ModelKind::E2eRnnt);
        assert_eq!(model.config.num_classes, 128);
        let rnnt = model.rnnt.unwrap();
        assert_eq!(rnnt.pred_hidden, 16);
        assert_eq!(rnnt.pred_rnn_layers, 2);
        assert_eq!(rnnt.joint_hidden, 12);
        assert_eq!(rnnt.max_symbols_per_step, 6);
        assert!(model.sentencepiece_model.is_none());
    }

    #[test]
    fn config_extraction_rejects_nonmatching_mel_audio_contract() {
        let base = serde_json::json!({"model_name":"v3_ssl","preprocessor":{"sample_rate":16000,"features":8},
            "encoder":{"feat_in":8,"d_model":8,"n_heads":2,"n_layers":1}});
        let mut bad = base.clone();
        bad["preprocessor"]["hop_length"] = serde_json::json!(256);
        assert_eq!(
            ModelManifest::from_upstream_values(bad, Path::new("v3_ssl.ckpt"), None)
                .unwrap()
                .mel
                .hop_length,
            256
        );
        let mut bad = base.clone();
        bad["preprocessor"]["n_fft"] = serde_json::json!(128);
        assert!(ModelManifest::from_upstream_values(bad, Path::new("v3_ssl.ckpt"), None).is_err());
        let mut official = base.clone();
        official["preprocessor"]["n_fft"] = serde_json::json!(320);
        official["preprocessor"]["win_length"] = serde_json::json!(320);
        official["preprocessor"]["center"] = serde_json::json!(false);
        let parsed =
            ModelManifest::from_upstream_values(official, Path::new("v3_ssl.ckpt"), None).unwrap();
        assert_eq!(
            parsed.mel,
            MelConfig {
                n_fft: 320,
                win_length: 320,
                hop_length: 160,
                center: false
            }
        );
        let mut bad = base.clone();
        bad["encoder"]["feat_in"] = serde_json::json!(16);
        assert!(ModelManifest::from_upstream_values(bad, Path::new("v3_ssl.ckpt"), None).is_err());
        assert!(ModelManifest::from_upstream_values(base, Path::new("v3_ssl.ckpt"), None).is_ok());
    }
}
