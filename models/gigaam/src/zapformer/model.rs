use super::{
    ZapformerEncoder,
    config::{CheckpointConfig, HeadConfig, HeadsConfig, TokenizerConfig},
    language::{LanguageConditioner, LanguageTag},
};
use crate::{
    audio::{Features, log_mel_with_config},
    checkpoint_config::read_checkpoint_metadata,
    decode::{Transcription, ctc_argmax},
    model::CtcHead,
    rnnt::{RnntConfig, RnntHead},
    tokenizer::Tokenizer,
    weights::Weights,
};
use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Device, Tensor, TensorData};
use std::{path::Path, sync::Arc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadSelection {
    Raw,
    #[default]
    E2e,
}

struct AsrHead {
    tokenizer: Tokenizer,
    decoder: Decoder,
}
enum Decoder {
    Ctc(CtcHead),
    Rnnt(RnntHead),
}
enum Heads {
    EncoderOnly,
    Single(Arc<AsrHead>),
    Dual { raw: Arc<AsrHead>, e2e: Arc<AsrHead> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeadTranscriptions {
    Single(Vec<Transcription>),
    Dual {
        raw: Vec<Transcription>,
        e2e: Vec<Transcription>,
    },
}

impl AsrHead {
    fn load(
        w: &mut Weights,
        prefix: &str,
        config: &HeadConfig,
        width: usize,
        tokenizer_dir: &Path,
        device: &Device,
    ) -> Result<Self> {
        let (classes, tokenizer, decoder) = match config {
            HeadConfig::Ctc {
                num_classes,
                tokenizer,
            } => (
                *num_classes,
                tokenizer,
                Decoder::Ctc(CtcHead::load(w, prefix, width, *num_classes, device)?),
            ),
            HeadConfig::Rnnt {
                num_classes,
                pred_hidden,
                pred_rnn_layers,
                joint_hidden,
                max_symbols_per_step,
                tokenizer,
            } => (
                *num_classes,
                tokenizer,
                Decoder::Rnnt(RnntHead::load(
                    w,
                    prefix,
                    width,
                    *num_classes,
                    &RnntConfig {
                        pred_hidden: *pred_hidden,
                        pred_rnn_layers: *pred_rnn_layers,
                        joint_hidden: *joint_hidden,
                        max_symbols_per_step: *max_symbols_per_step,
                    },
                    device,
                )?),
            ),
        };
        let tokenizer = match tokenizer {
            TokenizerConfig::Vocabulary { tokens } => Tokenizer::load(tokens.clone(), None)?,
            TokenizerConfig::SentencePiece { checkpoint_path } => {
                let name = checkpoint_path
                    .rsplit(['/', '\\'])
                    .next()
                    .filter(|s| !s.is_empty() && *s != "." && *s != "..")
                    .context("invalid tokenizer filename")?;
                Tokenizer::load(Vec::new(), Some(&tokenizer_dir.join(name)))?
            }
        };
        ensure!(
            tokenizer.len() == classes - 1,
            "{prefix}: tokenizer size differs from head classes"
        );
        Ok(Self { tokenizer, decoder })
    }

    fn decode(&self, encoded: Tensor<3>, lengths: &[usize]) -> Result<Vec<Transcription>> {
        let [batch, time, width] = encoded.dims();
        ensure!(
            lengths.len() == batch && lengths.iter().all(|&n| n <= time),
            "invalid decoder lengths"
        );
        let mut results = Vec::with_capacity(batch);
        match &self.decoder {
            Decoder::Ctc(head) => {
                let classes = head.num_classes();
                let labels = ctc_argmax(
                    &head.logits(encoded)?.into_data().try_to_vec::<f32>()?,
                    batch * time,
                    classes,
                )?;
                for (i, &n) in lengths.iter().enumerate() {
                    let mut ids = Vec::new();
                    let mut frames = Vec::new();
                    let mut previous = classes - 1;
                    for (frame, &id) in labels[i * time..i * time + n].iter().enumerate() {
                        if id != classes - 1 && id != previous {
                            ids.push(id);
                            frames.push(frame);
                        }
                        previous = id;
                    }
                    results.push(Transcription {
                        text: self.tokenizer.decode(&ids)?,
                        token_ids: ids,
                        frames,
                    });
                }
            }
            Decoder::Rnnt(head) => {
                let device = encoded.device();
                for (i, &n) in lengths.iter().enumerate() {
                    let (ids, frames) = if n == 0 {
                        (Vec::new(), Vec::new())
                    } else {
                        let (ids, frames, _) = head.decode_encoded(
                            encoded.clone().slice([i..i + 1, 0..n, 0..width]),
                            &device,
                        )?;
                        (ids, frames)
                    };
                    results.push(Transcription {
                        text: self.tokenizer.decode(&ids)?,
                        token_ids: ids,
                        frames,
                    });
                }
            }
        }
        Ok(results)
    }
}

/// Shared weights for full-context and stateful native packet inference.
pub struct ZapformerModel {
    config: CheckpointConfig,
    encoder: Arc<ZapformerEncoder>,
    heads: Heads,
    language: Option<Arc<LanguageConditioner>>,
    device: Device,
}
impl ZapformerModel {
    pub fn load_checkpoint(path: &Path, device: Device) -> Result<Self> {
        let metadata = read_checkpoint_metadata(path)?;
        Self::load_with_config(
            path,
            CheckpointConfig::from_upstream(&metadata.config)?,
            device,
        )
    }
    pub fn load_with_config(path: &Path, config: CheckpointConfig, device: Device) -> Result<Self> {
        config.mel.validate()?;
        let mut weights = Weights::open(path)?;
        let directory = path.parent().context("missing checkpoint parent")?;
        let width = config.encoder.embed_dim;
        let encoder =
            ZapformerEncoder::load(&mut weights, "encoder", &device, config.encoder.clone())?;
        let language = config
            .language_conditioning
            .clone()
            .map(|cfg| LanguageConditioner::load(&mut weights, cfg, &config.encoder, &device))
            .transpose()?;
        let heads = match &config.heads {
            HeadsConfig::EncoderOnly => Heads::EncoderOnly,
            HeadsConfig::Single { head } => Heads::Single(Arc::new(AsrHead::load(
                &mut weights,
                "head",
                head,
                width,
                directory,
                &device,
            )?)),
            HeadsConfig::Dual { raw, e2e } => Heads::Dual {
                raw: Arc::new(AsrHead::load(&mut weights, "heads.raw", raw, width, directory, &device)?),
                e2e: Arc::new(AsrHead::load(&mut weights, "heads.e2e", e2e, width, directory, &device)?),
            },
        };
        Ok(Self {
            config,
            encoder,
            heads,
            language,
            device,
        })
    }
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }
    pub fn encode_features(
        &self,
        features: Tensor<3>,
        lengths: &[usize],
        tags: Option<&[LanguageTag]>,
    ) -> Result<(Tensor<3>, Vec<usize>)> {
        ensure!(
            self.language.is_some() || tags.is_none(),
            "checkpoint has no language conditioning"
        );
        let language = self
            .language
            .as_ref()
            .map(|l| l.request(tags, features.dims()[0]))
            .transpose()?;
        let (encoded, lengths) =
            self.encoder
                .forward_conditioned(features, lengths, language.as_ref())?;
        Ok((
            match language {
                Some(l) => l.fuse(encoded),
                None => encoded,
            },
            lengths,
        ))
    }
    pub fn has_dual_heads(&self) -> bool {
        matches!(self.heads, Heads::Dual { .. })
    }

    pub fn validate_language(&self, language: Option<&LanguageTag>) -> Result<()> {
        if let Some(tag) = language {
            self.config.language_conditioning.as_ref().context("checkpoint has no language conditioning")?.vector(tag)?;
        }
        Ok(())
    }

    fn selected_head(&self, selection: HeadSelection) -> Result<&AsrHead> {
        match &self.heads {
            Heads::EncoderOnly => bail!("encoder-only checkpoint does not support ASR"),
            Heads::Single(head) => Ok(head),
            Heads::Dual { raw, e2e } => Ok(match selection {
                HeadSelection::Raw => raw,
                HeadSelection::E2e => e2e,
            }),
        }
    }

    pub fn tokenizer(&self, selection: HeadSelection) -> Result<&Tokenizer> {
        Ok(&self.selected_head(selection)?.tokenizer)
    }

    pub fn decode_selected(&self, encoded: Tensor<3>, lengths: &[usize], selection: HeadSelection) -> Result<Vec<Transcription>> {
        self.selected_head(selection)?.decode(encoded, lengths)
    }

    pub fn selected_ctc_logits(&self, encoded: Tensor<3>, selection: HeadSelection) -> Result<Tensor<3>> {
        match &self.selected_head(selection)?.decoder {
            Decoder::Ctc(head) => head.logits(encoded),
            Decoder::Rnnt(_) => bail!("CTC logits require a CTC head"),
        }
    }

    pub(crate) fn encode_padded(&self, features: &[Features], tags: Option<&[LanguageTag]>) -> Result<(Tensor<3>, Vec<usize>)> {
        ensure!(!features.is_empty(), "empty feature batch");
        let mel = self.config.encoder.in_channels;
        ensure!(features.iter().all(|f| f.n_mels == mel && f.frames > 0 && f.values.len() == mel * f.frames), "invalid mel batch dimensions");
        let batch = features.len();
        let time = features.iter().map(|f| f.frames).max().unwrap();
        let lengths: Vec<_> = features.iter().map(|f| f.frames).collect();
        let mut values = vec![0.0; batch * time * mel];
        for (i, f) in features.iter().enumerate() {
            for m in 0..mel {
                let offset = (i * mel + m) * time;
                values[offset..offset + f.frames].copy_from_slice(&f.values[m * f.frames..(m+1) * f.frames]);
            }
        }
        self.encode_features(Tensor::<3>::from_data(TensorData::new(values, [batch, mel, time]), &self.device).swap_dims(1, 2), &lengths, tags)
    }

    pub(crate) fn pcm_features(&self, pcm: &[f32]) -> Result<Features> {
        let features = log_mel_with_config(pcm, self.config.encoder.in_channels, self.config.mel)?;
        ensure!(features.frames == self.config.mel.reported_frames(pcm.len()), "STFT/reported length mismatch");
        Ok(features)
    }

    pub fn decode_encoded(
        &self,
        encoded: Tensor<3>,
        lengths: &[usize],
    ) -> Result<HeadTranscriptions> {
        match &self.heads {
            Heads::EncoderOnly => bail!("encoder-only checkpoint does not support ASR"),
            Heads::Single(head) => Ok(HeadTranscriptions::Single(head.decode(encoded, lengths)?)),
            Heads::Dual { raw, e2e } => Ok(HeadTranscriptions::Dual {
                raw: raw.decode(encoded.clone(), lengths)?,
                e2e: e2e.decode(encoded, lengths)?,
            }),
        }
    }
    pub fn encode_pcm_16k(
        &self,
        pcm: &[f32],
        tag: Option<&LanguageTag>,
    ) -> Result<(Tensor<3>, Vec<usize>)> {
        let features = self.pcm_features(pcm)?;
        self.encode_padded(&[features], tag.map(std::slice::from_ref))
    }
    pub fn transcribe_pcm_16k(
        &self,
        pcm: &[f32],
        tag: Option<&LanguageTag>,
    ) -> Result<HeadTranscriptions> {
        let (encoded, lengths) = self.encode_pcm_16k(pcm, tag)?;
        self.decode_encoded(encoded, &lengths)
    }
}

mod stream;
pub use stream::{StreamPacket, StreamTranscription, ZapformerStream};
