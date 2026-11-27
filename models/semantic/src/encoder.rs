use crate::{Pooling, normalize};
use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};
use candle_transformers::models::{bert, distilbert, xlm_roberta as roberta};
use serde_json::{Value, json};
use std::path::Path;
use tokenizers::{EncodeInput, Encoding, Tokenizer, TruncationParams};

enum Backbone {
    Bert(bert::BertModel),
    DistilBert(distilbert::DistilBertModel),
    Roberta(roberta::XLMRobertaModel),
    RobertaClassifier(roberta::XLMRobertaForSequenceClassification),
}

enum Head {
    Bert {
        pooler: Linear,
        classifier: Linear,
    },
    DistilBert {
        pre_classifier: Linear,
        classifier: Linear,
    },
    Roberta,
}

pub struct Encoder {
    tokenizer: Tokenizer,
    backbone: Backbone,
    head: Option<Head>,
    pooling: Option<Pooling>,
    positive: Option<usize>,
}

impl Encoder {
    pub fn load(
        weights: &Path,
        mut config: Value,
        mut tokenizer: Tokenizer,
        pooling: Option<Pooling>,
        positive: Option<usize>,
        max_tokens: usize,
    ) -> Result<Self> {
        let vb = VarBuilder::from_buffered_safetensors(
            std::fs::read(weights)?,
            DType::F32,
            &Device::Cpu,
        )?;
        tokenizer.with_padding(None);
        let (backbone, head, limit) = match config["model_type"].as_str() {
            Some("bert") => {
                let cfg: bert::Config = serde_json::from_value(config)?;
                let weights = model_weights(&vb, "bert");
                let backbone = Backbone::Bert(bert::BertModel::load(weights.clone(), &cfg)?);
                let head = if pooling.is_none() {
                    Some(Head::Bert {
                        pooler: linear(
                            cfg.hidden_size,
                            cfg.hidden_size,
                            weights.pp("pooler.dense"),
                        )?,
                        classifier: linear(
                            cfg.hidden_size,
                            labels(&vb, "classifier.weight", positive)?,
                            vb.pp("classifier"),
                        )?,
                    })
                } else {
                    None
                };
                (backbone, head, cfg.max_position_embeddings)
            }
            Some("distilbert") => {
                let limit = serde_json::from_value(config["max_position_embeddings"].clone())?;
                let cfg: distilbert::Config = serde_json::from_value(config)?;
                let weights = model_weights(&vb, "distilbert");
                let backbone =
                    Backbone::DistilBert(distilbert::DistilBertModel::load(weights, &cfg)?);
                let head = if pooling.is_none() {
                    Some(Head::DistilBert {
                        pre_classifier: linear(cfg.dim, cfg.dim, vb.pp("pre_classifier"))?,
                        classifier: linear(
                            cfg.dim,
                            labels(&vb, "classifier.weight", positive)?,
                            vb.pp("classifier"),
                        )?,
                    })
                } else {
                    None
                };
                (backbone, head, limit)
            }
            Some("roberta" | "xlm-roberta") => {
                if config.get("position_embedding_type").is_none() {
                    config["position_embedding_type"] = json!("absolute");
                }
                let cfg: roberta::Config = serde_json::from_value(config)?;
                let limit = cfg
                    .max_position_embeddings
                    .checked_sub(cfg.pad_token_id as usize + 1)
                    .context("invalid RoBERTa position configuration")?;
                if pooling.is_some() {
                    (
                        Backbone::Roberta(roberta::XLMRobertaModel::new(
                            &cfg,
                            model_weights(&vb, "roberta"),
                        )?),
                        None,
                        limit,
                    )
                } else {
                    let count = labels(&vb, "classifier.out_proj.weight", positive)?;
                    (
                        Backbone::RobertaClassifier(
                            roberta::XLMRobertaForSequenceClassification::new(count, &cfg, vb)?,
                        ),
                        Some(Head::Roberta),
                        limit,
                    )
                }
            }
            arch => bail!("unsupported encoder architecture {arch:?}"),
        };
        let max_length = max_tokens.min(limit).min(
            tokenizer
                .get_truncation()
                .map_or(usize::MAX, |x| x.max_length),
        );
        ensure!(max_length >= 8, "token limit is too small");
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(anyhow::Error::msg)?;
        Ok(Self {
            tokenizer,
            backbone,
            head,
            pooling,
            positive,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoded = self.encode(text.into())?;
        let values = self.forward(&encoded)?;
        let pooled = match self.pooling.context("embedding pooling is missing")? {
            Pooling::Cls => values.get(0)?,
            Pooling::Mean => values.mean(0)?,
            Pooling::Last => values.get(encoded.len() - 1)?,
        };
        normalize(pooled.to_vec1::<f32>()?)
    }

    pub fn rank(&self, query: &str, document: &str) -> Result<f32> {
        let encoded = self.encode((query, document).into())?;
        let output = self.forward(&encoded)?;
        let logits = match self
            .head
            .as_ref()
            .context("classification head is missing")?
        {
            Head::Bert { pooler, classifier } => classifier
                .forward(&pooler.forward(&output.narrow(0, 0, 1)?)?.tanh()?)?
                .squeeze(0)?,
            Head::DistilBert {
                pre_classifier,
                classifier,
            } => classifier
                .forward(&pre_classifier.forward(&output.narrow(0, 0, 1)?)?.relu()?)?
                .squeeze(0)?,
            Head::Roberta => output,
        };
        let scores = logits.to_vec1::<f32>()?;
        let score = if let [logit] = scores.as_slice() {
            1.0 / (1.0 + (-logit).exp())
        } else {
            let probabilities = candle_nn::ops::softmax_last_dim(&logits)?.to_vec1::<f32>()?;
            probabilities[self.positive.context("positive label must be specified")?]
        };
        ensure!(score.is_finite(), "invalid classifier score");
        Ok(score)
    }

    fn encode<'a>(&self, text: EncodeInput<'a>) -> Result<Encoding> {
        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)?;
        ensure!(!encoded.is_empty(), "tokenizer produced no tokens");
        Ok(encoded)
    }

    fn forward(&self, encoded: &Encoding) -> Result<Tensor> {
        let shape = (1, encoded.len());
        let ids = Tensor::from_slice(encoded.get_ids(), shape, &Device::Cpu)?;
        let types = Tensor::from_slice(encoded.get_type_ids(), shape, &Device::Cpu)?;
        let mask = Tensor::from_slice(encoded.get_attention_mask(), shape, &Device::Cpu)?;
        let result = match &self.backbone {
            Backbone::Bert(model) => model.forward(&ids, &types, Some(&mask))?,
            Backbone::DistilBert(model) => {
                let blocked = mask.eq(0_u32)?.unsqueeze(1)?.unsqueeze(1)?;
                model.forward(&ids, &blocked)?
            }
            Backbone::Roberta(model) => model.forward(&ids, &mask, &types, None, None, None)?,
            Backbone::RobertaClassifier(model) => model.forward(&ids, &mask, &types)?,
        };
        Ok(result.squeeze(0)?)
    }
}

fn model_weights<'a>(vb: &VarBuilder<'a>, prefix: &str) -> VarBuilder<'a> {
    if vb.contains_tensor(&format!("{prefix}.embeddings.word_embeddings.weight")) {
        vb.pp(prefix)
    } else {
        vb.clone()
    }
}

fn labels(vb: &VarBuilder<'_>, classifier: &str, positive: Option<usize>) -> Result<usize> {
    let (count, _) = vb.get_unchecked(classifier)?.dims2()?;
    ensure!(
        count == 1 || positive.is_some_and(|label| label < count),
        "specify --positive-label for a multi-label classifier"
    );
    ensure!(
        count != 1 || positive.is_none(),
        "--positive-label is only used for classifiers with multiple logits"
    );
    Ok(count)
}
