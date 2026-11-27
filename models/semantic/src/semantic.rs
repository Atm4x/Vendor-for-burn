use anyhow::{Context, Result, bail, ensure};
use burn::tensor::Device;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::{
    encoder::Encoder,
    qwen::{DEFAULT_INSTRUCTION, Qwen},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Cls,
    Mean,
    Last,
}

impl Pooling {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "cls" => Ok(Self::Cls),
            "mean" => Ok(Self::Mean),
            "last" => Ok(Self::Last),
            _ => bail!("expected --pooling cls|mean|last"),
        }
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let config: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let modes: Vec<_> = config
            .as_object()
            .context("expected a pooling configuration object")?
            .iter()
            .filter(|(name, value)| {
                name.starts_with("pooling_mode_") && value.as_bool() == Some(true)
            })
            .map(|(name, _)| name.as_str())
            .collect();
        match modes.as_slice() {
            ["pooling_mode_cls_token"] => Ok(Self::Cls),
            ["pooling_mode_mean_tokens"] => Ok(Self::Mean),
            ["pooling_mode_lasttoken"] => Ok(Self::Last),
            _ => bail!("unsupported Sentence Transformers pooling modes: {modes:?}"),
        }
    }
}

pub enum Model {
    Qwen(Box<Qwen>),
    Encoder(Box<Encoder>),
}

pub struct Artifacts {
    pub checkpoint: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub pooling: Option<Pooling>,
    pub positive: Option<usize>,
    pub instruction: Option<String>,
    pub max_tokens: usize,
}

impl Artifacts {
    pub fn load(&self, reranker: bool, backend: &str) -> Result<Model> {
        let config: Value = serde_json::from_slice(&std::fs::read(&self.config)?)?;
        let model_type = config["model_type"]
            .as_str()
            .context("config.json has no model_type")?;
        let tokenizer = Tokenizer::from_file(&self.tokenizer).map_err(anyhow::Error::msg)?;
        if model_type == "qwen3" {
            ensure!(
                self.pooling.is_none_or(|p| p == Pooling::Last),
                "Qwen3 embedding requires last-token pooling"
            );
            ensure!(
                self.positive.is_none(),
                "--positive-label only applies to classifier heads"
            );
            let instruction = reranker.then(|| {
                self.instruction
                    .as_deref()
                    .unwrap_or(DEFAULT_INSTRUCTION)
                    .to_owned()
            });
            let device = match backend {
                "cpu" => Device::flex(),
                "cuda" => Device::cuda(0),
                "wgpu" => Device::wgpu(Default::default()),
                _ => unreachable!(),
            };
            let max_tokens = self
                .max_tokens
                .min(config["max_position_embeddings"].as_u64().unwrap_or(32_768) as usize);
            return Ok(Model::Qwen(Box::new(Qwen::load(
                &self.checkpoint,
                config,
                tokenizer,
                &device,
                instruction,
                max_tokens,
            )?)));
        }
        ensure!(
            matches!(
                model_type,
                "bert" | "distilbert" | "roberta" | "xlm-roberta"
            ),
            "unsupported model_type {model_type}"
        );
        ensure!(
            backend == "cpu",
            "{model_type} currently uses the Candle CPU backend"
        );
        ensure!(
            self.instruction.is_none(),
            "--instruction only applies to Qwen3 reranking"
        );
        ensure!(
            reranker || self.pooling.is_some(),
            "specify --pooling-config or --pooling for {model_type} embeddings"
        );
        Ok(Model::Encoder(Box::new(Encoder::load(
            &self.checkpoint,
            config,
            tokenizer,
            self.pooling,
            self.positive,
            self.max_tokens,
        )?)))
    }
}

impl Model {
    pub fn rerank_protocol(&self) -> &'static str {
        match self {
            Self::Qwen(_) => "local-lm-reranker-v1",
            Self::Encoder(_) => "local-cross-encoder-v1",
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Qwen(model) => model.embed(text),
            Self::Encoder(model) => model.embed(text),
        }
    }

    pub fn rank(&self, query: &str, document: &str) -> Result<f32> {
        match self {
            Self::Qwen(model) => model.rank(query, document),
            Self::Encoder(model) => model.rank(query, document),
        }
    }
}

pub fn normalize(mut values: Vec<f32>) -> Result<Vec<f32>> {
    let norm = values
        .iter()
        .map(|&value| value * value)
        .sum::<f32>()
        .sqrt();
    ensure!(norm.is_finite() && norm > 0.0, "invalid embedding norm");
    for value in &mut values {
        *value /= norm;
    }
    Ok(values)
}
