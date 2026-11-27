use crate::normalize;
use anyhow::{Context, Result, ensure};
use burn::tensor::{Bool, DType, Device, Int, Tensor, TensorData};
use burn_store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use native_burn_qwen3::{LLMConfig, Qwen3Model, Qwen3ModelConfig};
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

pub const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

pub struct Qwen {
    backbone: Qwen3Model,
    tokenizer: Tokenizer,
    device: Device,
    decision_head: Option<Tensor<2>>,
    instruction: String,
    max_tokens: usize,
}

impl Qwen {
    pub fn load(
        weights: &Path,
        config: Value,
        mut tokenizer: Tokenizer,
        device: &Device,
        instruction: Option<String>,
        max_tokens: usize,
    ) -> Result<Self> {
        let tied = config["tie_word_embeddings"] == true;
        let config: LLMConfig = serde_json::from_value(config)?;
        config.validate()?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            tokenizer.get_vocab_size(true) <= config.vocab_size,
            "tokenizer exceeds vocabulary"
        );
        let mut backbone = Qwen3ModelConfig::new().init(&config, device);
        let mut store = SafetensorsStore::from_file(weights)
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .with_key_remapping(r"^model\.", "");
        let results = backbone.load_from(&mut store).map_err(anyhow::Error::msg)?;
        ensure!(
            !results.applied.is_empty() && results.errors.is_empty(),
            "Qwen3 weights: {results}"
        );
        let decision_head = if instruction.is_some() {
            let no = tokenizer
                .token_to_id("no")
                .context("tokenizer has no 'no' token")?;
            let yes = tokenizer
                .token_to_id("yes")
                .context("tokenizer has no 'yes' token")?;
            ensure!(
                (yes as usize) < config.vocab_size && (no as usize) < config.vocab_size,
                "invalid decision tokens"
            );
            let ids = Tensor::<1, Int>::from_data([no as i64, yes as i64], device);
            let head = if tied {
                backbone.embed_tokens.weight.val().select(0, ids)
            } else {
                let cpu = candle_core::Device::Cpu;
                let vb = candle_nn::VarBuilder::from_buffered_safetensors(
                    std::fs::read(weights)?,
                    candle_core::DType::F32,
                    &cpu,
                )?;
                let weight = vb.get((config.vocab_size, config.hidden_size), "lm_head.weight")?;
                let rows = weight.index_select(&candle_core::Tensor::new(&[no, yes], &cpu)?, 0)?;
                Tensor::from_data(
                    TensorData::new(
                        rows.flatten_all()?.to_vec1::<f32>()?,
                        [2, config.hidden_size],
                    ),
                    device,
                )
            };
            Some(head.transpose())
        } else {
            None
        };
        device.sync()?;
        Ok(Self {
            backbone,
            tokenizer,
            device: device.clone(),
            decision_head,
            instruction: instruction.unwrap_or_default(),
            max_tokens,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut tokens = self.ids(text, true)?;
        tokens.truncate(self.max_tokens);
        normalize(
            self.last_hidden(&tokens)?
                .into_data()
                .try_to_vec::<f32>()
                .map_err(anyhow::Error::msg)?,
        )
    }

    pub fn rank(&self, query: &str, document: &str) -> Result<f32> {
        let head = self
            .decision_head
            .as_ref()
            .context("reranker head is not loaded")?;
        let prefix = format!(
            r#"<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {}
<Query>: {query}
<Document>: "#,
            self.instruction
        );
        let mut tokens = self.ids(&format!("{prefix}{document}{SUFFIX}"), false)?;
        if tokens.len() > self.max_tokens {
            let tail = self.ids(SUFFIX, false)?;
            ensure!(
                tail.len() < self.max_tokens,
                "token limit is too small for Qwen3 reranking"
            );
            tokens.truncate(self.max_tokens - tail.len());
            tokens.extend(tail);
        }
        let logits = self
            .last_hidden(&tokens)?
            .matmul(head.clone())
            .into_data()
            .try_to_vec::<f32>()
            .map_err(anyhow::Error::msg)?;
        let score = 1.0 / (1.0 + (logits[0] - logits[1]).exp());
        ensure!(score.is_finite(), "invalid reranker score");
        Ok(score)
    }

    fn ids(&self, text: &str, special: bool) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, special)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec())
    }

    fn last_hidden(&self, tokens: &[u32]) -> Result<Tensor<2>> {
        let len = tokens.len();
        ensure!((1..=self.max_tokens).contains(&len), "invalid token count");
        let ids = Tensor::<2, Int>::from_data(
            TensorData::new(
                tokens.iter().map(|&id| id as i64).collect::<Vec<_>>(),
                [1, len],
            ),
            &self.device,
        );
        let positions = Tensor::arange(0..len as i64, &self.device).unsqueeze_dim::<2>(0);
        let mask: Vec<bool> = (0..len)
            .flat_map(|row| (0..len).map(move |column| column <= row))
            .collect();
        let mask =
            Tensor::<4, Bool>::from_data(TensorData::new(mask, [1, 1, len, len]), &self.device);
        let hidden =
            self.backbone
                .forward(positions, self.backbone.embed_tokens.forward(ids), mask);
        let width = hidden.dims()[2];
        Ok(hidden.slice([0..1, len - 1..len, 0..width]).squeeze_dim(1))
    }
}
