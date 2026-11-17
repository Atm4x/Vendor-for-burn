use std::collections::HashSet;

use crate::audio::{firefly_decoded_samples, firefly_encoded_frames, N_MELS};
use crate::config::{FishModelConfig, TransformerDims};
use crate::error::{FishError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    SlowPrefill,
    SlowDecode,
    FastDecode,
    FireflyEncode,
    FireflyDecode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int32,
    Float32,
    BFloat16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    EmbeddingSum,
    QkvProjection,
    RotaryEmbedding,
    GqaAttention,
    KvWrite,
    RmsNorm,
    SwiGlu,
    Residual,
    Projection,
    Conv1d,
    ConvTranspose1d,
    ConvNeXtBlock,
    LogMelInput,
    FsqEncode,
    FsqDecode,
    HiFiGanResBlock,
    Tanh,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DataType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedOp {
    pub name: String,
    pub kind: Op,
    pub output: TensorSpec,
}

#[derive(Debug, Clone)]
pub struct NetworkPlan {
    pub stage: Stage,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    pub operations: Vec<PlannedOp>,
    pub active_kv_len: Option<usize>,
    pub kv_capacity: Option<usize>,
}

impl NetworkPlan {
    pub fn validate(&self) -> Result<()> {
        if self.inputs.is_empty() || self.outputs.is_empty() || self.operations.is_empty() {
            return Err(FishError::InvalidShape("empty graph plan"));
        }
        let mut names = HashSet::new();
        for tensor in self
            .inputs
            .iter()
            .chain(self.operations.iter().map(|op| &op.output))
        {
            if tensor.shape.is_empty() || tensor.shape.contains(&0) {
                return Err(FishError::InvalidShape("zero extent"));
            }
            if !names.insert(tensor.name.as_str()) {
                return Err(FishError::InvalidShape("duplicate tensor definition"));
            }
        }
        for output in &self.outputs {
            let declared = self
                .inputs
                .iter()
                .chain(self.operations.iter().map(|op| &op.output))
                .any(|tensor| {
                    tensor.name == output.name
                        && tensor.shape == output.shape
                        && tensor.dtype == output.dtype
                });
            if !declared {
                return Err(FishError::InvalidShape(
                    "unbound or mismatched graph output",
                ));
            }
        }
        match (self.active_kv_len, self.kv_capacity) {
            (Some(active), Some(capacity)) if active >= 1 && active <= capacity => {}
            (None, None) => {}
            _ => {
                return Err(FishError::InvalidShape(
                    "invalid active-prefix or KV capacity",
                ))
            }
        }
        Ok(())
    }
}

fn tensor(name: impl Into<String>, shape: &[usize], dtype: DataType) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        shape: shape.to_vec(),
        dtype,
    }
}

fn push(p: &mut NetworkPlan, name: impl Into<String>, kind: Op, shape: &[usize], dtype: DataType) {
    let name = name.into();
    p.operations.push(PlannedOp {
        output: tensor(name.clone(), shape, dtype),
        name,
        kind,
    });
}

pub struct PlanFactory<'a> {
    config: &'a FishModelConfig,
}

impl<'a> PlanFactory<'a> {
    pub fn new(config: &'a FishModelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    fn transformer_plan(
        &self,
        stage: Stage,
        context_len: usize,
        query_len: usize,
        capacity: usize,
        dims: &TransformerDims,
        vocab_size: usize,
    ) -> Result<NetworkPlan> {
        if context_len == 0 || query_len == 0 || query_len > context_len || context_len > capacity {
            return Err(FishError::InvalidShape(
                "invalid transformer sequence lengths",
            ));
        }
        let c = self.config;
        let mut p = NetworkPlan {
            stage,
            inputs: vec![
                tensor(
                    "tokens",
                    &[1, 1 + c.num_codebooks, query_len],
                    DataType::Int32,
                ),
                tensor(
                    "kv_k",
                    &[dims.layers, 1, dims.n_kv_heads, capacity, dims.head_dim()],
                    DataType::BFloat16,
                ),
                tensor(
                    "kv_v",
                    &[dims.layers, 1, dims.n_kv_heads, capacity, dims.head_dim()],
                    DataType::BFloat16,
                ),
            ],
            outputs: vec![],
            operations: vec![],
            active_kv_len: Some(context_len),
            kv_capacity: Some(capacity),
        };
        push(
            &mut p,
            "embedding_sum",
            Op::EmbeddingSum,
            &[1, query_len, dims.dim],
            DataType::Float32,
        );
        for layer in 0..dims.layers {
            push(
                &mut p,
                format!("layer{layer}.attn_norm"),
                Op::RmsNorm,
                &[1, query_len, dims.dim],
                DataType::Float32,
            );
            push(
                &mut p,
                format!("layer{layer}.wqkv"),
                Op::QkvProjection,
                &[1, query_len, dims.qkv_width()],
                DataType::Float32,
            );
            push(
                &mut p,
                format!("layer{layer}.rope"),
                Op::RotaryEmbedding,
                &[1, dims.n_heads, query_len, dims.head_dim()],
                DataType::Float32,
            );
            push(
                &mut p,
                format!("layer{layer}.kv_update"),
                Op::KvWrite,
                &[1, dims.n_kv_heads, context_len, dims.head_dim()],
                DataType::BFloat16,
            );
            push(
                &mut p,
                format!("layer{layer}.gqa"),
                Op::GqaAttention,
                &[1, dims.n_heads, query_len, dims.head_dim()],
                DataType::Float32,
            );
            push(
                &mut p,
                format!("layer{layer}.attn_residual"),
                Op::Residual,
                &[1, query_len, dims.dim],
                DataType::Float32,
            );
            push(
                &mut p,
                format!("layer{layer}.ffn_norm"),
                Op::RmsNorm,
                &[1, query_len, dims.dim],
                DataType::Float32,
            );
            push(
                &mut p,
                format!("layer{layer}.swiglu"),
                Op::SwiGlu,
                &[1, query_len, dims.intermediate_size],
                DataType::Float32,
            );
            push(
                &mut p,
                format!("layer{layer}.ffn_residual"),
                Op::Residual,
                &[1, query_len, dims.dim],
                DataType::Float32,
            );
        }
        push(
            &mut p,
            "hidden",
            Op::RmsNorm,
            &[1, 1, dims.dim],
            DataType::Float32,
        );
        push(
            &mut p,
            "logits",
            Op::Projection,
            &[1, 1, vocab_size],
            DataType::Float32,
        );
        p.outputs = vec![
            tensor("hidden", &[1, 1, dims.dim], DataType::Float32),
            tensor("logits", &[1, 1, vocab_size], DataType::Float32),
        ];
        p.validate()?;
        Ok(p)
    }

    pub fn slow_prefill(&self, prompt_len: usize) -> Result<NetworkPlan> {
        self.transformer_plan(
            Stage::SlowPrefill,
            prompt_len,
            prompt_len,
            self.config.max_seq_len,
            &self.config.slow,
            self.config.vocab_size,
        )
    }

    pub fn slow_decode(&self, active_kv_len: usize) -> Result<NetworkPlan> {
        self.transformer_plan(
            Stage::SlowDecode,
            active_kv_len,
            1,
            self.config.max_seq_len,
            &self.config.slow,
            self.config.vocab_size,
        )
    }

    pub fn fast_decode(&self, step: usize) -> Result<NetworkPlan> {
        let dims = self
            .config
            .fast
            .as_ref()
            .ok_or(FishError::Unsupported("Fast AR absent in Naive model"))?;
        if step >= self.config.num_codebooks {
            return Err(FishError::InvalidPosition {
                position: step,
                capacity: self.config.num_codebooks,
            });
        }
        let mut p = self.transformer_plan(
            Stage::FastDecode,
            step + 1,
            1,
            self.config.num_codebooks,
            dims,
            self.config.codebook_size,
        )?;
        p.inputs[0] = tensor("fast_embedded_input", &[1, 1, dims.dim], DataType::Float32);
        p.operations.remove(0);
        p.validate()?;
        Ok(p)
    }

    pub fn firefly_encode(&self, mel_frames: usize) -> Result<NetworkPlan> {
        let frames = firefly_encoded_frames(mel_frames)?;
        let mut p = NetworkPlan {
            stage: Stage::FireflyEncode,
            inputs: vec![tensor(
                "log_mel",
                &[1, N_MELS, mel_frames],
                DataType::Float32,
            )],
            outputs: vec![],
            operations: vec![],
            active_kv_len: None,
            kv_capacity: None,
        };
        push(
            &mut p,
            "mel_input",
            Op::LogMelInput,
            &[1, N_MELS, mel_frames],
            DataType::Float32,
        );
        for (i, depth) in [3, 3, 9, 3].into_iter().enumerate() {
            let dim = [128, 256, 384, 512][i];
            push(
                &mut p,
                format!("encoder.stage{i}.stem"),
                Op::Conv1d,
                &[1, dim, mel_frames],
                DataType::Float32,
            );
            for block in 0..depth {
                push(
                    &mut p,
                    format!("encoder.stage{i}.block{block}"),
                    Op::ConvNeXtBlock,
                    &[1, dim, mel_frames],
                    DataType::Float32,
                );
            }
        }
        push(
            &mut p,
            "fsq_downsample0",
            Op::Conv1d,
            &[1, 512, (mel_frames + 1) / 2],
            DataType::Float32,
        );
        push(
            &mut p,
            "fsq_downsample1",
            Op::Conv1d,
            &[1, 512, frames],
            DataType::Float32,
        );
        push(
            &mut p,
            "acoustic_codes",
            Op::FsqEncode,
            &[1, 8, frames],
            DataType::Int32,
        );
        p.outputs = vec![tensor("acoustic_codes", &[1, 8, frames], DataType::Int32)];
        p.validate()?;
        Ok(p)
    }

    pub fn firefly_decode(&self, code_frames: usize) -> Result<NetworkPlan> {
        let output_samples = firefly_decoded_samples(code_frames)?;
        let mut p = NetworkPlan {
            stage: Stage::FireflyDecode,
            inputs: vec![tensor(
                "acoustic_codes",
                &[1, 8, code_frames],
                DataType::Int32,
            )],
            outputs: vec![],
            operations: vec![],
            active_kv_len: None,
            kv_capacity: None,
        };
        push(
            &mut p,
            "fsq_values",
            Op::FsqDecode,
            &[1, 512, code_frames],
            DataType::Float32,
        );
        push(
            &mut p,
            "fsq_upsample0",
            Op::ConvTranspose1d,
            &[1, 512, code_frames * 2],
            DataType::Float32,
        );
        push(
            &mut p,
            "fsq_upsample1",
            Op::ConvTranspose1d,
            &[1, 512, code_frames * 4],
            DataType::Float32,
        );
        let mut frames = code_frames * 4;
        for (i, &stride) in [8, 8, 2, 2, 2].iter().enumerate() {
            frames *= stride;
            push(
                &mut p,
                format!("hifigan.upsample{i}"),
                Op::ConvTranspose1d,
                &[1, 256 / (1 << i), frames],
                DataType::Float32,
            );
            push(
                &mut p,
                format!("hifigan.resblocks{i}"),
                Op::HiFiGanResBlock,
                &[1, 256 / (1 << i), frames],
                DataType::Float32,
            );
        }
        push(
            &mut p,
            "waveform",
            Op::Tanh,
            &[1, 1, output_samples],
            DataType::Float32,
        );
        p.outputs = vec![tensor(
            "waveform",
            &[1, 1, output_samples],
            DataType::Float32,
        )];
        p.validate()?;
        Ok(p)
    }
}

#[derive(Default)]
pub struct MockBackend;
impl MockBackend {
    pub fn validate_plan(&self, plan: &NetworkPlan) -> Result<()> {
        plan.validate()
    }
    pub fn execute(&self, _plan: &NetworkPlan) -> Result<()> {
        Err(FishError::Unsupported(
            "mock backend cannot execute inference",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> FishModelConfig {
        FishModelConfig::synthetic_dual_ar_fixture()
    }
    #[test]
    fn prefills_and_decode_shapes() {
        let c = fixture();
        let b = PlanFactory::new(&c).unwrap();
        let pre = b.slow_prefill(33).unwrap();
        assert_eq!(pre.inputs[0].shape, [1, 9, 33]);
        assert_eq!(pre.active_kv_len, Some(33));
        let dec = b.slow_decode(34).unwrap();
        assert_eq!(dec.inputs[0].shape, [1, 9, 1]);
        assert_eq!(dec.active_kv_len, Some(34));
        assert_eq!(dec.kv_capacity, Some(256));
    }
    #[test]
    fn gqa_is_not_materialized() {
        let c = fixture();
        let b = PlanFactory::new(&c).unwrap();
        let p = b.slow_decode(35).unwrap();
        let kv = p.operations.iter().find(|n| n.kind == Op::KvWrite).unwrap();
        assert_eq!(kv.output.shape, [1, 2, 35, 8]);
    }
    #[test]
    fn fast_cache_progression() {
        let c = fixture();
        let b = PlanFactory::new(&c).unwrap();
        for step in 0..8 {
            let p = b.fast_decode(step).unwrap();
            assert_eq!(p.active_kv_len, Some(step + 1));
        }
        assert!(b.fast_decode(8).is_err());
    }
    #[test]
    fn firefly_graph_shapes() {
        let c = fixture();
        let b = PlanFactory::new(&c).unwrap();
        let enc = b.firefly_encode(13).unwrap();
        assert_eq!(enc.outputs[0].shape, [1, 8, 4]);
        let dec = b.firefly_decode(4).unwrap();
        assert_eq!(dec.outputs[0].shape, [1, 1, 8192]);
    }
    #[test]
    fn poison_invalid_plan() {
        let c = fixture();
        let b = PlanFactory::new(&c).unwrap();
        let mut p = b.slow_prefill(2).unwrap();
        p.active_kv_len = Some(257);
        assert!(MockBackend.validate_plan(&p).is_err());
    }
    #[test]
    fn mock_never_executes() {
        let c = fixture();
        let p = PlanFactory::new(&c).unwrap().slow_prefill(2).unwrap();
        assert!(MockBackend.execute(&p).is_err());
    }
    #[test]
    fn reject_zero_and_overrun() {
        let c = fixture();
        let b = PlanFactory::new(&c).unwrap();
        assert!(b.slow_prefill(0).is_err());
        assert!(b.slow_decode(257).is_err());
        assert!(b.firefly_encode(0).is_err());
    }
}
