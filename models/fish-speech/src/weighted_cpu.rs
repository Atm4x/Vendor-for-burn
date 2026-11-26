use fish_speech_core::config::{FishModelConfig, ModelKind};
use fish_speech_core::conversation::EncodedConversation;
use fish_speech_core::cpu_weighted_ar::{CpuWeightedDualAr, SelectionMode};
use fish_speech_core::cpu_weighted_naive::CpuWeightedNaive;
use fish_speech_core::error::Result as FishResult;
use fish_speech_core::sampling::SamplingParams;
use fish_speech_core::synthesis::SemanticGenerator;
use fish_speech_core::tokenizer::{SEMANTIC_TOKEN_COUNT, SpecialTokens};

use crate::weights::{LoadedWeights, Result as WeightResult, WeightError};

pub enum CpuSemanticGenerator {
    DualAr(CpuWeightedDualAr),
    Naive(CpuWeightedNaive),
}

impl CpuSemanticGenerator {
    pub fn from_checkpoint(
        config: FishModelConfig,
        weights: &LoadedWeights,
        special: &SpecialTokens,
        seed: u64,
        mode: SelectionMode,
    ) -> WeightResult<Self> {
        let end = special
            .semantic_begin
            .checked_add(SEMANTIC_TOKEN_COUNT)
            .ok_or_else(|| WeightError::Invalid("semantic token ID overflow".into()))?;
        if end > config.vocab_size || special.im_end >= config.vocab_size {
            return Err(WeightError::Invalid(
                "tokenizer semantic IDs exceed model vocabulary".into(),
            ));
        }
        let mut semantic_lut = vec![0i32; config.vocab_size];
        semantic_lut[special.semantic_begin..end].fill(1);
        let tensors = weights.semantic_cpu_weights(&config)?;
        let eos = special.im_end as u32;
        let generator = match config.kind {
            ModelKind::DualAr => Self::DualAr(
                CpuWeightedDualAr::new(
                    config,
                    tensors,
                    semantic_lut,
                    special.semantic_begin as u32,
                    eos,
                    seed,
                    mode,
                )
                .map_err(|e| WeightError::Invalid(format!("Dual-AR CPU model: {e}")))?,
            ),
            ModelKind::Naive => Self::Naive(
                CpuWeightedNaive::new(config, tensors, semantic_lut, eos, seed, mode)
                    .map_err(|e| WeightError::Invalid(format!("Naive CPU model: {e}")))?,
            ),
        };
        Ok(generator)
    }
}

impl SemanticGenerator for CpuSemanticGenerator {
    fn begin_sample(&mut self) -> FishResult<()> {
        match self {
            Self::DualAr(model) => model.begin_sample(),
            Self::Naive(model) => model.begin_sample(),
        }
    }

    fn generate(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        sampling: SamplingParams,
    ) -> FishResult<Vec<Vec<u32>>> {
        match self {
            Self::DualAr(model) => model.generate(prompt, max_new_tokens, sampling),
            Self::Naive(model) => model.generate(prompt, max_new_tokens, sampling),
        }
    }

    fn generate_with_cancel(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        sampling: SamplingParams,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> FishResult<Vec<Vec<u32>>> {
        match self {
            Self::DualAr(model) => {
                model.generate_with_cancel(prompt, max_new_tokens, sampling, is_cancelled)
            }
            Self::Naive(model) => {
                model.generate_with_cancel(prompt, max_new_tokens, sampling, is_cancelled)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fish_speech_core::config::TransformerDims;
    use fish_speech_core::graph_ir::ScalarType;
    use fish_speech_core::transformer_ir::TransformerIrFactory;
    use safetensors::tensor::{Dtype, TensorView, serialize};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::fs;

    use crate::weights::{MODEL_FAMILY, TensorEntry, WeightManifest};

    fn digest(x: &[u8]) -> String {
        format!("{:x}", Sha256::digest(x))
    }

    #[test]
    fn prepared_naive_checkpoint_runs_through_public_generator_contract() {
        let special = SpecialTokens::new(256).unwrap();
        let config = FishModelConfig {
            kind: ModelKind::Naive,
            slow: TransformerDims {
                dim: 8,
                n_heads: 2,
                n_kv_heads: 1,
                layers: 1,
                intermediate_size: 16,
                qkv_bias: false,
            },
            fast: None,
            vocab_size: special.semantic_end + 1,
            num_codebooks: 2,
            codebook_size: 8,
            max_seq_len: 16,
            rope_base: 10_000.,
            norm_eps: 1e-5,
            tie_word_embeddings: true,
            share_codebook_embeddings: true,
        };
        let factory = TransformerIrFactory::new(&config).unwrap();
        let needs = factory.slow_prefill(1).unwrap().weight_requirements();
        let mut contents = BTreeMap::new();
        let mut entries = BTreeMap::new();
        for (key, ty) in needs {
            assert_eq!(ty.dtype, ScalarType::F32);
            let count: usize = ty.shape.iter().product();
            let values = vec![0u8; count * 4];
            entries.insert(
                key.clone(),
                TensorEntry {
                    shard: "semantic.safetensors".into(),
                    dtype: "F32".into(),
                    shape: ty.shape.clone(),
                    sha256: digest(&values),
                },
            );
            contents.insert(key, (ty.shape, values));
        }
        let views = contents
            .iter()
            .map(|(name, (shape, bytes))| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let encoded = serialize(views, None).unwrap();
        let manifest = WeightManifest {
            schema_version: 1,
            model_family: MODEL_FAMILY.into(),
            component: "semantic".into(),
            source_checkpoint_sha256: digest(b"fake"),
            source_license: "fixture-only".into(),
            config_sha256: digest(b"config"),
            shards: BTreeMap::from([("semantic.safetensors".into(), digest(&encoded))]),
            tensors: entries,
        };
        let path =
            std::env::temp_dir().join(format!("fish-cpu-semantic-test-{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("semantic.safetensors"), encoded).unwrap();
        let loaded = LoadedWeights::load(&path, manifest).unwrap();
        let mut generator = CpuSemanticGenerator::from_checkpoint(
            config,
            &loaded,
            &special,
            2,
            SelectionMode::Greedy,
        )
        .unwrap();
        generator.begin_sample().unwrap();
        let result = generator
            .generate(
                &EncodedConversation {
                    channels: vec![vec![1], vec![0], vec![0]],
                    vq_positions: vec![false],
                },
                2,
                SamplingParams::default(),
            )
            .unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 2);
        fs::remove_dir_all(path).unwrap();
    }
}
