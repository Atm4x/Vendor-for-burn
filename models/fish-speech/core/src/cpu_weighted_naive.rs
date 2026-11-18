use crate::config::{FishModelConfig, ModelKind};
use crate::conversation::EncodedConversation;
use crate::cpu_ir::{NamedTensors, TensorData};
use crate::cpu_weighted_ar::{checked_i32_tokens, run_weighted_graph, SelectionMode};
use crate::error::{FishError, Result};
use crate::sampling::{greedy, logits_to_probs, SamplingParams, SessionRng};
use crate::synthesis::SemanticGenerator;
use crate::transformer_ir::TransformerIrFactory;

pub struct CpuWeightedNaive {
    config: FishModelConfig,
    weights: NamedTensors,
    semantic_lut: Vec<i32>,
    eos_id: u32,
    seed: u64,
    rng: SessionRng,
    mode: SelectionMode,
}

impl CpuWeightedNaive {
    pub fn new(
        config: FishModelConfig,
        weights: NamedTensors,
        semantic_lut: Vec<i32>,
        eos_id: u32,
        seed: u64,
        mode: SelectionMode,
    ) -> Result<Self> {
        config.validate()?;
        if config.kind != ModelKind::Naive {
            return Err(FishError::Unsupported(
                "CPU weighted Naive requires Naive config",
            ));
        }
        if semantic_lut.len() != config.vocab_size
            || semantic_lut.iter().any(|v| !matches!(v, 0 | 1))
            || eos_id as usize >= config.vocab_size
        {
            return Err(FishError::InvalidConfig(
                "invalid Naive semantic LUT or EOS",
            ));
        }
        let graph = TransformerIrFactory::new(&config)?.slow_prefill(1)?;
        for (key, ty) in graph.weight_requirements() {
            let binding = weights
                .get(&key)
                .ok_or(FishError::InvalidShape("missing Naive checkpoint weight"))?;
            if !matches!(binding, TensorData::F32(v) if v.len() == ty.shape.iter().product::<usize>())
            {
                return Err(FishError::InvalidShape(
                    "Naive checkpoint weight shape invalid",
                ));
            }
        }
        Ok(Self {
            config,
            weights,
            semantic_lut,
            eos_id,
            seed,
            rng: SessionRng::new(seed),
            mode,
        })
    }

    fn draw(&mut self, logits: &[f32], history: &[usize], params: SamplingParams) -> Result<u32> {
        let probs = logits_to_probs(logits, history, params)?;
        let id = match self.mode {
            SelectionMode::Greedy => greedy(&probs)?,
            SelectionMode::Sample => self.rng.sample_exponential_race(&probs)?,
        };
        u32::try_from(id).map_err(|_| FishError::InvalidShape("sample ID overflows u32"))
    }

    pub fn generate_columns(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        params: SamplingParams,
    ) -> Result<Vec<Vec<u32>>> {
        self.generate_columns_with_cancel(prompt, max_new_tokens, params, &mut || false)
    }

    pub fn generate_columns_with_cancel(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        params: SamplingParams,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<Vec<u32>>> {
        params.validate()?;
        let channels = self.config.num_codebooks + 1;
        if prompt.channels.len() != channels
            || prompt.channels[0].is_empty()
            || prompt.vq_positions.len() != prompt.channels[0].len()
            || prompt
                .channels
                .iter()
                .any(|c| c.len() != prompt.channels[0].len())
        {
            return Err(FishError::InvalidShape("invalid Naive prompt"));
        }
        let prompt_len = prompt.channels[0].len();
        if prompt_len >= self.config.max_seq_len {
            return Err(FishError::InvalidPosition {
                position: prompt_len,
                capacity: self.config.max_seq_len,
            });
        }
        let remaining = self.config.max_seq_len - prompt_len;
        let limit = if max_new_tokens == 0 {
            remaining
        } else {
            max_new_tokens.min(remaining)
        };
        let config = self.config.clone();
        let factory = TransformerIrFactory::new(&config)?;
        let mut slow_cache = NamedTensors::new();
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); channels];
        for i in 0..limit {
            if is_cancelled() {
                return Err(FishError::InvalidTransition("synthesis cancelled"));
            }
            let graph = if i == 0 {
                factory.slow_prefill(prompt_len)?
            } else {
                factory.slow_decode(prompt_len + i)?
            };
            let mut inputs = NamedTensors::new();
            let tokens: Vec<i32> = if i == 0 {
                checked_i32_tokens(prompt.channels.iter().flatten().copied())?
            } else {
                checked_i32_tokens(generated.iter().map(|c| *c.last().unwrap()))?
            };
            inputs.insert("tokens".into(), TensorData::I32(tokens));
            inputs.insert(
                "semantic_membership_lut".into(),
                TensorData::I32(self.semantic_lut.clone()),
            );
            let output = run_weighted_graph(&self.weights, &graph, &inputs, &mut slow_cache)?;
            let TensorData::F32(semantic_logits) = output
                .get("logits")
                .ok_or(FishError::InvalidShape("Naive semantic logits missing"))?
            else {
                return Err(FishError::InvalidShape(
                    "Naive semantic logits dtype invalid",
                ));
            };
            let main = SamplingParams {
                temperature: 0.1,
                top_p: 0.1,
                repetition_penalty: 1.0,
            };
            let semantic = self.draw(semantic_logits, &[], main)?;
            let TensorData::F32(codebook_logits) = output
                .get("codebook_logits")
                .ok_or(FishError::InvalidShape("Naive acoustic logits missing"))?
            else {
                return Err(FishError::InvalidShape(
                    "Naive acoustic logits dtype invalid",
                ));
            };
            if codebook_logits.len() != self.config.num_codebooks * self.config.codebook_size {
                return Err(FishError::InvalidShape(
                    "Naive acoustic logits shape invalid",
                ));
            }
            let mut row = vec![semantic];
            for cb in 0..self.config.num_codebooks {
                if is_cancelled() {
                    return Err(FishError::InvalidTransition("synthesis cancelled"));
                }
                let prior = if i == 0 {
                    Vec::new()
                } else {
                    let mut window: Vec<usize> = generated[cb + 1]
                        .iter()
                        .skip(1)
                        .rev()
                        .take(16)
                        .copied()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .map(|x| x as usize)
                        .collect();
                    window.resize(16, 0);
                    window
                };
                let stride = self.config.codebook_size;
                row.push(self.draw(
                    &codebook_logits[cb * stride..(cb + 1) * stride],
                    &prior,
                    params,
                )?);
            }
            for (dst, token) in generated.iter_mut().zip(row) {
                dst.push(token);
            }
            if semantic == self.eos_id {
                break;
            }
        }
        Ok(generated)
    }
}

impl SemanticGenerator for CpuWeightedNaive {
    fn begin_sample(&mut self) -> Result<()> {
        self.rng = SessionRng::new(self.seed);
        Ok(())
    }
    fn generate(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        sampling: SamplingParams,
    ) -> Result<Vec<Vec<u32>>> {
        self.generate_columns(prompt, max_new_tokens, sampling)
    }

    fn generate_with_cancel(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        sampling: SamplingParams,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<Vec<u32>>> {
        self.generate_columns_with_cancel(prompt, max_new_tokens, sampling, is_cancelled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TransformerDims;

    fn fixture() -> (FishModelConfig, NamedTensors) {
        let c = FishModelConfig {
            kind: ModelKind::Naive,
            slow: TransformerDims {
                dim: 16,
                n_heads: 4,
                n_kv_heads: 2,
                layers: 1,
                intermediate_size: 32,
                qkv_bias: false,
            },
            fast: None,
            vocab_size: 64,
            num_codebooks: 3,
            codebook_size: 16,
            max_seq_len: 16,
            rope_base: 10_000.0,
            norm_eps: 1e-5,
            tie_word_embeddings: true,
            share_codebook_embeddings: true,
        };
        let graph = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_prefill(2)
            .unwrap();
        let mut weights = NamedTensors::new();
        for (name, ty) in graph.weight_requirements() {
            let len: usize = ty.shape.iter().product();
            let values = if name.ends_with("norm.weight") || name == "norm.weight" {
                vec![1.0; len]
            } else {
                vec![0.0; len]
            };
            weights.insert(name, TensorData::F32(values));
        }
        (c, weights)
    }

    fn prompt() -> EncodedConversation {
        EncodedConversation {
            channels: vec![vec![1, 2], vec![0, 0], vec![0, 0], vec![0, 0]],
            vq_positions: vec![false, false],
        }
    }

    #[test]
    fn naive_graph_outputs_independent_codebook_logits() {
        let (c, _) = fixture();
        let graph = TransformerIrFactory::new(&c)
            .unwrap()
            .slow_prefill(2)
            .unwrap();
        assert_eq!(
            graph.tensor(*graph.outputs.last().unwrap()).unwrap().shape,
            [1, 1, 48]
        );
        let weights = graph.weight_requirements();
        assert_eq!(weights["codebook_output.weight"].shape, [48, 16]);
        assert_eq!(weights["codebook_norm.weight"].shape, [16]);
        assert!(!weights.contains_key("fast_output.weight"));
    }

    #[test]
    fn naive_weighted_generation_greedy_and_eos() {
        let (config, weights) = fixture();
        let mut naive =
            CpuWeightedNaive::new(config, weights, vec![0; 64], 63, 123, SelectionMode::Greedy)
                .unwrap();
        let generated = naive
            .generate_columns(&prompt(), 3, SamplingParams::default())
            .unwrap();
        assert_eq!(generated, vec![vec![0; 3]; 4]);
        let mut eos = CpuWeightedNaive::new(
            naive.config.clone(),
            naive.weights.clone(),
            vec![0; 64],
            0,
            123,
            SelectionMode::Greedy,
        )
        .unwrap();
        assert_eq!(
            eos.generate_columns(&prompt(), 3, SamplingParams::default())
                .unwrap(),
            vec![vec![0]; 4]
        );
    }

    #[test]
    fn cancellation_inside_naive_acoustic_loop_is_observed() {
        let (config, weights) = fixture();
        let mut naive =
            CpuWeightedNaive::new(config, weights, vec![0; 64], 63, 123, SelectionMode::Greedy)
                .unwrap();
        let mut polls = 0;
        let error = naive
            .generate_columns_with_cancel(&prompt(), 3, SamplingParams::default(), &mut || {
                polls += 1;
                polls == 3
            })
            .unwrap_err();
        assert_eq!(error, FishError::InvalidTransition("synthesis cancelled"));
        assert_eq!(polls, 3);
        naive.begin_sample().unwrap();
        assert_eq!(
            naive
                .generate_columns(&prompt(), 1, SamplingParams::default())
                .unwrap()[0]
                .len(),
            1
        );
    }

    #[test]
    fn naive_rejects_wrong_model_and_missing_weights() {
        let (mut c, mut weights) = fixture();
        weights.remove("codebook_output.weight");
        assert!(CpuWeightedNaive::new(
            c.clone(),
            weights,
            vec![0; 64],
            63,
            1,
            SelectionMode::Greedy
        )
        .is_err());
        c.kind = ModelKind::DualAr;
        assert!(CpuWeightedNaive::new(
            c,
            NamedTensors::new(),
            vec![0; 64],
            63,
            1,
            SelectionMode::Greedy
        )
        .is_err());
    }
}
