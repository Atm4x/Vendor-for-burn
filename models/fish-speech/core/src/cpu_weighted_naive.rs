use crate::config::{FishModelConfig, ModelKind};
use crate::conversation::EncodedConversation;
use crate::cpu_ir::{NamedTensors, TensorData};
use crate::cpu_weighted_ar::{checked_i32_tokens, run_weighted_graph, SelectionMode};
use crate::error::{FishError, Result};
use crate::naive_generation::{generate_naive_ar, NaiveArBackend, NaiveArForward};
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
        let factory = TransformerIrFactory::new(&self.config)?;
        let mut backend = CpuNaiveBackend {
            factory,
            weights: &self.weights,
            semantic_lut: &self.semantic_lut,
            slow_cache: NamedTensors::new(),
        };
        let mode = self.mode;
        let rng = &mut self.rng;
        generate_naive_ar(
            &self.config,
            &mut backend,
            prompt,
            max_new_tokens,
            self.eos_id,
            params,
            &mut |logits, history, params| {
                let probabilities = logits_to_probs(logits, history, params)?;
                let token = match mode {
                    SelectionMode::Greedy => greedy(&probabilities)?,
                    SelectionMode::Sample => rng.sample_exponential_race(&probabilities)?,
                };
                u32::try_from(token).map_err(|_| FishError::InvalidShape("sample ID overflows u32"))
            },
            is_cancelled,
        )
    }
}

struct CpuNaiveBackend<'a> {
    factory: TransformerIrFactory<'a>,
    weights: &'a NamedTensors,
    semantic_lut: &'a [i32],
    slow_cache: NamedTensors,
}

impl CpuNaiveBackend<'_> {
    fn forward(
        &mut self,
        graph: crate::graph_ir::Graph,
        tokens: Vec<i32>,
    ) -> Result<NaiveArForward> {
        let mut inputs = NamedTensors::new();
        inputs.insert("tokens".into(), TensorData::I32(tokens));
        inputs.insert(
            "semantic_membership_lut".into(),
            TensorData::I32(self.semantic_lut.to_vec()),
        );
        let output = run_weighted_graph(self.weights, &graph, &inputs, &mut self.slow_cache)?;
        let semantic_logits = match output.get("logits") {
            Some(TensorData::F32(v)) => v.clone(),
            _ => return Err(FishError::InvalidShape("Naive semantic logits missing")),
        };
        let acoustic_logits = match output.get("codebook_logits") {
            Some(TensorData::F32(v)) => v.clone(),
            _ => return Err(FishError::InvalidShape("Naive acoustic logits missing")),
        };
        Ok(NaiveArForward {
            semantic_logits,
            acoustic_logits,
        })
    }
}

impl NaiveArBackend for CpuNaiveBackend<'_> {
    fn slow_prefill(&mut self, prompt: &EncodedConversation) -> Result<NaiveArForward> {
        self.forward(
            self.factory.slow_prefill(prompt.length())?,
            checked_i32_tokens(prompt.channels.iter().flatten().copied())?,
        )
    }

    fn slow_decode(
        &mut self,
        previous_column: &[u32],
        active_len: usize,
    ) -> Result<NaiveArForward> {
        self.forward(
            self.factory.slow_decode(active_len)?,
            checked_i32_tokens(previous_column.iter().copied())?,
        )
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
