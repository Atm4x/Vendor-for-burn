use crate::ar_generation::{
    generate_dual_ar, DualArBackend, DualArTrace, SlowArForward, SlowArLogits,
};
use crate::config::{FishModelConfig, ModelKind};
use crate::conversation::EncodedConversation;
use crate::cpu_ir::{CpuIrInterpreter, NamedTensors, TensorData};
use crate::error::{FishError, Result};
use crate::graph_ir::{Graph, Layout, Origin};
use crate::sampling::{greedy, logits_to_probs, SamplingParams, SessionRng};
use crate::synthesis::SemanticGenerator;
use crate::transformer_ir::TransformerIrFactory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    Sample,
    Greedy,
}

pub type CpuArTrace = DualArTrace;

pub struct CpuWeightedDualAr {
    config: FishModelConfig,
    weights: NamedTensors,
    semantic_lut: Vec<i32>,
    semantic_begin_id: u32,
    eos_id: u32,
    seed: u64,
    rng: SessionRng,
    mode: SelectionMode,
    trace: CpuArTrace,
}

impl CpuWeightedDualAr {
    pub fn new(
        config: FishModelConfig,
        weights: NamedTensors,
        semantic_lut: Vec<i32>,
        semantic_begin_id: u32,
        eos_id: u32,
        seed: u64,
        mode: SelectionMode,
    ) -> Result<Self> {
        config.validate()?;
        if config.kind != ModelKind::DualAr {
            return Err(FishError::Unsupported(
                "CPU weighted Dual-AR requires DualAr config",
            ));
        }
        if semantic_lut.len() != config.vocab_size
            || semantic_lut.iter().any(|&v| v != 0 && v != 1)
            || eos_id as usize >= config.vocab_size
        {
            return Err(FishError::InvalidConfig(
                "invalid semantic LUT or EOS token",
            ));
        }
        let factory = TransformerIrFactory::new(&config)?;
        for graph in [
            factory.slow_prefill(1)?,
            factory.fast_decode(0)?,
            factory.fast_decode(config.num_codebooks.saturating_sub(1))?,
        ] {
            for (key, ty) in graph.weight_requirements() {
                let value = weights.get(&key).ok_or(FishError::InvalidShape(
                    "missing weighted AR checkpoint tensor",
                ))?;
                if !matches!(value, TensorData::F32(data) if data.len() == ty.shape.iter().product::<usize>())
                {
                    return Err(FishError::InvalidShape(
                        "AR checkpoint tensor shape or dtype mismatch",
                    ));
                }
            }
        }
        Ok(Self {
            config,
            weights,
            semantic_lut,
            semantic_begin_id,
            eos_id,
            seed,
            rng: SessionRng::new(seed),
            mode,
            trace: CpuArTrace::default(),
        })
    }

    pub fn trace(&self) -> &CpuArTrace {
        &self.trace
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
        let mut backend = CpuDualArBackend {
            factory: TransformerIrFactory::new(&self.config)?,
            weights: &self.weights,
            semantic_lut: &self.semantic_lut,
            slow_cache: NamedTensors::new(),
            fast_cache: NamedTensors::new(),
        };
        let mode = self.mode;
        let rng = &mut self.rng;
        let output = generate_dual_ar(
            &self.config,
            &mut backend,
            prompt,
            max_new_tokens,
            self.semantic_begin_id,
            self.eos_id,
            params,
            &mut |logits, history, params| {
                let probabilities = logits_to_probs(logits, history, params)?;
                let token = match mode {
                    SelectionMode::Greedy => greedy(&probabilities)?,
                    SelectionMode::Sample => rng.sample_exponential_race(&probabilities)?,
                };
                u32::try_from(token)
                    .map_err(|_| FishError::InvalidShape("generated token overflows u32"))
            },
            &mut self.trace,
            is_cancelled,
        )?;
        Ok(output)
    }
}

struct CpuDualArBackend<'a> {
    factory: TransformerIrFactory<'a>,
    weights: &'a NamedTensors,
    semantic_lut: &'a [i32],
    slow_cache: NamedTensors,
    fast_cache: NamedTensors,
}

impl CpuDualArBackend<'_> {
    fn slow(
        &mut self,
        tokens: Vec<i32>,
        active_len: usize,
        prefill: bool,
    ) -> Result<SlowArForward<TensorData>> {
        let graph = if prefill {
            self.factory.slow_prefill(active_len)?
        } else {
            self.factory.slow_decode(active_len)?
        };
        let mut input = NamedTensors::new();
        input.insert("tokens".into(), TensorData::I32(tokens));
        input.insert(
            "semantic_membership_lut".into(),
            TensorData::I32(self.semantic_lut.to_vec()),
        );
        let result = run_weighted_graph(self.weights, &graph, &input, &mut self.slow_cache)?;
        let logits = match result.get("logits") {
            Some(TensorData::F32(values)) => values.clone(),
            _ => return Err(FishError::InvalidShape("slow logits missing")),
        };
        let fast_context = result
            .get("fast_context")
            .ok_or(FishError::InvalidShape("slow Fast-AR context missing"))?
            .clone();
        Ok(SlowArForward {
            logits: SlowArLogits::Host(logits),
            fast_context,
        })
    }
}

impl DualArBackend for CpuDualArBackend<'_> {
    type FastContext = TensorData;

    fn slow_prefill(&mut self, prompt: &EncodedConversation) -> Result<SlowArForward<TensorData>> {
        self.slow(
            checked_i32_tokens(prompt.channels.iter().flatten().copied())?,
            prompt.length(),
            true,
        )
    }

    fn slow_decode(
        &mut self,
        previous_column: &[u32],
        active_len: usize,
    ) -> Result<SlowArForward<TensorData>> {
        self.slow(
            checked_i32_tokens(previous_column.iter().copied())?,
            active_len,
            false,
        )
    }

    fn reset_fast(&mut self) -> Result<()> {
        self.fast_cache.clear();
        Ok(())
    }

    fn prime_fast(&mut self, context: TensorData) -> Result<()> {
        let graph = self.factory.fast_decode(0)?;
        let mut input = NamedTensors::new();
        input.insert("fast_context".into(), context);
        run_weighted_graph(self.weights, &graph, &input, &mut self.fast_cache)?;
        Ok(())
    }

    fn decode_fast(&mut self, codebook_index: usize, previous_code: u32) -> Result<Vec<f32>> {
        let graph = self.factory.fast_decode(codebook_index)?;
        let mut input = NamedTensors::new();
        input.insert(
            "previous_acoustic_code".into(),
            TensorData::I32(checked_i32_tokens([previous_code])?),
        );
        let result = run_weighted_graph(self.weights, &graph, &input, &mut self.fast_cache)?;
        match result.get("fast_logits") {
            Some(TensorData::F32(values)) => Ok(values.clone()),
            _ => Err(FishError::InvalidShape("Fast AR logits missing")),
        }
    }
}

impl SemanticGenerator for CpuWeightedDualAr {
    fn begin_sample(&mut self) -> Result<()> {
        self.rng = SessionRng::new(self.seed);
        self.trace = CpuArTrace::default();
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

pub(crate) fn checked_i32_tokens(tokens: impl IntoIterator<Item = u32>) -> Result<Vec<i32>> {
    tokens
        .into_iter()
        .map(|value| {
            i32::try_from(value).map_err(|_| FishError::InvalidShape("AR token ID overflows i32"))
        })
        .collect()
}

pub(crate) fn run_weighted_graph(
    weights: &NamedTensors,
    graph: &Graph,
    inputs: &NamedTensors,
    caches: &mut NamedTensors,
) -> Result<NamedTensors> {
    let mut bindings = NamedTensors::new();
    for value in &graph.values {
        match value.origin {
            Origin::Weight { .. } => {
                bindings.insert(
                    value.name.clone(),
                    weights
                        .get(&value.name)
                        .ok_or(FishError::InvalidShape("missing AR weight binding"))?
                        .clone(),
                );
            }
            Origin::Input => {
                let tensor = if value.ty.layout == Layout::CacheBhtd {
                    caches.entry(value.name.clone()).or_insert_with(|| {
                        TensorData::F32(vec![0.0; value.ty.shape.iter().product()])
                    })
                } else {
                    inputs
                        .get(&value.name)
                        .ok_or(FishError::InvalidShape("missing AR runtime input"))?
                };
                bindings.insert(value.name.clone(), tensor.clone());
            }
            Origin::Node(_) => {}
        }
    }
    let output = CpuIrInterpreter::run(graph, &bindings)?;
    for value in &graph.values {
        if value.ty.layout == Layout::CacheBhtd && matches!(value.origin, Origin::Input) {
            let new_name = value.name.replace("_cache", "_updated");
            if let Some(updated) = output.get(&new_name) {
                caches.insert(value.name.clone(), updated.clone());
            } else {
                return Err(FishError::InvalidShape("missing updated KV cache output"));
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TransformerDims;
    use crate::conversation::TokenEncoder;
    use crate::synthesis::{synthesize, AcousticDecoder, SynthesisRequest};
    use crate::tokenizer::SpecialTokens;

    fn config() -> FishModelConfig {
        let dims = TransformerDims {
            dim: 16,
            n_heads: 4,
            n_kv_heads: 2,
            layers: 1,
            intermediate_size: 32,
            qkv_bias: false,
        };
        FishModelConfig {
            kind: ModelKind::DualAr,
            slow: dims.clone(),
            fast: Some(dims),
            vocab_size: 64,
            num_codebooks: 3,
            codebook_size: 32,
            max_seq_len: 64,
            rope_base: 10_000.0,
            norm_eps: 1e-5,
            tie_word_embeddings: true,
            share_codebook_embeddings: true,
        }
    }

    fn weights(config: &FishModelConfig, nonzero: bool) -> NamedTensors {
        let factory = TransformerIrFactory::new(config).unwrap();
        let mut out = NamedTensors::new();
        for graph in [
            factory.slow_prefill(2).unwrap(),
            factory.fast_decode(0).unwrap(),
            factory.fast_decode(1).unwrap(),
        ] {
            for (key, ty) in graph.weight_requirements() {
                out.entry(key.clone()).or_insert_with(|| {
                    TensorData::F32(
                        (0..ty.shape.iter().product::<usize>())
                            .map(|i| {
                                if key.ends_with("norm.weight") || key == "norm.weight" {
                                    1.0
                                } else if nonzero {
                                    ((i * 11 % 37) as f32 - 18.0) * 0.004
                                } else {
                                    0.0
                                }
                            })
                            .collect(),
                    )
                });
            }
        }
        out
    }

    fn new_generator(nonzero: bool, eos: u32, mode: SelectionMode) -> CpuWeightedDualAr {
        let cfg = config();
        let mut lut = vec![0; cfg.vocab_size];
        lut[32..64].fill(1);
        CpuWeightedDualAr::new(cfg.clone(), weights(&cfg, nonzero), lut, 32, eos, 42, mode).unwrap()
    }

    fn prompt() -> EncodedConversation {
        EncodedConversation {
            channels: vec![vec![1, 3], vec![0, 0], vec![0, 0], vec![0, 0]],
            vq_positions: vec![false, false],
        }
    }

    #[test]
    fn full_weight_bound_dual_ar_prefill_and_decode() {
        let mut generator = new_generator(true, 63, SelectionMode::Greedy);
        let result = generator
            .generate_columns(&prompt(), 3, SamplingParams::default())
            .unwrap();
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|c| c.len() == 3));
        for i in 0..3 {
            assert_eq!(result[1][i], result[0][i].saturating_sub(32));
            assert!(result.iter().skip(1).all(|channel| channel[i] < 32));
        }
        assert_eq!(generator.trace().slow_active_lengths, [2, 3, 4]);
        assert_eq!(
            generator.trace().fast_active_lengths,
            [1, 2, 3, 1, 2, 3, 1, 2, 3]
        );
        assert_eq!(generator.trace().fast_resets, 3);
    }

    #[test]
    fn cancellation_during_fast_codebook_generation_stops_before_next_gpu_analogue_step() {
        let mut generator = new_generator(true, 63, SelectionMode::Greedy);
        let mut calls = 0;
        let error = generator
            .generate_columns_with_cancel(&prompt(), 3, SamplingParams::default(), &mut || {
                calls += 1;
                calls == 2
            })
            .unwrap_err();
        assert_eq!(error, FishError::InvalidTransition("synthesis cancelled"));
        assert_eq!(calls, 2);
        assert_eq!(generator.trace().fast_active_lengths, [1]);
        generator.begin_sample().unwrap();
        let recovered = generator
            .generate_columns(&prompt(), 2, SamplingParams::default())
            .unwrap();
        assert_eq!(recovered[0].len(), 2);
    }

    #[test]
    fn zero_weights_produce_deterministic_greedy_and_eos() {
        let mut generator = new_generator(false, 0, SelectionMode::Greedy);
        let result = generator
            .generate_columns(&prompt(), 5, SamplingParams::default())
            .unwrap();
        assert_eq!(result, vec![vec![0], vec![0], vec![0], vec![0]]);
        assert_eq!(generator.trace().fast_resets, 1);
    }

    #[test]
    fn session_reset_does_not_leak_slow_or_fast_caches() {
        let mut generator = new_generator(true, 63, SelectionMode::Greedy);
        let first = generator
            .generate_columns(&prompt(), 2, SamplingParams::default())
            .unwrap();
        let first_trace = generator.trace().clone();
        let second = generator
            .generate_columns(&prompt(), 2, SamplingParams::default())
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(generator.trace(), &first_trace);
    }

    #[test]
    fn repeated_sampling_seed_reproduces_same_acoustic_channels() {
        let mut generator = new_generator(true, 63, SelectionMode::Sample);
        generator.begin_sample().unwrap();
        let a = generator
            .generate_columns(&prompt(), 2, SamplingParams::default())
            .unwrap();
        generator.begin_sample().unwrap();
        let b = generator
            .generate_columns(&prompt(), 2, SamplingParams::default())
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_fast_dimension_uses_projected_slow_context() {
        let mut cfg = config();
        cfg.fast = Some(TransformerDims {
            dim: 8,
            n_heads: 2,
            n_kv_heads: 1,
            layers: 1,
            intermediate_size: 16,
            qkv_bias: false,
        });
        let tensor_map = weights(&cfg, true);
        assert!(tensor_map.contains_key("fast_project_in.weight"));
        assert!(tensor_map.contains_key("fast_project_in.bias"));
        let mut generator = CpuWeightedDualAr::new(
            cfg.clone(),
            tensor_map,
            vec![0; cfg.vocab_size],
            32,
            63,
            123,
            SelectionMode::Greedy,
        )
        .unwrap();
        let output = generator
            .generate_columns(&prompt(), 2, SamplingParams::default())
            .unwrap();
        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|row| row.len() == 2));
        assert_eq!(generator.trace().fast_active_lengths, [1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn reject_missing_or_misshapen_weight_or_prompt() {
        let cfg = config();
        let lut = vec![0; cfg.vocab_size];
        let mut w = weights(&cfg, true);
        w.remove("layers.0.attention.wqkv.weight");
        assert!(CpuWeightedDualAr::new(
            cfg.clone(),
            w,
            lut.clone(),
            32,
            63,
            1,
            SelectionMode::Greedy
        )
        .is_err());
        let mut w = weights(&cfg, true);
        w.insert("fast_norm.weight".into(), TensorData::F32(vec![1.0]));
        assert!(
            CpuWeightedDualAr::new(cfg.clone(), w, lut, 32, 63, 1, SelectionMode::Greedy).is_err()
        );
        let mut generator = new_generator(false, 63, SelectionMode::Greedy);
        let mut bad = prompt();
        bad.channels[2].pop();
        assert!(generator
            .generate_columns(&bad, 2, SamplingParams::default())
            .is_err());
    }

    struct SingleTokenEncoder;
    impl TokenEncoder for SingleTokenEncoder {
        fn encode(&self, _: &str) -> Result<Vec<u32>> {
            Ok(vec![1])
        }
    }
    struct CountingDecoder;
    impl AcousticDecoder for CountingDecoder {
        fn decode(&mut self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
            assert_eq!(codes.len(), 3);
            assert!(codes.iter().all(|channel| channel.len() == 2));
            Ok(vec![0.25, 0.5])
        }
    }

    #[test]
    fn weighted_semantic_generator_connects_to_synthesis_pipeline() {
        let mut generator = new_generator(false, 63, SelectionMode::Greedy);
        let mut decoder = CountingDecoder;
        let request = SynthesisRequest {
            text: "hello",
            references: &[],
            num_samples: 1,
            num_codebooks: 3,
            iterative_prompt: false,
            chunk_length: 100,
            max_context_length: 2048,
            max_new_tokens: 3,
            sampling: SamplingParams::default(),
        };
        let results = synthesize(
            &SingleTokenEncoder,
            &SpecialTokens::new(2).unwrap(),
            &request,
            &mut generator,
            &mut decoder,
            || false,
        )
        .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pcm, [0.25, 0.5]);
    }
}
