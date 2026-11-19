use std::{collections::BTreeMap, sync::Arc};

use cudarc::driver::CudaStream;
use fish_speech_core::{
    ar_generation::{DualArBackend, DualArTrace, SlowArForward, SlowArLogits, generate_dual_ar},
    ar_stream_plan::{ArStreamPhase, ArStreamPlan},
    config::{FishModelConfig, ModelKind},
    conversation::EncodedConversation,
    error::{FishError, Result},
    graph::Stage,
    graph_ir::{Graph, ScalarType, TensorType},
    sampling::{SamplingParams, SessionRng, logits_to_probs},
    shape_profiles::ArProfileGroup,
    transformer_ir::TransformerIrFactory,
    trtx_blueprint::{Binding, TrtBlueprint},
};
use trtx::Logger;

use crate::{
    cuda_fast_sampling::{
        CudaFastSampler, allocate_token, parse_column, upload_column, upload_history_mask,
    },
    cuda_semantic_sampling::{
        CudaSemanticSampler, allocate_semantic_result, parse_semantic_result,
        upload_semantic_history,
    },
    cuda_stream::{CudaBuffer, CudaScratchPool},
    cuda_trtx::{CachedArDispatchConfig, WeightedCudaArSession, with_resident_cuda_ar_session},
    device_memory::DeviceMemoryBudget,
    dynamic_decode::{
        KV_MASK, ROPE_COS, ROPE_SIN, decode_control_bytes, parameterize_slow_decode,
        positioned_slow_decode,
    },
    trtx_cache::EngineRegistry,
    weights::LoadedWeights,
};

pub(crate) struct PreparedArGraph {
    pub(crate) plan: ArStreamPlan,
    pub(crate) outputs: BTreeMap<String, Binding>,
}

impl PreparedArGraph {
    pub(crate) fn from_graph(graph: Graph) -> Result<Self> {
        let outputs = graph
            .outputs
            .iter()
            .map(|id| (graph.values[id.0].name.clone(), Binding::Value(*id)))
            .collect();
        Ok(Self {
            plan: ArStreamPlan::from_graph(&graph)?,
            outputs,
        })
    }
}

pub struct PreparedDualArPlans {
    slow_prefill: PreparedArGraph,
    slow_decode: BTreeMap<usize, PreparedArGraph>,
    slow_decode_profiles: ArProfileGroup,
    fast_decode: Vec<PreparedArGraph>,
    prompt_len: usize,
    max_new_tokens: usize,
    config: FishModelConfig,
}

impl PreparedDualArPlans {
    pub fn new(config: &FishModelConfig, prompt_len: usize, max_new_tokens: usize) -> Result<Self> {
        if config.kind != ModelKind::DualAr {
            return Err(FishError::Unsupported(
                "CUDA DualAR plan requires DualAr model",
            ));
        }
        let factory = TransformerIrFactory::new(config)?;
        if prompt_len == 0 || prompt_len >= config.max_seq_len || max_new_tokens == 0 {
            return Err(FishError::InvalidShape(
                "invalid CUDA DualAR generation length",
            ));
        }
        if max_new_tokens > config.max_seq_len - prompt_len {
            return Err(FishError::InvalidShape(
                "CUDA DualAR profile exceeds sequence length",
            ));
        }
        let slow_prefill = PreparedArGraph::from_graph(factory.slow_prefill(prompt_len)?)?;
        let slow_decode_profiles = ArProfileGroup::new(Stage::SlowDecode, config.max_seq_len, 32)?;
        let mut slow_decode = BTreeMap::new();
        if max_new_tokens > 1 {
            let first_active = prompt_len + 1;
            let last_active = prompt_len + max_new_tokens - 1;
            for bucket in &slow_decode_profiles.buckets {
                if bucket.max < first_active || bucket.min > last_active {
                    continue;
                }
                let mut profile = PreparedArGraph::from_graph(factory.slow_decode(bucket.max)?)?;
                parameterize_slow_decode(&mut profile.plan, bucket.max)?;
                slow_decode.insert(bucket.max, profile);
            }
        }
        let fast_decode = (0..config.num_codebooks)
            .map(|step| PreparedArGraph::from_graph(factory.fast_decode(step)?))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config: config.clone(),
            slow_prefill,
            slow_decode,
            slow_decode_profiles,
            fast_decode,
            prompt_len,
            max_new_tokens,
        })
    }

    pub(crate) fn ar_plans(&self) -> Vec<&ArStreamPlan> {
        let mut plans = vec![&self.slow_prefill.plan];
        plans.extend(self.slow_decode.values().map(|item| &item.plan));
        plans.extend(self.fast_decode.iter().map(|item| &item.plan));
        plans
    }
}

struct CudaDualArBackend<'a, F>
where
    F: FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()>,
{
    session: &'a mut WeightedCudaArSession<F>,
    plans: &'a PreparedDualArPlans,
    semantic_lut: CudaBuffer,
    fast_sampler: CudaFastSampler,
    semantic_sampler: CudaSemanticSampler,
    slow_logits: Option<CudaBuffer>,
    semantic_steps: u64,
    seed: u64,
    fast_columns: u64,
}

impl<'a, F> CudaDualArBackend<'a, F>
where
    F: FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()>,
{
    pub fn new(
        session: &'a mut WeightedCudaArSession<F>,
        plans: &'a PreparedDualArPlans,
        semantic_lut: &[i32],
        seed: u64,
    ) -> Result<Self> {
        let vocab = expected_runtime_input(&plans.slow_prefill.plan, "semantic_membership_lut")?;
        if vocab.shape != [semantic_lut.len()] || vocab.dtype != ScalarType::I32 {
            return Err(FishError::InvalidShape(
                "CUDA semantic LUT dtype or vocabulary mismatch",
            ));
        }
        let bytes = semantic_lut
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let lut = session.upload_runtime_input(&vocab, &bytes)?;
        let fast_sampler = CudaFastSampler::new(session.device_mut().context())?;
        let semantic_sampler = CudaSemanticSampler::new(session.device_mut().context())?;
        Ok(Self {
            session,
            plans,
            semantic_lut: lut,
            fast_sampler,
            semantic_sampler,
            slow_logits: None,
            semantic_steps: 0,
            seed,
            fast_columns: 0,
        })
    }

    fn slow(
        &mut self,
        graph: &PreparedArGraph,
        tokens: &[u32],
        decode_position: Option<(usize, usize)>,
    ) -> Result<SlowArForward<CudaBuffer>> {
        let input = self.tokens(&graph.plan, "tokens", tokens)?;
        let mut runtime = BTreeMap::new();
        runtime.insert("tokens".to_owned(), input);
        runtime.insert(
            "semantic_membership_lut".to_owned(),
            self.semantic_lut.clone(),
        );
        if let Some((active_len, bucket_end)) = decode_position {
            let (index, _) = self.plans.slow_decode_profiles.locate(1, active_len)?;
            if self.plans.slow_decode_profiles.buckets[index].max != bucket_end {
                return Err(FishError::InvalidShape(
                    "dynamic Slow decode bucket mismatch",
                ));
            }
            let (cos, sin, mask) = decode_control_bytes(
                active_len,
                bucket_end,
                self.plans.config.slow.head_dim(),
                self.plans.config.rope_base,
            )?;
            for (name, data) in [(ROPE_COS, cos), (ROPE_SIN, sin), (KV_MASK, mask)] {
                let ty = expected_runtime_input(&graph.plan, name)?;
                runtime.insert(
                    name.to_owned(),
                    self.session.upload_runtime_input(&ty, &data)?,
                );
            }
        }
        let result = self.session.enqueue(&graph.plan, &runtime)?;
        let context = named_output(graph, &result, "fast_context")?;
        let logits = named_output(graph, &result, "logits")?;
        let ty = output_type(graph, "logits")?;
        if ty.dtype != ScalarType::F32 || ty.shape != [1, 1, self.plans.config.vocab_size] {
            return Err(FishError::InvalidShape(
                "CUDA semantic logits geometry mismatch",
            ));
        }
        self.slow_logits = Some(logits);
        Ok(SlowArForward {
            logits: SlowArLogits::DevicePending,
            fast_context: context,
        })
    }

    fn tokens(&mut self, plan: &ArStreamPlan, name: &str, tokens: &[u32]) -> Result<CudaBuffer> {
        let ty = expected_runtime_input(plan, name)?;
        if ty.dtype != ScalarType::I32 || ty.shape.iter().product::<usize>() != tokens.len() {
            return Err(FishError::InvalidShape("CUDA token tensor shape mismatch"));
        }
        let bytes = tokens
            .iter()
            .map(|&value| {
                i32::try_from(value)
                    .map(i32::to_le_bytes)
                    .map_err(|_| FishError::InvalidShape("CUDA token exceeds i32"))
            })
            .collect::<Result<Vec<_>>>()?
            .concat();
        self.session.upload_runtime_input(&ty, &bytes)
    }

    fn read_logits(
        &mut self,
        graph: &PreparedArGraph,
        outputs: &BTreeMap<Binding, CudaBuffer>,
        name: &str,
    ) -> Result<Vec<f32>> {
        let tensor = named_output(graph, outputs, name)?;
        let ty = output_type(graph, name)?;
        if ty.dtype != ScalarType::F32 || ty.shape.first() != Some(&1) {
            return Err(FishError::InvalidShape(
                "CUDA AR logits must be F32 batch one",
            ));
        }
        let bytes = self.session.read_output_bytes(&tensor)?;
        if bytes.len() % 4 != 0 || bytes.len() / 4 != ty.shape.iter().product::<usize>() {
            return Err(FishError::InvalidShape("CUDA logits output size mismatch"));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
}

impl<F> DualArBackend for CudaDualArBackend<'_, F>
where
    F: FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()>,
{
    type FastContext = CudaBuffer;

    fn slow_prefill(&mut self, prompt: &EncodedConversation) -> Result<SlowArForward<CudaBuffer>> {
        if prompt.length() != self.plans.prompt_len {
            return Err(FishError::InvalidShape(
                "CUDA prompt differs from compiled profile",
            ));
        }
        let tokens = prompt
            .channels
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        self.slow(&self.plans.slow_prefill, &tokens, None)
    }

    fn slow_decode(
        &mut self,
        previous_column: &[u32],
        active_len: usize,
    ) -> Result<SlowArForward<CudaBuffer>> {
        let (index, _) = self.plans.slow_decode_profiles.locate(1, active_len)?;
        let bucket_end = self.plans.slow_decode_profiles.buckets[index].max;
        let template = self
            .plans
            .slow_decode
            .get(&bucket_end)
            .ok_or(FishError::InvalidShape(
                "CUDA Slow decode bucket was not prepared",
            ))?;
        let plan = PreparedArGraph {
            plan: positioned_slow_decode(&template.plan, active_len, bucket_end)?,
            outputs: template.outputs.clone(),
        };
        self.slow(&plan, previous_column, Some((active_len, bucket_end)))
    }

    fn sample_semantic(
        &mut self,
        logits: &SlowArLogits,
        history: &[usize],
        params: SamplingParams,
        eos_id: u32,
        _host_sample: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
    ) -> Result<(u32, bool)> {
        if !matches!(logits, SlowArLogits::DevicePending) {
            return Err(FishError::InvalidTransition(
                "CUDA semantic logits were not device resident",
            ));
        }
        let tensor = self.slow_logits.take().ok_or(FishError::InvalidTransition(
            "CUDA semantic logits were not produced",
        ))?;
        let vocabulary = self.plans.config.vocab_size;
        let history_mask = upload_semantic_history(self.session, vocabulary, history)?;
        let result = allocate_semantic_result(self.session)?;
        self.session.retain_cuda_buffer(tensor.clone());
        self.session.retain_cuda_buffer(history_mask.clone());
        self.session.retain_cuda_buffer(result.clone());
        let stream = self.session.device_mut().stream().clone();
        self.semantic_sampler.enqueue(
            &stream,
            &tensor,
            &history_mask,
            vocabulary,
            params,
            self.seed,
            self.semantic_steps,
            eos_id,
            &result,
        )?;
        self.semantic_steps = self.semantic_steps.wrapping_add(1);
        let bytes = self.session.read_output_bytes(&result)?;
        parse_semantic_result(&bytes, vocabulary, eos_id)
    }

    fn reset_fast(&mut self) -> Result<()> {
        self.session.reset_fast_prefix()
    }

    fn prime_fast(&mut self, context: CudaBuffer) -> Result<()> {
        let plan = &self.plans.fast_decode[0];
        let mut inputs = BTreeMap::new();
        inputs.insert("fast_context".to_owned(), context);
        let _ = self.session.enqueue(&plan.plan, &inputs)?;
        Ok(())
    }

    fn decode_fast(&mut self, codebook_index: usize, previous_code: u32) -> Result<Vec<f32>> {
        let plan = self
            .plans
            .fast_decode
            .get(codebook_index)
            .ok_or(FishError::InvalidShape(
                "CUDA Fast decode index out of bounds",
            ))?;
        let input = self.tokens(&plan.plan, "previous_acoustic_code", &[previous_code])?;
        let mut inputs = BTreeMap::new();
        inputs.insert("previous_acoustic_code".to_owned(), input);
        let outputs = self.session.enqueue(&plan.plan, &inputs)?;
        self.read_logits(plan, &outputs, "fast_logits")
    }

    fn sample_fast_codes(
        &mut self,
        initial: u32,
        codebook_size: usize,
        histories: &[Vec<usize>],
        params: SamplingParams,
        _host_sample: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
        on_codebook: &mut dyn FnMut(usize) -> Result<()>,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<u32>> {
        let count = histories.len() + 1;
        if count != self.plans.config.num_codebooks {
            return Err(FishError::InvalidShape("CUDA Fast codebook count mismatch"));
        }
        let vocab = self.plans.config.codebook_size;
        if codebook_size != vocab {
            return Err(FishError::InvalidShape(
                "CUDA Fast codebook vocabulary mismatch",
            ));
        }
        let column = upload_column(self.session, initial, count)?;
        self.session.retain_cuda_buffer(column.clone());
        let initial_plan = self.plans.fast_decode.get(1);
        let mut previous = if let Some(plan) = initial_plan {
            self.tokens(&plan.plan, "previous_acoustic_code", &[initial])?
        } else {
            self.session.upload_runtime_input(
                &TensorType::new(
                    vec![1],
                    ScalarType::I32,
                    fish_speech_core::graph_ir::Layout::Plain,
                ),
                &i32::try_from(initial)
                    .map_err(|_| FishError::InvalidShape("CUDA acoustic code exceeds i32"))?
                    .to_le_bytes(),
            )?
        };
        self.session.retain_cuda_buffer(previous.clone());
        let stream = self.session.device_mut().stream().clone();
        let seed = self.seed ^ self.fast_columns.wrapping_mul(0xd1b54a32d192ed03);
        self.fast_columns = self.fast_columns.wrapping_add(1);
        for (offset, history) in histories.iter().enumerate() {
            if is_cancelled() {
                return Err(FishError::InvalidTransition("synthesis cancelled"));
            }
            let index = offset + 1;
            let plan = self
                .plans
                .fast_decode
                .get(index)
                .ok_or(FishError::InvalidShape(
                    "CUDA Fast plan missing for acoustic codebook",
                ))?;
            let mut inputs = BTreeMap::new();
            inputs.insert("previous_acoustic_code".to_owned(), previous);
            let outputs = self.session.enqueue(&plan.plan, &inputs)?;
            let logits = named_output(plan, &outputs, "fast_logits")?;
            let ty = output_type(plan, "fast_logits")?;
            if ty.dtype != ScalarType::F32 || ty.shape != [1, 1, vocab] {
                return Err(FishError::InvalidShape("CUDA Fast logits shape mismatch"));
            }
            let history_mask = upload_history_mask(self.session, vocab, history)?;
            let sampled = allocate_token(self.session)?;
            self.session.retain_cuda_buffer(logits.clone());
            self.session.retain_cuda_buffer(history_mask.clone());
            self.session.retain_cuda_buffer(sampled.clone());
            self.fast_sampler.enqueue(
                &stream,
                &logits,
                &history_mask,
                vocab,
                params,
                seed,
                index,
                &sampled,
                &column,
            )?;
            previous = sampled;
            on_codebook(index)?;
        }
        let result = parse_column(&self.session.read_output_bytes(&column)?, count, vocab)?;
        Ok(result)
    }
}

impl<F> Drop for CudaDualArBackend<'_, F>
where
    F: FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()>,
{
    fn drop(&mut self) {
        let _ = self.session.retire_completed();
    }
}

pub(crate) fn expected_runtime_input(plan: &ArStreamPlan, name: &str) -> Result<TensorType> {
    for phase in &plan.phases {
        if let ArStreamPhase::Network(network) = phase {
            for step in &network.steps {
                if step.name == name {
                    return Ok(step.output_type.clone());
                }
            }
        }
    }
    Err(FishError::InvalidShape(
        "CUDA runtime input absent from graph",
    ))
}

pub(crate) fn output_type(graph: &PreparedArGraph, name: &str) -> Result<TensorType> {
    let binding = graph.outputs.get(name).ok_or(FishError::InvalidShape(
        "CUDA output absent from original graph",
    ))?;
    for phase in &graph.plan.phases {
        if let ArStreamPhase::Network(network) = phase {
            for step in &network.steps {
                if &step.result == binding && graph.plan.final_outputs.contains(binding) {
                    return Ok(step.output_type.clone());
                }
            }
        }
    }
    Err(FishError::InvalidShape("CUDA output absent from graph"))
}

pub(crate) fn named_output(
    graph: &PreparedArGraph,
    outputs: &BTreeMap<Binding, CudaBuffer>,
    name: &str,
) -> Result<CudaBuffer> {
    let binding = graph.outputs.get(name).ok_or(FishError::InvalidShape(
        "CUDA output absent from original graph",
    ))?;
    if !graph.plan.final_outputs.contains(binding) {
        return Err(FishError::InvalidShape(
            "CUDA requested output was not retained",
        ));
    }
    outputs
        .get(binding)
        .cloned()
        .ok_or(FishError::InvalidShape("CUDA named output missing"))
}

fn validate_cuda_generation_request(
    model: &FishModelConfig,
    prompt: &EncodedConversation,
    plans: &PreparedDualArPlans,
    semantic_lut: &[i32],
    semantic_begin_id: u32,
    eos_id: u32,
    params: SamplingParams,
) -> Result<()> {
    model.validate()?;
    params.validate()?;
    if model.kind != ModelKind::DualAr
        || &plans.config != model
        || plans.prompt_len != prompt.length()
        || plans.max_new_tokens == 0
        || prompt.channels.len() != model.num_codebooks + 1
        || prompt.channels[0].is_empty()
        || prompt
            .channels
            .iter()
            .any(|channel| channel.len() != plans.prompt_len)
        || prompt.vq_positions.len() != plans.prompt_len
        || semantic_lut.len() != model.vocab_size
        || semantic_lut.iter().any(|&v| v != 0 && v != 1)
        || usize::try_from(eos_id)
            .ok()
            .is_none_or(|id| id >= model.vocab_size)
        || usize::try_from(semantic_begin_id)
            .ok()
            .is_none_or(|id| id >= model.vocab_size)
    {
        return Err(FishError::InvalidShape(
            "invalid prepared CUDA DualAR request",
        ));
    }
    Ok(())
}

/// Runs a bounded, explicitly precompiled DualAR profile using the same generator as the CPU
/// reference, retaining TensorRT contexts and KV caches for the entire request. This blocks at
/// each sampled logits boundary, not at each TensorRT network segment.
#[allow(clippy::too_many_arguments)]
pub fn generate_prepared_cuda_dual_ar(
    registry: &EngineRegistry,
    logger: &Logger,
    config: CachedArDispatchConfig<'_>,
    model: &FishModelConfig,
    prompt: &EncodedConversation,
    plans: &PreparedDualArPlans,
    checkpoint: &LoadedWeights,
    device_ordinal: usize,
    memory: DeviceMemoryBudget,
    scratch_pool: CudaScratchPool,
    semantic_lut: &[i32],
    semantic_begin_id: u32,
    eos_id: u32,
    params: SamplingParams,
    seed: u64,
    is_cancelled: &mut dyn FnMut() -> bool,
) -> Result<Vec<Vec<u32>>> {
    validate_cuda_generation_request(
        model,
        prompt,
        plans,
        semantic_lut,
        semantic_begin_id,
        eos_id,
        params,
    )?;
    if is_cancelled() {
        return Err(FishError::InvalidTransition("synthesis cancelled"));
    }
    let references = plans.ar_plans();
    with_resident_cuda_ar_session(
        registry,
        logger,
        config,
        &references,
        checkpoint,
        device_ordinal,
        memory,
        scratch_pool,
        |session| {
            let mut backend = CudaDualArBackend::new(session, plans, semantic_lut, seed)?;
            let mut rng = SessionRng::new(seed);
            let mut trace = DualArTrace::default();
            generate_dual_ar(
                model,
                &mut backend,
                prompt,
                plans.max_new_tokens,
                semantic_begin_id,
                eos_id,
                params,
                &mut |logits, history, params| {
                    let probs = logits_to_probs(logits, history, params)?;
                    u32::try_from(rng.sample_exponential_race(&probs)?)
                        .map_err(|_| FishError::InvalidShape("CUDA token overflows u32"))
                },
                &mut trace,
                is_cancelled,
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_bucket_edges_are_contiguous_and_use_constant_networks_within_each_bucket() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let plans = PreparedDualArPlans::new(&config, 4, 130).unwrap();
        let mut unique_profiles = std::collections::BTreeSet::new();
        for active_len in 5..134 {
            let (index, _) = plans.slow_decode_profiles.locate(1, active_len).unwrap();
            let bucket_end = plans.slow_decode_profiles.buckets[index].max;
            let template = plans.slow_decode.get(&bucket_end).unwrap();
            let positioned =
                positioned_slow_decode(&template.plan, active_len, bucket_end).unwrap();
            assert_eq!(positioned.network_count(), template.plan.network_count());
            unique_profiles.insert(bucket_end);
        }
        assert_eq!(unique_profiles, [32, 64, 128, 256].into());
        assert_eq!(unique_profiles.len(), plans.slow_decode.len());
        assert_eq!(plans.slow_decode_profiles.locate(1, 32).unwrap().0, 0);
        assert_eq!(plans.slow_decode_profiles.locate(1, 33).unwrap().0, 1);
        assert_eq!(plans.slow_decode_profiles.locate(1, 65).unwrap().0, 2);
    }

    #[test]
    fn prepared_plans_cover_bounded_source_prefill_decode_and_fast_loop() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let plans = PreparedDualArPlans::new(&config, 4, 3).unwrap();
        assert_eq!(plans.slow_decode.keys().copied().collect::<Vec<_>>(), [32]);
        assert_eq!(plans.fast_decode.len(), config.num_codebooks);
        assert_eq!(plans.config, config);
        assert_eq!(plans.ar_plans().len(), 2 + config.num_codebooks);
        assert_eq!(plans.slow_decode_profiles.locate(1, 5).unwrap().0, 0);
        assert_eq!(
            output_type(&plans.slow_prefill, "logits").unwrap().shape,
            [1, 1, config.vocab_size]
        );
        assert_eq!(
            output_type(&plans.fast_decode[1], "fast_logits")
                .unwrap()
                .shape,
            [1, 1, config.codebook_size]
        );
        assert_eq!(
            output_type(&plans.slow_prefill, "fast_context")
                .unwrap()
                .shape,
            [1, 1, config.fast.as_ref().unwrap().dim]
        );
        assert_eq!(
            expected_runtime_input(&plans.fast_decode[1].plan, "previous_acoustic_code")
                .unwrap()
                .shape,
            [1, 1]
        );
        assert!(PreparedDualArPlans::new(&config, 4, 0).is_err());
        assert!(PreparedDualArPlans::new(&config, 4, config.max_seq_len).is_err());
        let longer = PreparedDualArPlans::new(&config, 4, config.max_seq_len - 4).unwrap();
        assert!(longer.slow_decode.len() < 16);
        assert!(longer.slow_decode.len() < config.max_seq_len - 4);
        let prompt = EncodedConversation {
            channels: vec![vec![1; 4]; config.num_codebooks + 1],
            vq_positions: vec![false; 4],
        };
        let lut = vec![0i32; config.vocab_size];
        let params = SamplingParams::default();
        assert!(
            validate_cuda_generation_request(&config, &prompt, &plans, &lut, 0, 1, params).is_ok()
        );
        let mut wrong_config = config.clone();
        wrong_config.rope_base *= 2.0;
        assert!(
            validate_cuda_generation_request(&wrong_config, &prompt, &plans, &lut, 0, 1, params)
                .is_err()
        );
        let mut bad_prompt = prompt.clone();
        bad_prompt.channels[0].pop();
        assert!(
            validate_cuda_generation_request(&config, &bad_prompt, &plans, &lut, 0, 1, params)
                .is_err()
        );
        assert!(
            validate_cuda_generation_request(&config, &prompt, &plans, &[2], 0, 1, params).is_err()
        );
        assert!(
            validate_cuda_generation_request(
                &config,
                &prompt,
                &plans,
                &lut,
                0,
                config.vocab_size as u32,
                params
            )
            .is_err()
        );
    }
}
