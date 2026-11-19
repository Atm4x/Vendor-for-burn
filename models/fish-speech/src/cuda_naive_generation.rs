use std::{collections::BTreeMap, sync::Arc};

use cudarc::driver::CudaStream;
use fish_speech_core::{
    ar_stream_plan::ArStreamPlan,
    config::{FishModelConfig, ModelKind},
    conversation::EncodedConversation,
    error::{FishError, Result},
    graph::Stage,
    graph_ir::ScalarType,
    sampling::SamplingParams,
    shape_profiles::ArProfileGroup,
    transformer_ir::TransformerIrFactory,
    trtx_blueprint::{Binding, TrtBlueprint},
};

use crate::{
    cuda_fast_sampling::{CudaFastSampler, allocate_token, upload_column, upload_history_mask},
    cuda_generation::{PreparedArGraph, expected_runtime_input, named_output, output_type},
    cuda_semantic_sampling::{
        CudaSemanticSampler, allocate_semantic_result, parse_semantic_result,
        upload_semantic_history,
    },
    cuda_stream::{CudaBuffer, CudaScratchPool},
    cuda_trtx::{
        CachedArDispatchConfig, ResidentCudaAr, WeightedCudaArSession,
        with_resident_cuda_ar_session,
    },
    device_memory::DeviceMemoryBudget,
    dynamic_decode::{
        KV_MASK, ROPE_COS, ROPE_SIN, decode_control_bytes, parameterize_slow_decode,
        positioned_slow_decode,
    },
    trtx_cache::EngineRegistry,
    weights::LoadedWeights,
};

pub struct PreparedNaiveArPlans {
    prefill: PreparedArGraph,
    decode: BTreeMap<usize, PreparedArGraph>,
    profiles: ArProfileGroup,
    prompt_len: usize,
    max_new_tokens: usize,
    config: FishModelConfig,
}

impl PreparedNaiveArPlans {
    pub fn new(config: &FishModelConfig, prompt_len: usize, max_new_tokens: usize) -> Result<Self> {
        config.validate()?;
        if config.kind != ModelKind::Naive {
            return Err(FishError::Unsupported(
                "Naive GPU plans require Naive model",
            ));
        }
        if prompt_len == 0
            || prompt_len >= config.max_seq_len
            || max_new_tokens == 0
            || max_new_tokens > config.max_seq_len - prompt_len
        {
            return Err(FishError::InvalidShape("invalid Naive CUDA profile length"));
        }
        let factory = TransformerIrFactory::new(config)?;
        let prefill = PreparedArGraph::from_graph(factory.slow_prefill(prompt_len)?)?;
        let profiles = ArProfileGroup::new(Stage::SlowDecode, config.max_seq_len, 32)?;
        let mut decode = BTreeMap::new();
        if max_new_tokens > 1 {
            let first_active = prompt_len + 1;
            let last_active = prompt_len + max_new_tokens - 1;
            for bucket in &profiles.buckets {
                if bucket.max < first_active || bucket.min > last_active {
                    continue;
                }
                let mut graph = PreparedArGraph::from_graph(factory.slow_decode(bucket.max)?)?;
                parameterize_slow_decode(&mut graph.plan, bucket.max)?;
                decode.insert(bucket.max, graph);
            }
        }
        let result = Self {
            prefill,
            decode,
            profiles,
            prompt_len,
            max_new_tokens,
            config: config.clone(),
        };
        for plan in result.graphs() {
            if output_type(plan, "logits")?.shape != [1, 1, config.vocab_size]
                || output_type(plan, "codebook_logits")?.shape
                    != [1, 1, config.num_codebooks * config.codebook_size]
            {
                return Err(FishError::InvalidShape(
                    "Naive CUDA output geometry mismatch",
                ));
            }
        }
        Ok(result)
    }

    pub(crate) fn ar_plans(&self) -> Vec<&ArStreamPlan> {
        self.graphs().into_iter().map(|g| &g.plan).collect()
    }

    fn graphs(&self) -> Vec<&PreparedArGraph> {
        let mut result = vec![&self.prefill];
        result.extend(self.decode.values());
        result
    }

    fn graph_for_step(&self, step: usize) -> Result<PreparedArGraph> {
        if step == 0 {
            return Ok(PreparedArGraph {
                plan: self.prefill.plan.clone(),
                outputs: self.prefill.outputs.clone(),
            });
        }
        let active_len = self.prompt_len + step;
        let (index, _) = self.profiles.locate(1, active_len)?;
        let bucket_end = self.profiles.buckets[index].max;
        let original = self.decode.get(&bucket_end).ok_or(FishError::InvalidShape(
            "Naive CUDA Slow decode bucket absent",
        ))?;
        Ok(PreparedArGraph {
            plan: positioned_slow_decode(&original.plan, active_len, bucket_end)?,
            outputs: original.outputs.clone(),
        })
    }
}

fn checked_token_bytes(tokens: impl IntoIterator<Item = u32>) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for token in tokens {
        bytes.extend_from_slice(
            &i32::try_from(token)
                .map_err(|_| FishError::InvalidShape("Naive CUDA token exceeds i32"))?
                .to_le_bytes(),
        );
    }
    Ok(bytes)
}

fn naive_history(columns: &[Vec<u32>], codebook: usize) -> Vec<usize> {
    let mut history = columns[codebook + 1]
        .iter()
        .skip(1)
        .rev()
        .take(16)
        .copied()
        .collect::<Vec<_>>();
    history.reverse();
    let mut history = history.into_iter().map(|v| v as usize).collect::<Vec<_>>();
    history.resize(16, 0);
    history
}

fn decode_naive_column(
    bytes: &[u8],
    semantic_vocab: usize,
    codebook_vocab: usize,
    codebooks: usize,
) -> Result<Vec<u32>> {
    if bytes.len() != (codebooks + 1) * 4 {
        return Err(FishError::InvalidShape(
            "Naive CUDA sampled column byte count mismatch",
        ));
    }
    bytes
        .chunks_exact(4)
        .enumerate()
        .map(|(index, data)| {
            let token = i32::from_le_bytes(data.try_into().unwrap());
            let limit = if index == 0 {
                semantic_vocab
            } else {
                codebook_vocab
            };
            u32::try_from(token)
                .ok()
                .filter(|&value| (value as usize) < limit)
                .ok_or(FishError::InvalidSampling(
                    "Naive CUDA produced invalid sampled token",
                ))
        })
        .collect()
}

fn validate_naive_request(
    config: &FishModelConfig,
    prompt: &EncodedConversation,
    plans: &PreparedNaiveArPlans,
    semantic_lut: &[i32],
    eos_id: u32,
    sampling: SamplingParams,
) -> Result<()> {
    config.validate()?;
    sampling.validate()?;
    if config.kind != ModelKind::Naive
        || config != &plans.config
        || plans.prompt_len != prompt.length()
        || plans.max_new_tokens == 0
        || prompt.channels.len() != config.num_codebooks + 1
        || prompt
            .channels
            .iter()
            .any(|channel| channel.len() != plans.prompt_len)
        || prompt.vq_positions.len() != plans.prompt_len
        || semantic_lut.len() != config.vocab_size
        || semantic_lut.iter().any(|&v| !matches!(v, 0 | 1))
        || eos_id as usize >= config.vocab_size
        || config.codebook_size > 4096
    {
        return Err(FishError::InvalidShape(
            "invalid Naive CUDA generation request",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_prepared_cuda_naive(
    registry: &EngineRegistry,
    resident: &mut Option<ResidentCudaAr>,
    config: CachedArDispatchConfig<'_>,
    model: &FishModelConfig,
    prompt: &EncodedConversation,
    plans: &PreparedNaiveArPlans,
    checkpoint: &LoadedWeights,
    device_ordinal: usize,
    memory: DeviceMemoryBudget,
    scratch_pool: CudaScratchPool,
    semantic_lut: &[i32],
    eos_id: u32,
    sampling: SamplingParams,
    seed: u64,
    is_cancelled: &mut dyn FnMut() -> bool,
) -> Result<Vec<Vec<u32>>> {
    validate_naive_request(model, prompt, plans, semantic_lut, eos_id, sampling)?;
    if is_cancelled() {
        return Err(FishError::InvalidTransition("synthesis cancelled"));
    }
    let references = plans.ar_plans();
    with_resident_cuda_ar_session(
        registry,
        resident,
        config,
        &references,
        checkpoint,
        device_ordinal,
        memory,
        scratch_pool,
        |session| {
            generate_naive_with_session(
                model,
                prompt,
                plans,
                session,
                semantic_lut,
                eos_id,
                sampling,
                seed,
                is_cancelled,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_naive_with_session<F>(
    config: &FishModelConfig,
    prompt: &EncodedConversation,
    plans: &PreparedNaiveArPlans,
    session: &mut WeightedCudaArSession<F>,
    semantic_lut: &[i32],
    eos_id: u32,
    params: SamplingParams,
    seed: u64,
    is_cancelled: &mut dyn FnMut() -> bool,
) -> Result<Vec<Vec<u32>>>
where
    F: FnMut(&TrtBlueprint, &BTreeMap<Binding, CudaBuffer>, &Arc<CudaStream>) -> Result<()>,
{
    let sampler = CudaFastSampler::new(session.device_mut().context())?;
    let semantic_sampler = CudaSemanticSampler::new(session.device_mut().context())?;
    let lut_type = expected_runtime_input(&plans.prefill.plan, "semantic_membership_lut")?;
    if lut_type.dtype != ScalarType::I32 || lut_type.shape != [semantic_lut.len()] {
        return Err(FishError::InvalidShape(
            "Naive semantic LUT graph type mismatch",
        ));
    }
    let lut_bytes = semantic_lut
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let lut = session.upload_runtime_input(&lut_type, &lut_bytes)?;
    session.retain_cuda_buffer(lut.clone());
    let main_params = SamplingParams {
        temperature: 0.1,
        top_p: 0.1,
        repetition_penalty: 1.0,
    };
    let mut columns = vec![Vec::new(); config.num_codebooks + 1];
    for step in 0..plans.max_new_tokens {
        if is_cancelled() {
            return Err(FishError::InvalidTransition("synthesis cancelled"));
        }
        let graph = plans.graph_for_step(step)?;
        let tokens = if step == 0 {
            checked_token_bytes(prompt.channels.iter().flatten().copied())?
        } else {
            checked_token_bytes(
                columns
                    .iter()
                    .map(|row| *row.last().expect("prior Naive token")),
            )?
        };
        let token_type = expected_runtime_input(&graph.plan, "tokens")?;
        if token_type.dtype != ScalarType::I32
            || token_type.shape.iter().product::<usize>() * 4 != tokens.len()
        {
            return Err(FishError::InvalidShape(
                "Naive CUDA tokens differ from plan",
            ));
        }
        let input = session.upload_runtime_input(&token_type, &tokens)?;
        let mut runtime = BTreeMap::from([
            ("tokens".to_owned(), input),
            ("semantic_membership_lut".to_owned(), lut.clone()),
        ]);
        if step > 0 {
            let active_len = plans.prompt_len + step;
            let (index, _) = plans.profiles.locate(1, active_len)?;
            let bucket = plans.profiles.buckets[index].max;
            let (cos, sin, mask) =
                decode_control_bytes(active_len, bucket, config.slow.head_dim(), config.rope_base)?;
            for (name, bytes) in [(ROPE_COS, cos), (ROPE_SIN, sin), (KV_MASK, mask)] {
                let ty = expected_runtime_input(&graph.plan, name)?;
                runtime.insert(name.to_owned(), session.upload_runtime_input(&ty, &bytes)?);
            }
        }
        let outputs = session.enqueue(&graph.plan, &runtime)?;
        let semantic_logits = named_output(&graph, &outputs, "logits")?;
        let acoustic_logits = named_output(&graph, &outputs, "codebook_logits")?;
        if output_type(&graph, "logits")?.dtype != ScalarType::F32
            || output_type(&graph, "codebook_logits")?.dtype != ScalarType::F32
        {
            return Err(FishError::InvalidShape("Naive CUDA logits must be FP32"));
        }
        let codebook_count = config.num_codebooks;
        let column = upload_column(session, 0, codebook_count + 1)?;
        let semantic_result = allocate_semantic_result(session)?;
        let semantic_history = upload_semantic_history(session, config.vocab_size, &[])?;
        session.retain_cuda_buffer(semantic_logits.clone());
        session.retain_cuda_buffer(acoustic_logits.clone());
        session.retain_cuda_buffer(column.clone());
        session.retain_cuda_buffer(semantic_result.clone());
        session.retain_cuda_buffer(semantic_history.clone());
        let stream = session.device_mut().stream().clone();
        semantic_sampler.enqueue_with_column(
            &stream,
            &semantic_logits,
            &semantic_history,
            config.vocab_size,
            main_params,
            seed,
            step as u64,
            eos_id,
            &semantic_result,
            Some(&column),
        )?;
        for cb in 0..codebook_count {
            if is_cancelled() {
                return Err(FishError::InvalidTransition("synthesis cancelled"));
            }
            let history = naive_history(&columns, cb);
            let history_mask = upload_history_mask(session, config.codebook_size, &history)?;
            let sampled = allocate_token(session)?;
            session.retain_cuda_buffer(history_mask.clone());
            session.retain_cuda_buffer(sampled.clone());
            sampler.enqueue_slice(
                &stream,
                &acoustic_logits,
                &history_mask,
                config.codebook_size,
                cb * config.codebook_size,
                params,
                seed ^ (step as u64).wrapping_mul(0xd1b54a32d192ed03),
                cb + 1,
                &sampled,
                &column,
            )?;
        }
        let sampled = decode_naive_column(
            &session.read_output_bytes(&column)?,
            config.vocab_size,
            config.codebook_size,
            codebook_count,
        )?;
        let (semantic_id, eos) = parse_semantic_result(
            &session.read_output_bytes(&semantic_result)?,
            config.vocab_size,
            eos_id,
        )?;
        if sampled[0] != semantic_id {
            return Err(FishError::InvalidSampling(
                "Naive GPU EOS and semantic token disagree",
            ));
        }
        for (dst, token) in columns.iter_mut().zip(sampled) {
            dst.push(token);
        }
        if eos {
            break;
        }
    }
    Ok(columns)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_fixture() -> FishModelConfig {
        let mut config = FishModelConfig::synthetic_dual_ar_fixture();
        config.kind = ModelKind::Naive;
        config.fast = None;
        config
    }

    #[test]
    fn naive_plans_use_one_slow_forward_for_all_codebooks() {
        let config = naive_fixture();
        let plans = PreparedNaiveArPlans::new(&config, 4, 6).unwrap();
        assert_eq!(
            output_type(&plans.prefill, "logits").unwrap().shape,
            [1, 1, config.vocab_size]
        );
        assert_eq!(
            output_type(&plans.prefill, "codebook_logits")
                .unwrap()
                .shape,
            [1, 1, config.num_codebooks * config.codebook_size]
        );
        assert!(plans.prefill.outputs.get("fast_context").is_none());
        assert_eq!(plans.ar_plans().len(), plans.decode.len() + 1);
        assert!(plans.decode.len() < 6);
        assert!(PreparedNaiveArPlans::new(&config, 4, 0).is_err());
        assert!(PreparedNaiveArPlans::new(&config, 4, config.max_seq_len).is_err());
    }

    #[test]
    fn naive_dynamic_decode_reuses_bucket_and_retains_outputs() {
        let config = naive_fixture();
        let plans = PreparedNaiveArPlans::new(&config, 4, 4).unwrap();
        for step in 1..4 {
            let graph = plans.graph_for_step(step).unwrap();
            assert_eq!(
                output_type(&graph, "logits").unwrap().shape,
                [1, 1, config.vocab_size]
            );
            assert_eq!(
                output_type(&graph, "codebook_logits").unwrap().shape,
                [1, 1, config.num_codebooks * config.codebook_size]
            );
            assert!(expected_runtime_input(&graph.plan, ROPE_COS).is_ok());
            assert!(expected_runtime_input(&graph.plan, KV_MASK).is_ok());
        }
    }

    #[test]
    fn naive_eos_column_and_history_follow_python_contract() {
        let semantic_vocab = 64;
        let acoustic_vocab = 16;
        let row = [63i32, 2, 15, 0]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            decode_naive_column(&row, semantic_vocab, acoustic_vocab, 3).unwrap(),
            [63, 2, 15, 0]
        );
        assert!(decode_naive_column(&row, 60, acoustic_vocab, 3).is_err());
        let cols = vec![vec![4, 5, 6], vec![1, 7, 8], vec![2, 4, 5], vec![3, 10, 11]];
        let h = naive_history(&cols, 0);
        assert_eq!(h.len(), 16);
        assert_eq!(&h[..2], &[7, 8]);
        assert!(h[2..].iter().all(|&v| v == 0));
    }
}
