//! CubeCL-backed semantic generation. The neural graph is executed by CubeCL;
//! sampled and greedy DualAR tokens are selected by CubeCL kernels.

use cubecl::prelude::Client;
use fish_speech_core::{
    ar_generation::{DualArBackend, DualArTrace, SlowArForward, SlowArLogits, generate_dual_ar},
    config::{FishModelConfig, ModelKind},
    conversation::EncodedConversation,
    cpu_ir::{NamedTensors, TensorData},
    cpu_weighted_ar::SelectionMode,
    error::{FishError, Result},
    graph_ir::{Layout, ScalarType, TensorType},
    naive_generation::{NaiveArBackend, NaiveArForward, generate_naive_ar},
    sampling::{SamplingParams, SessionRng},
    synthesis::SemanticGenerator,
    tokenizer::{SEMANTIC_TOKEN_COUNT, SpecialTokens},
};

use super::session::{
    CubeClDeviceTensor, CubeClModel, CubeClPreparedModel, CubeClRequest, CubeClRequestLimits,
    CubeClSampledToken,
};
use crate::weights::LoadedWeights;

fn checked_tokens(tokens: impl IntoIterator<Item = u32>) -> Result<Vec<i32>> {
    tokens
        .into_iter()
        .map(|token| {
            i32::try_from(token)
                .map_err(|_| FishError::InvalidShape("CubeCL token exceeds i32 range"))
        })
        .collect()
}

fn float_output(outputs: &NamedTensors, name: &str, expected: usize) -> Result<Vec<f32>> {
    match outputs.get(name) {
        Some(TensorData::F32(values)) if values.len() == expected => Ok(values.clone()),
        _ => Err(FishError::InvalidShape(
            "CubeCL AR output is absent or has wrong type or shape",
        )),
    }
}

fn checked_prompt(prompt: &EncodedConversation, codebooks: usize) -> Result<usize> {
    let count = codebooks + 1;
    if prompt.channels.len() != count || prompt.channels[0].is_empty() {
        return Err(FishError::InvalidShape(
            "CubeCL prompt has invalid channel count",
        ));
    }
    let len = prompt.channels[0].len();
    if prompt.vq_positions.len() != len || prompt.channels.iter().any(|c| c.len() != len) {
        return Err(FishError::InvalidShape(
            "CubeCL prompt channel lengths differ",
        ));
    }
    Ok(len)
}

fn request_limit(config: &FishModelConfig, prompt_len: usize, requested: usize) -> Result<usize> {
    if prompt_len == 0 || prompt_len >= config.max_seq_len {
        return Err(FishError::InvalidShape(
            "CubeCL prompt exceeds model context",
        ));
    }
    let available = config.max_seq_len - prompt_len;
    let count = if requested == 0 {
        available
    } else {
        requested.min(available)
    };
    if count == 0 || count > available || count > 64 {
        return Err(FishError::Unsupported(
            "CubeCL request requires 1..=64 prepared decode tokens within context",
        ));
    }
    Ok(count)
}

struct CubeClDualArBackend<'a, 'b> {
    request: &'a mut CubeClRequest<'b>,
    semantic_lut: &'a [i32],
    codebook_size: usize,
    vocab_size: usize,
    mode: SelectionMode,
    rng: &'a mut SessionRng,
    pending_semantic: Option<CubeClDeviceTensor>,
    pending_semantic_token: Option<CubeClSampledToken>,
    pending_column: Option<CubeClDeviceTensor>,
    semantic_begin_id: u32,
}

impl CubeClDualArBackend<'_, '_> {
    fn slow(
        &mut self,
        key: &str,
        tokens: Vec<i32>,
        device_tokens: Option<CubeClDeviceTensor>,
    ) -> Result<SlowArForward<CubeClDeviceTensor>> {
        if self.request.cursor().fast_active_len > 0 {
            self.request.retire_blocking()?;
        }
        let mut inputs = NamedTensors::new();
        let mut devices = std::collections::BTreeMap::new();
        if let Some(tensor) = device_tokens {
            tensor.checked_token_shadow(&tokens)?;
            devices.insert("tokens".into(), tensor);
        } else {
            inputs.insert("tokens".into(), TensorData::I32(tokens));
        }
        inputs.insert(
            "semantic_membership_lut".into(),
            TensorData::I32(self.semantic_lut.to_vec()),
        );
        let readbacks: &[&str] = &[];
        let outputs = self
            .request
            .run_selected(key, &inputs, &devices, readbacks)?;
        self.pending_semantic = Some(
            outputs
                .device
                .get("logits")
                .ok_or(FishError::InvalidShape(
                    "CubeCL semantic device logits missing",
                ))?
                .clone(),
        );
        let logits = SlowArLogits::DevicePending;
        let fast_context = outputs
            .device
            .get("fast_context")
            .ok_or(FishError::InvalidShape(
                "CubeCL fast_context output missing",
            ))?
            .clone();
        Ok(SlowArForward {
            logits,
            fast_context,
        })
    }
}

impl DualArBackend for CubeClDualArBackend<'_, '_> {
    type FastContext = CubeClDeviceTensor;

    fn slow_prefill(
        &mut self,
        prompt: &EncodedConversation,
    ) -> Result<SlowArForward<CubeClDeviceTensor>> {
        self.slow(
            "slow_prefill",
            checked_tokens(prompt.channels.iter().flatten().copied())?,
            None,
        )
    }

    fn slow_decode(
        &mut self,
        previous_column: &[u32],
        active_len: usize,
    ) -> Result<SlowArForward<CubeClDeviceTensor>> {
        let device = self
            .pending_column
            .take()
            .ok_or(FishError::InvalidTransition(
                "CubeCL DualAR slow decode has no sampled device token column",
            ))?;
        self.slow(
            &format!("slow_decode_{active_len}"),
            checked_tokens(previous_column.iter().copied())?,
            Some(device),
        )
    }

    fn sample_semantic(
        &mut self,
        logits: &SlowArLogits,
        history: &[usize],
        params: SamplingParams,
        eos_id: u32,
        host_sample: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
    ) -> Result<(u32, bool)> {
        match logits {
            SlowArLogits::Host(values) => {
                let token = host_sample(values, history, params)?;
                Ok((token, token == eos_id))
            }
            SlowArLogits::DevicePending => {
                let pending = self
                    .pending_semantic
                    .take()
                    .ok_or(FishError::InvalidTransition(
                        "CubeCL semantic sampler has no pending logits",
                    ))?;
                let selection = match self.mode {
                    SelectionMode::Greedy => self.request.sample_greedy(
                        &pending,
                        0,
                        self.vocab_size,
                        history,
                        params,
                        eos_id,
                    ),
                    SelectionMode::Sample => self.request.sample_top_p(
                        &pending,
                        0,
                        self.vocab_size,
                        history,
                        params,
                        eos_id,
                        self.rng,
                    ),
                }?;
                self.pending_semantic_token = Some(self.request.take_sampled(selection.0)?);
                Ok(selection)
            }
        }
    }

    fn reset_fast(&mut self) -> Result<()> {
        self.request.retire_blocking()?;
        self.request.reset_fast();
        Ok(())
    }

    fn prime_fast(&mut self, context: CubeClDeviceTensor) -> Result<()> {
        let mut devices = std::collections::BTreeMap::new();
        devices.insert("fast_context".into(), context);
        self.request
            .run_selected("fast_0", &NamedTensors::new(), &devices, &[])?;
        Ok(())
    }

    fn decode_fast(&mut self, codebook_index: usize, previous_code: u32) -> Result<Vec<f32>> {
        let mut runtime = NamedTensors::new();
        runtime.insert(
            "previous_acoustic_code".into(),
            TensorData::I32(checked_tokens([previous_code])?),
        );
        let result = self
            .request
            .run(&format!("fast_{codebook_index}"), &runtime)?;
        float_output(&result, "fast_logits", self.codebook_size)
    }
    fn sample_fast_codes(
        &mut self,
        initial: u32,
        codebook_size: usize,
        histories: &[Vec<usize>],
        params: SamplingParams,
        _sample: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
        on_codebook: &mut dyn FnMut(usize) -> Result<()>,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<u32>> {
        let semantic = self
            .pending_semantic_token
            .take()
            .ok_or(FishError::InvalidTransition(
                "CubeCL Fast AR lacks semantic device token",
            ))?;
        let mut codes = vec![initial];
        let mut selections = vec![(semantic.clone(), false), (semantic.clone(), true)];
        let fast_type = TensorType::new([1, 1], ScalarType::I32, Layout::Plain);
        for (index, history) in histories.iter().enumerate() {
            if is_cancelled() {
                return Err(FishError::InvalidTransition("synthesis cancelled"));
            }
            let codebook = index + 1;
            let previous = *codes.last().expect("Fast AR initial code present");
            let token = {
                let inputs = NamedTensors::new();
                let previous_source = selections.last().expect("Fast code 0 present");
                let device_previous = self.request.pack_token_column(
                    &[(&previous_source.0, previous_source.1)],
                    &[previous],
                    self.semantic_begin_id,
                    codebook_size,
                    fast_type.clone(),
                )?;
                let mut devices = std::collections::BTreeMap::new();
                devices.insert("previous_acoustic_code".into(), device_previous);
                let outputs = self.request.run_selected(
                    &format!("fast_{codebook}"),
                    &inputs,
                    &devices,
                    &[],
                )?;
                let device_logits =
                    outputs
                        .device
                        .get("fast_logits")
                        .ok_or(FishError::InvalidShape(
                            "CubeCL Fast AR device logits missing",
                        ))?;
                match self.mode {
                    SelectionMode::Greedy => {
                        self.request
                            .sample_greedy(
                                device_logits,
                                0,
                                codebook_size,
                                history,
                                params,
                                i32::MAX as u32,
                            )?
                            .0
                    }
                    SelectionMode::Sample => {
                        self.request
                            .sample_top_p(
                                device_logits,
                                0,
                                codebook_size,
                                history,
                                params,
                                i32::MAX as u32,
                                self.rng,
                            )?
                            .0
                    }
                }
            };
            if token as usize >= codebook_size {
                return Err(FishError::InvalidShape(
                    "Fast AR sample exceeds codebook size",
                ));
            }
            let sampled = self.request.take_sampled(token)?;
            selections.push((sampled, false));
            codes.push(token);
            on_codebook(codebook)?;
        }
        let mut complete = vec![semantic.token];
        complete.extend(codes.iter().copied());
        let refs = selections
            .iter()
            .map(|(s, derived)| (s, *derived))
            .collect::<Vec<_>>();
        self.pending_column = Some(self.request.pack_token_column(
            &refs,
            &complete,
            self.semantic_begin_id,
            codebook_size,
            TensorType::new([1, complete.len(), 1], ScalarType::I32, Layout::TokensBct),
        )?);
        Ok(codes)
    }
}

struct CubeClNaiveBackend<'a, 'b> {
    request: &'a mut CubeClRequest<'b>,
    semantic_lut: &'a [i32],
    vocab_size: usize,
    codebook_logits_len: usize,
    mode: SelectionMode,
    rng: &'a mut SessionRng,
    pending_semantic: Option<CubeClDeviceTensor>,
    pending_acoustic: Option<CubeClDeviceTensor>,
    sampled_column: Vec<CubeClSampledToken>,
}

impl CubeClNaiveBackend<'_, '_> {
    fn forward(
        &mut self,
        key: &str,
        tokens: Vec<i32>,
        device_tokens: Option<CubeClDeviceTensor>,
    ) -> Result<NaiveArForward> {
        if self.request.cursor().slow_active_len > 0 {
            self.request.retire_blocking()?;
        }
        let mut input = NamedTensors::new();
        let mut devices = std::collections::BTreeMap::new();
        if let Some(tensor) = device_tokens {
            tensor.checked_token_shadow(&tokens)?;
            devices.insert("tokens".into(), tensor);
        } else {
            input.insert("tokens".into(), TensorData::I32(tokens));
        }
        input.insert(
            "semantic_membership_lut".into(),
            TensorData::I32(self.semantic_lut.to_vec()),
        );
        let output = self.request.run_selected(key, &input, &devices, &[])?;
        let semantic = output.device.get("logits").ok_or(FishError::InvalidShape(
            "CubeCL Naive device semantic logits missing",
        ))?;
        let acoustic = output
            .device
            .get("codebook_logits")
            .ok_or(FishError::InvalidShape(
                "CubeCL Naive device acoustic logits missing",
            ))?;
        if semantic.shape_elements() != self.vocab_size
            || acoustic.shape_elements() != self.codebook_logits_len
        {
            return Err(FishError::InvalidShape(
                "CubeCL Naive device logits shape mismatch",
            ));
        }
        self.pending_semantic = Some(semantic.clone());
        self.pending_acoustic = Some(acoustic.clone());
        Ok(NaiveArForward::DevicePending)
    }

    fn select(
        &mut self,
        logits: &CubeClDeviceTensor,
        offset: usize,
        vocabulary: usize,
        history: &[usize],
        params: SamplingParams,
        eos: u32,
    ) -> Result<u32> {
        Ok(match self.mode {
            SelectionMode::Greedy => {
                self.request
                    .sample_greedy(logits, offset, vocabulary, history, params, eos)?
                    .0
            }
            SelectionMode::Sample => {
                self.request
                    .sample_top_p(logits, offset, vocabulary, history, params, eos, self.rng)?
                    .0
            }
        })
    }
}

impl NaiveArBackend for CubeClNaiveBackend<'_, '_> {
    fn slow_prefill(&mut self, prompt: &EncodedConversation) -> Result<NaiveArForward> {
        self.forward(
            "slow_prefill",
            checked_tokens(prompt.channels.iter().flatten().copied())?,
            None,
        )
    }

    fn slow_decode(
        &mut self,
        previous_column: &[u32],
        active_len: usize,
    ) -> Result<NaiveArForward> {
        let ids = checked_tokens(previous_column.iter().copied())?;
        if self.sampled_column.len() != ids.len() {
            return Err(FishError::InvalidTransition(
                "CubeCL Naive sampled column incomplete",
            ));
        }
        let refs = self
            .sampled_column
            .iter()
            .map(|sample| (sample, false))
            .collect::<Vec<_>>();
        let device = self.request.pack_token_column(
            &refs,
            previous_column,
            0,
            self.codebook_logits_len,
            TensorType::new([1, ids.len(), 1], ScalarType::I32, Layout::TokensBct),
        )?;
        self.sampled_column.clear();
        self.forward(&format!("slow_decode_{active_len}"), ids, Some(device))
    }

    fn sample_semantic(
        &mut self,
        _: &NaiveArForward,
        params: SamplingParams,
        _: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
    ) -> Result<u32> {
        let logits = self
            .pending_semantic
            .take()
            .ok_or(FishError::InvalidTransition(
                "CubeCL Naive semantic logits not pending",
            ))?;
        let token = self.select(&logits, 0, self.vocab_size, &[], params, i32::MAX as u32)?;
        self.sampled_column.clear();
        self.sampled_column.push(self.request.take_sampled(token)?);
        Ok(token)
    }

    fn sample_acoustic(
        &mut self,
        _: &NaiveArForward,
        codebook: usize,
        codebook_size: usize,
        history: &[usize],
        params: SamplingParams,
        _: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
    ) -> Result<u32> {
        let logits = self
            .pending_acoustic
            .as_ref()
            .ok_or(FishError::InvalidTransition(
                "CubeCL Naive acoustic logits not pending",
            ))?
            .clone();
        let offset = codebook
            .checked_mul(codebook_size)
            .ok_or(FishError::InvalidShape(
                "CubeCL Naive acoustic offset overflow",
            ))?;
        let token = self.select(
            &logits,
            offset,
            codebook_size,
            history,
            params,
            i32::MAX as u32,
        )?;
        self.sampled_column.push(self.request.take_sampled(token)?);
        Ok(token)
    }
}

/// Independent CubeCL semantic generator using the existing FishModel TTS API.
/// Model weights are uploaded once per cached request geometry; GPU sampling is C04.
pub struct CubeClSemanticGenerator {
    config: FishModelConfig,
    checkpoint: LoadedWeights,
    client: Client,
    limits: CubeClRequestLimits,
    semantic_lut: Vec<i32>,
    semantic_begin_id: u32,
    eos_id: u32,
    seed: u64,
    rng: SessionRng,
    mode: SelectionMode,
    cache: Option<((usize, usize), CubeClModel)>,
}

impl CubeClSemanticGenerator {
    pub fn new(
        client: Client,
        config: FishModelConfig,
        checkpoint: LoadedWeights,
        special: &SpecialTokens,
        seed: u64,
        mode: SelectionMode,
        limits: CubeClRequestLimits,
    ) -> Result<Self> {
        config.validate()?;
        if checkpoint.manifest().component != "semantic" {
            return Err(FishError::InvalidConfig(
                "CubeCL requires semantic checkpoint",
            ));
        }
        if limits.max_pending_bytes == 0
            || limits.max_kv_bytes == 0
            || limits.max_pending_steps == 0
        {
            return Err(FishError::InvalidConfig(
                "CubeCL request limits must be positive",
            ));
        }
        let end = special
            .semantic_begin
            .checked_add(SEMANTIC_TOKEN_COUNT)
            .ok_or(FishError::InvalidShape("CubeCL semantic range overflow"))?;
        if end > config.vocab_size || special.im_end >= config.vocab_size {
            return Err(FishError::InvalidShape(
                "CubeCL semantic vocabulary mismatch",
            ));
        }
        let semantic_begin_id = u32::try_from(special.semantic_begin)
            .map_err(|_| FishError::InvalidShape("CubeCL semantic token ID exceeds u32"))?;
        let eos_id = u32::try_from(special.im_end)
            .map_err(|_| FishError::InvalidShape("CubeCL EOS token ID exceeds u32"))?;
        let mut lut = vec![0i32; config.vocab_size];
        lut[special.semantic_begin..end].fill(1);
        Ok(Self {
            config,
            checkpoint,
            client,
            limits,
            semantic_lut: lut,
            semantic_begin_id,
            eos_id,
            seed,
            rng: SessionRng::new(seed),
            mode,
            cache: None,
        })
    }

    pub fn prepared_geometry(&self) -> Option<(usize, usize)> {
        self.cache.as_ref().map(|(key, _)| *key)
    }

    fn generate_inner(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        sampling: SamplingParams,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<Vec<u32>>> {
        sampling.validate()?;
        let prompt_len = checked_prompt(prompt, self.config.num_codebooks)?;
        let limit = request_limit(&self.config, prompt_len, max_new_tokens)?;
        if is_cancelled() {
            return Err(FishError::InvalidTransition("synthesis cancelled"));
        }
        let geometry = (prompt_len, limit);
        if self.cache.as_ref().is_none_or(|(key, _)| *key != geometry) {
            let plans = CubeClPreparedModel::from_transformer(&self.config, prompt_len, limit)?;
            let model = CubeClModel::from_checkpoint(self.client.clone(), plans, &self.checkpoint)
                .map_err(|_| {
                    FishError::InvalidShape("CubeCL checkpoint weights failed validation or upload")
                })?;
            self.cache = Some((geometry, model));
        }
        let model = &self.cache.as_ref().expect("prepared CubeCL model").1;
        let mut request = model.start_request(self.limits)?;
        let outcome = match self.config.kind {
            ModelKind::DualAr => {
                let mut backend = CubeClDualArBackend {
                    request: &mut request,
                    semantic_lut: &self.semantic_lut,
                    codebook_size: self.config.codebook_size,
                    vocab_size: self.config.vocab_size,
                    mode: self.mode,
                    rng: &mut self.rng,
                    pending_semantic: None,
                    pending_semantic_token: None,
                    pending_column: None,
                    semantic_begin_id: self.semantic_begin_id,
                };
                generate_dual_ar(
                    &self.config,
                    &mut backend,
                    prompt,
                    limit,
                    self.semantic_begin_id,
                    self.eos_id,
                    sampling,
                    &mut |_, _, _| {
                        Err(FishError::InvalidTransition(
                            "CubeCL DualAR unexpectedly requested host logits sampling",
                        ))
                    },
                    &mut DualArTrace::default(),
                    is_cancelled,
                )
            }
            ModelKind::Naive => {
                let mut backend = CubeClNaiveBackend {
                    request: &mut request,
                    semantic_lut: &self.semantic_lut,
                    vocab_size: self.config.vocab_size,
                    codebook_logits_len: self.config.num_codebooks * self.config.codebook_size,
                    mode: self.mode,
                    rng: &mut self.rng,
                    pending_semantic: None,
                    pending_acoustic: None,
                    sampled_column: Vec::new(),
                };
                generate_naive_ar(
                    &self.config,
                    &mut backend,
                    prompt,
                    limit,
                    self.eos_id,
                    sampling,
                    &mut |_, _, _| {
                        Err(FishError::InvalidTransition(
                            "CubeCL Naive unexpectedly requested host logits sampling",
                        ))
                    },
                    is_cancelled,
                )
            }
        };
        let retired = request.retire_blocking();
        match (outcome, retired) {
            (Ok(tokens), Ok(())) => Ok(tokens),
            (Err(err), _) => Err(err),
            (Ok(_), Err(err)) => Err(err),
        }
    }
}

impl SemanticGenerator for CubeClSemanticGenerator {
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
        self.generate_inner(prompt, max_new_tokens, sampling, &mut || false)
    }

    fn generate_with_cancel(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        sampling: SamplingParams,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<Vec<u32>>> {
        self.generate_inner(prompt, max_new_tokens, sampling, is_cancelled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_and_geometry_preflight_reject_invalid_inputs_before_gpu_use() {
        let config = FishModelConfig::synthetic_dual_ar_fixture();
        let valid = EncodedConversation {
            channels: vec![vec![1, 2, 3]; config.num_codebooks + 1],
            vq_positions: vec![false; 3],
        };
        assert_eq!(checked_prompt(&valid, config.num_codebooks).unwrap(), 3);
        assert_eq!(request_limit(&config, 3, 5).unwrap(), 5);
        assert!(request_limit(&config, 3, 65).is_err());
        assert_eq!(request_limit(&config, 252, config.max_seq_len).unwrap(), 4);
        assert!(request_limit(&config, 3, 0).is_err());
        let mut bad = valid.clone();
        bad.channels[1].pop();
        assert!(checked_prompt(&bad, config.num_codebooks).is_err());
        bad = valid;
        bad.vq_positions.pop();
        assert!(checked_prompt(&bad, config.num_codebooks).is_err());
        assert!(checked_tokens([i32::MAX as u32 + 1]).is_err());
    }

    #[test]
    fn naive_zero_means_available_context_with_profile_limit() {
        let mut config = FishModelConfig::synthetic_dual_ar_fixture();
        config.kind = ModelKind::Naive;
        config.fast = None;
        config.max_seq_len = 32;
        assert_eq!(request_limit(&config, 6, 0).unwrap(), 26);
        assert_eq!(request_limit(&config, 6, 100).unwrap(), 26);
        assert!(request_limit(&config, 32, 1).is_err());
    }

    #[test]
    fn logits_preflight_checks_dtype_and_exact_vocab() {
        let mut output = NamedTensors::new();
        output.insert("logits".into(), TensorData::F32(vec![1.0, 2.0]));
        assert_eq!(float_output(&output, "logits", 2).unwrap(), [1.0, 2.0]);
        assert!(float_output(&output, "logits", 3).is_err());
        output.insert("logits".into(), TensorData::I32(vec![1, 2]));
        assert!(float_output(&output, "logits", 2).is_err());
        assert!(float_output(&output, "missing", 2).is_err());
    }
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod device_chain_tests {
    use super::*;
    use cubecl::Device;

    fn fixture(kind: ModelKind) -> FishModelConfig {
        let mut c = FishModelConfig::synthetic_dual_ar_fixture();
        c.slow.dim = 8;
        c.slow.n_heads = 2;
        c.slow.n_kv_heads = 1;
        c.slow.layers = 1;
        c.slow.intermediate_size = 16;
        if let Some(fast) = c.fast.as_mut() {
            *fast = c.slow.clone();
        }
        c.vocab_size = 16;
        c.num_codebooks = 2;
        c.codebook_size = 8;
        c.max_seq_len = 6;
        c.kind = kind;
        if kind == ModelKind::Naive {
            c.fast = None;
        }
        c
    }

    fn prepared(config: &FishModelConfig) -> CubeClModel {
        let plans = CubeClPreparedModel::from_transformer(config, 2, 2).unwrap();
        let mut weights = NamedTensors::new();
        for (name, ty) in plans.weight_specs() {
            weights.insert(
                name.clone(),
                TensorData::F32(vec![0.0; ty.shape.iter().product()]),
            );
        }
        CubeClModel::new(Device::Cpu(Default::default()).client(), plans, &weights).unwrap()
    }

    fn prompt(config: &FishModelConfig) -> EncodedConversation {
        EncodedConversation {
            channels: vec![vec![1, 2]; config.num_codebooks + 1],
            vq_positions: vec![false; 2],
        }
    }

    #[test]
    fn dual_ar_runs_two_full_frames_with_forwarded_device_codes() {
        let config = fixture(ModelKind::DualAr);
        let model = prepared(&config);
        let mut request = model.start_request(CubeClRequestLimits::default()).unwrap();
        let mut rng = SessionRng::new(9);
        let mut backend = CubeClDualArBackend {
            request: &mut request,
            semantic_lut: &[0; 16],
            codebook_size: 8,
            vocab_size: 16,
            mode: SelectionMode::Greedy,
            rng: &mut rng,
            pending_semantic: None,
            pending_semantic_token: None,
            pending_column: None,
            semantic_begin_id: 0,
        };
        let mut trace = DualArTrace::default();
        let output = generate_dual_ar(
            &config,
            &mut backend,
            &prompt(&config),
            2,
            0,
            15,
            SamplingParams::default(),
            &mut |_, _, _| panic!("host logits should not be read"),
            &mut trace,
            &mut || false,
        )
        .unwrap();
        assert_eq!(output, vec![vec![0, 0]; 3]);
        assert_eq!(trace.slow_active_lengths, [2, 3]);
        assert_eq!(trace.fast_resets, 2);
        request.retire_blocking().unwrap();
        assert!(!request.is_poisoned());
    }

    #[test]
    fn naive_runs_two_full_frames_with_forwarded_device_codes() {
        let config = fixture(ModelKind::Naive);
        let model = prepared(&config);
        let mut request = model.start_request(CubeClRequestLimits::default()).unwrap();
        let mut rng = SessionRng::new(9);
        let mut backend = CubeClNaiveBackend {
            request: &mut request,
            semantic_lut: &[0; 16],
            vocab_size: 16,
            codebook_logits_len: 16,
            mode: SelectionMode::Greedy,
            rng: &mut rng,
            pending_semantic: None,
            pending_acoustic: None,
            sampled_column: Vec::new(),
        };
        let output = generate_naive_ar(
            &config,
            &mut backend,
            &prompt(&config),
            2,
            15,
            SamplingParams::default(),
            &mut |_, _, _| panic!("host logits should not be read"),
            &mut || false,
        )
        .unwrap();
        assert_eq!(output, vec![vec![0, 0]; 3]);
        request.retire_blocking().unwrap();
        assert!(!request.is_poisoned());
    }

    #[test]
    fn stochastic_device_chain_runs_two_frames_for_both_ar_variants() {
        for kind in [ModelKind::DualAr, ModelKind::Naive] {
            let mut config = fixture(kind);
            config.codebook_size = 16;
            let model = prepared(&config);
            let mut request = model.start_request(CubeClRequestLimits::default()).unwrap();
            let mut rng = SessionRng::new(41);
            let params = SamplingParams {
                temperature: 0.8,
                top_p: 0.75,
                repetition_penalty: 1.3,
            };
            // EOS is deliberately outside the synthetic vocabulary to force two chained steps.
            let generated = if kind == ModelKind::DualAr {
                let mut backend = CubeClDualArBackend {
                    request: &mut request,
                    semantic_lut: &[0; 16],
                    codebook_size: 16,
                    vocab_size: 16,
                    mode: SelectionMode::Sample,
                    rng: &mut rng,
                    pending_semantic: None,
                    pending_semantic_token: None,
                    pending_column: None,
                    semantic_begin_id: 0,
                };
                generate_dual_ar(
                    &config,
                    &mut backend,
                    &prompt(&config),
                    2,
                    0,
                    16,
                    params,
                    &mut |_, _, _| panic!("host sampler used"),
                    &mut DualArTrace::default(),
                    &mut || false,
                )
                .unwrap()
            } else {
                let mut backend = CubeClNaiveBackend {
                    request: &mut request,
                    semantic_lut: &[0; 16],
                    vocab_size: 16,
                    codebook_logits_len: 32,
                    mode: SelectionMode::Sample,
                    rng: &mut rng,
                    pending_semantic: None,
                    pending_acoustic: None,
                    sampled_column: Vec::new(),
                };
                generate_naive_ar(
                    &config,
                    &mut backend,
                    &prompt(&config),
                    2,
                    16,
                    params,
                    &mut |_, _, _| panic!("host sampler used"),
                    &mut || false,
                )
                .unwrap()
            };
            assert_eq!(generated.len(), 3);
            for tokens in &generated {
                assert_eq!(tokens.len(), 2);
                assert!(tokens.iter().all(|&word| word < 16));
            }
            request.retire_blocking().unwrap();
        }
    }

    #[test]
    fn greedy_eos_stops_after_first_forwarded_frame() {
        for kind in [ModelKind::DualAr, ModelKind::Naive] {
            let config = fixture(kind);
            let model = prepared(&config);
            let mut request = model.start_request(CubeClRequestLimits::default()).unwrap();
            let mut rng = SessionRng::new(3);
            let params = SamplingParams::default();
            let generated = if kind == ModelKind::DualAr {
                let mut backend = CubeClDualArBackend {
                    request: &mut request,
                    semantic_lut: &[0; 16],
                    codebook_size: 8,
                    vocab_size: 16,
                    mode: SelectionMode::Greedy,
                    rng: &mut rng,
                    pending_semantic: None,
                    pending_semantic_token: None,
                    pending_column: None,
                    semantic_begin_id: 0,
                };
                generate_dual_ar(
                    &config,
                    &mut backend,
                    &prompt(&config),
                    2,
                    0,
                    0,
                    params,
                    &mut |_, _, _| panic!("host sampler used"),
                    &mut DualArTrace::default(),
                    &mut || false,
                )
                .unwrap()
            } else {
                let mut backend = CubeClNaiveBackend {
                    request: &mut request,
                    semantic_lut: &[0; 16],
                    vocab_size: 16,
                    codebook_logits_len: 16,
                    mode: SelectionMode::Greedy,
                    rng: &mut rng,
                    pending_semantic: None,
                    pending_acoustic: None,
                    sampled_column: Vec::new(),
                };
                generate_naive_ar(
                    &config,
                    &mut backend,
                    &prompt(&config),
                    2,
                    0,
                    params,
                    &mut |_, _, _| panic!("host sampler used"),
                    &mut || false,
                )
                .unwrap()
            };
            assert_eq!(generated, vec![vec![0]; 3]);
            request.retire_blocking().unwrap();
        }
    }
    #[test]
    fn cancellation_between_frames_does_not_reuse_speculative_device_tokens() {
        for kind in [ModelKind::DualAr, ModelKind::Naive] {
            let config = fixture(kind);
            let model = prepared(&config);
            let mut request = model.start_request(CubeClRequestLimits::default()).unwrap();
            let mut rng = SessionRng::new(21);
            let mut calls = 0;
            let mut cancelled = || {
                calls += 1;
                calls > if kind == ModelKind::DualAr { 2 } else { 3 }
            };
            let result = if kind == ModelKind::DualAr {
                let mut backend = CubeClDualArBackend {
                    request: &mut request,
                    semantic_lut: &[0; 16],
                    codebook_size: 8,
                    vocab_size: 16,
                    mode: SelectionMode::Greedy,
                    rng: &mut rng,
                    pending_semantic: None,
                    pending_semantic_token: None,
                    pending_column: None,
                    semantic_begin_id: 0,
                };
                generate_dual_ar(
                    &config,
                    &mut backend,
                    &prompt(&config),
                    2,
                    0,
                    15,
                    SamplingParams::default(),
                    &mut |_, _, _| panic!("host sampler used"),
                    &mut DualArTrace::default(),
                    &mut cancelled,
                )
            } else {
                let mut backend = CubeClNaiveBackend {
                    request: &mut request,
                    semantic_lut: &[0; 16],
                    vocab_size: 16,
                    codebook_logits_len: 16,
                    mode: SelectionMode::Greedy,
                    rng: &mut rng,
                    pending_semantic: None,
                    pending_acoustic: None,
                    sampled_column: Vec::new(),
                };
                generate_naive_ar(
                    &config,
                    &mut backend,
                    &prompt(&config),
                    2,
                    15,
                    SamplingParams::default(),
                    &mut |_, _, _| panic!("host sampler used"),
                    &mut cancelled,
                )
            };
            assert!(matches!(
                result,
                Err(FishError::InvalidTransition("synthesis cancelled"))
            ));
            request.retire_blocking().unwrap();
        }
    }
}
