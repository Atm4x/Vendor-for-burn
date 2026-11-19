//! CubeCL-backed semantic generation. The neural graph is executed by CubeCL;
//! token sampling remains on the host until the separate GPU-sampling phase.

use cubecl::prelude::Client;
use fish_speech_core::{
    ar_generation::{DualArBackend, DualArTrace, SlowArForward, SlowArLogits, generate_dual_ar},
    config::{FishModelConfig, ModelKind},
    conversation::EncodedConversation,
    cpu_ir::{NamedTensors, TensorData},
    cpu_weighted_ar::SelectionMode,
    error::{FishError, Result},
    naive_generation::{NaiveArBackend, NaiveArForward, generate_naive_ar},
    sampling::{SamplingParams, SessionRng, greedy, logits_to_probs},
    synthesis::SemanticGenerator,
    tokenizer::{SEMANTIC_TOKEN_COUNT, SpecialTokens},
};

use super::session::{
    CubeClDeviceTensor, CubeClModel, CubeClPreparedModel, CubeClRequest, CubeClRequestLimits,
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

fn sample_host(
    rng: &mut SessionRng,
    mode: SelectionMode,
    logits: &[f32],
    history: &[usize],
    params: SamplingParams,
) -> Result<u32> {
    let probabilities = logits_to_probs(logits, history, params)?;
    let index = match mode {
        SelectionMode::Sample => rng.sample_exponential_race(&probabilities)?,
        SelectionMode::Greedy => greedy(&probabilities)?,
    };
    u32::try_from(index).map_err(|_| FishError::InvalidShape("CubeCL sampled token exceeds u32"))
}

struct CubeClDualArBackend<'a, 'b> {
    request: &'a mut CubeClRequest<'b>,
    semantic_lut: &'a [i32],
    codebook_size: usize,
    vocab_size: usize,
    mode: SelectionMode,
    pending_semantic: Option<CubeClDeviceTensor>,
}

impl CubeClDualArBackend<'_, '_> {
    fn slow(&mut self, key: &str, tokens: Vec<i32>) -> Result<SlowArForward<CubeClDeviceTensor>> {
        if self.request.cursor().fast_active_len > 0 {
            self.request.retire_blocking()?;
        }
        let mut inputs = NamedTensors::new();
        inputs.insert("tokens".into(), TensorData::I32(tokens));
        inputs.insert(
            "semantic_membership_lut".into(),
            TensorData::I32(self.semantic_lut.to_vec()),
        );
        let readbacks: &[&str] = if self.mode == SelectionMode::Greedy {
            &[]
        } else {
            &["logits"]
        };
        let outputs = self
            .request
            .run_selected(key, &inputs, &Default::default(), readbacks)?;
        let logits = if self.mode == SelectionMode::Greedy {
            self.pending_semantic = Some(
                outputs
                    .device
                    .get("logits")
                    .ok_or(FishError::InvalidShape(
                        "CubeCL semantic device logits missing",
                    ))?
                    .clone(),
            );
            SlowArLogits::DevicePending
        } else {
            SlowArLogits::Host(float_output(&outputs.host, "logits", self.vocab_size)?)
        };
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
        )
    }

    fn slow_decode(
        &mut self,
        previous_column: &[u32],
        active_len: usize,
    ) -> Result<SlowArForward<CubeClDeviceTensor>> {
        self.slow(
            &format!("slow_decode_{active_len}"),
            checked_tokens(previous_column.iter().copied())?,
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
                self.request
                    .sample_greedy(&pending, 0, self.vocab_size, history, params, eos_id)
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
        sample: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
        on_codebook: &mut dyn FnMut(usize) -> Result<()>,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<u32>> {
        let mut codes = vec![initial];
        for (index, history) in histories.iter().enumerate() {
            if is_cancelled() {
                return Err(FishError::InvalidTransition("synthesis cancelled"));
            }
            let codebook = index + 1;
            let previous = *codes.last().expect("Fast AR initial code present");
            let token = if self.mode == SelectionMode::Greedy {
                let mut inputs = NamedTensors::new();
                inputs.insert(
                    "previous_acoustic_code".into(),
                    TensorData::I32(checked_tokens([previous])?),
                );
                let outputs = self.request.run_selected(
                    &format!("fast_{codebook}"),
                    &inputs,
                    &Default::default(),
                    &[],
                )?;
                let device_logits =
                    outputs
                        .device
                        .get("fast_logits")
                        .ok_or(FishError::InvalidShape(
                            "CubeCL Fast AR device logits missing",
                        ))?;
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
            } else {
                let logits = self.decode_fast(codebook, previous)?;
                sample(&logits, history, params)?
            };
            if token as usize >= codebook_size {
                return Err(FishError::InvalidShape(
                    "Fast AR sample exceeds codebook size",
                ));
            }
            codes.push(token);
            on_codebook(codebook)?;
        }
        Ok(codes)
    }
}

struct CubeClNaiveBackend<'a, 'b> {
    request: &'a mut CubeClRequest<'b>,
    semantic_lut: &'a [i32],
    vocab_size: usize,
    codebook_logits_len: usize,
}

impl CubeClNaiveBackend<'_, '_> {
    fn forward(&mut self, key: &str, tokens: Vec<i32>) -> Result<NaiveArForward> {
        if self.request.cursor().slow_active_len > 0 {
            self.request.retire_blocking()?;
        }
        let mut input = NamedTensors::new();
        input.insert("tokens".into(), TensorData::I32(tokens));
        input.insert(
            "semantic_membership_lut".into(),
            TensorData::I32(self.semantic_lut.to_vec()),
        );
        let output = self.request.run(key, &input)?;
        Ok(NaiveArForward {
            semantic_logits: float_output(&output, "logits", self.vocab_size)?,
            acoustic_logits: float_output(&output, "codebook_logits", self.codebook_logits_len)?,
        })
    }
}

impl NaiveArBackend for CubeClNaiveBackend<'_, '_> {
    fn slow_prefill(&mut self, prompt: &EncodedConversation) -> Result<NaiveArForward> {
        self.forward(
            "slow_prefill",
            checked_tokens(prompt.channels.iter().flatten().copied())?,
        )
    }

    fn slow_decode(
        &mut self,
        previous_column: &[u32],
        active_len: usize,
    ) -> Result<NaiveArForward> {
        self.forward(
            &format!("slow_decode_{active_len}"),
            checked_tokens(previous_column.iter().copied())?,
        )
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
                    pending_semantic: None,
                };
                generate_dual_ar(
                    &self.config,
                    &mut backend,
                    prompt,
                    limit,
                    self.semantic_begin_id,
                    self.eos_id,
                    sampling,
                    &mut |logits, history, params| {
                        sample_host(&mut self.rng, self.mode, logits, history, params)
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
                };
                generate_naive_ar(
                    &self.config,
                    &mut backend,
                    prompt,
                    limit,
                    self.eos_id,
                    sampling,
                    &mut |logits, history, params| {
                        sample_host(&mut self.rng, self.mode, logits, history, params)
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
