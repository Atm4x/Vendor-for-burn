use crate::config::{FishModelConfig, ModelKind};
use crate::conversation::EncodedConversation;
use crate::error::{FishError, Result};
use crate::sampling::SamplingParams;
use crate::session::{initial_acoustic_code, GenerationSession, Phase};

pub struct SlowArForward<C> {
    pub logits: Vec<f32>,
    pub fast_context: C,
}

/// Executes the original DualAR ordering, independent of CPU or TensorRT tensor ownership.
/// The backend owns KV caches for the duration of one request.
pub trait DualArBackend {
    type FastContext;

    fn slow_prefill(
        &mut self,
        prompt: &EncodedConversation,
    ) -> Result<SlowArForward<Self::FastContext>>;
    fn slow_decode(
        &mut self,
        previous_column: &[u32],
        active_len: usize,
    ) -> Result<SlowArForward<Self::FastContext>>;
    fn reset_fast(&mut self) -> Result<()>;
    fn prime_fast(&mut self, context: Self::FastContext) -> Result<()>;
    fn decode_fast(&mut self, codebook_index: usize, previous_code: u32) -> Result<Vec<f32>>;

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
            let logits =
                self.decode_fast(codebook, *codes.last().expect("initial code present"))?;
            let token = sample(&logits, history, params)?;
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DualArTrace {
    pub slow_active_lengths: Vec<usize>,
    pub fast_active_lengths: Vec<usize>,
    pub fast_resets: usize,
}

pub fn generate_dual_ar<B, S>(
    config: &FishModelConfig,
    backend: &mut B,
    prompt: &EncodedConversation,
    max_new_tokens: usize,
    semantic_begin_id: u32,
    eos_id: u32,
    params: SamplingParams,
    sample: &mut S,
    trace: &mut DualArTrace,
    is_cancelled: &mut dyn FnMut() -> bool,
) -> Result<Vec<Vec<u32>>>
where
    B: DualArBackend,
    S: FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
{
    config.validate()?;
    params.validate()?;
    if config.kind != ModelKind::DualAr {
        return Err(FishError::Unsupported(
            "DualAR generation requires DualAr model",
        ));
    }
    let count = config.num_codebooks + 1;
    if prompt.channels.len() != count || prompt.channels[0].is_empty() {
        return Err(FishError::InvalidShape("invalid Dual-AR prompt channels"));
    }
    let prompt_len = prompt.channels[0].len();
    if prompt
        .channels
        .iter()
        .any(|channel| channel.len() != prompt_len)
        || prompt.vq_positions.len() != prompt_len
    {
        return Err(FishError::InvalidShape(
            "inconsistent prompt channel lengths",
        ));
    }
    let mut session = GenerationSession::new(config, prompt_len, max_new_tokens, eos_id)?;
    let mut result = vec![Vec::new(); count];
    *trace = DualArTrace::default();
    session.prefill(prompt_len)?;

    while session.phase != Phase::Finished {
        if is_cancelled() {
            return Err(FishError::InvalidTransition("synthesis cancelled"));
        }
        let active_len = prompt_len + result[0].len();
        let slow = if result[0].is_empty() {
            backend.slow_prefill(prompt)?
        } else {
            let previous = result
                .iter()
                .map(|channel| *channel.last().expect("nonempty emitted column"))
                .collect::<Vec<_>>();
            backend.slow_decode(&previous, active_len)?
        };
        trace.slow_active_lengths.push(active_len);
        let history = if result[0].is_empty() {
            Vec::new()
        } else {
            session
                .previous_window(0)?
                .iter()
                .map(|&token| token as usize)
                .collect()
        };
        let semantic = sample(&slow.logits, &history, params)?;
        session.begin_semantic_step()?;
        backend.reset_fast()?;
        trace.fast_resets += 1;
        backend.prime_fast(slow.fast_context)?;
        if session.next_fast_codebook()? != 0 {
            return Err(FishError::InvalidTransition("Fast AR step zero mismatch"));
        }
        trace.fast_active_lengths.push(1);
        let initial = initial_acoustic_code(semantic, semantic_begin_id, config.codebook_size)?;
        let mut histories = Vec::with_capacity(config.num_codebooks.saturating_sub(1));
        for index in 1..config.num_codebooks {
            let history = if result[0].is_empty() {
                Vec::new()
            } else {
                session
                    .previous_window(index + 1)?
                    .iter()
                    .map(|&token| token as usize)
                    .collect()
            };
            histories.push(history);
        }
        let acoustics = backend.sample_fast_codes(
            initial,
            config.codebook_size,
            &histories,
            params,
            sample,
            &mut |index| {
                if session.next_fast_codebook()? != index {
                    return Err(FishError::InvalidTransition("Fast AR step mismatch"));
                }
                trace.fast_active_lengths.push(index + 1);
                Ok(())
            },
            is_cancelled,
        )?;
        if acoustics.len() != config.num_codebooks
            || acoustics
                .iter()
                .any(|&token| token as usize >= config.codebook_size)
        {
            return Err(FishError::InvalidShape(
                "Fast AR sample exceeds codebook size",
            ));
        }
        let mut next = Vec::with_capacity(count);
        next.push(semantic);
        next.extend(acoustics);
        session.finish_step(next.clone())?;
        for (channel, token) in result.iter_mut().zip(next) {
            channel.push(token);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TransformerDims;

    struct ScriptedBackend {
        steps: Vec<String>,
    }

    impl DualArBackend for ScriptedBackend {
        type FastContext = usize;
        fn slow_prefill(&mut self, prompt: &EncodedConversation) -> Result<SlowArForward<usize>> {
            self.steps.push(format!("prefill:{}", prompt.length()));
            Ok(SlowArForward {
                logits: vec![0., 1.],
                fast_context: 11,
            })
        }
        fn slow_decode(
            &mut self,
            previous: &[u32],
            active_len: usize,
        ) -> Result<SlowArForward<usize>> {
            self.steps.push(format!("slow:{active_len}:{previous:?}"));
            Ok(SlowArForward {
                logits: vec![1., 0.],
                fast_context: 22,
            })
        }
        fn reset_fast(&mut self) -> Result<()> {
            self.steps.push("reset".into());
            Ok(())
        }
        fn prime_fast(&mut self, context: usize) -> Result<()> {
            self.steps.push(format!("prime:{context}"));
            Ok(())
        }
        fn decode_fast(&mut self, index: usize, previous: u32) -> Result<Vec<f32>> {
            self.steps.push(format!("fast:{index}:{previous}"));
            Ok(vec![0., 1., 0.])
        }
    }

    fn config() -> FishModelConfig {
        let dims = TransformerDims {
            dim: 8,
            n_heads: 2,
            n_kv_heads: 1,
            layers: 1,
            intermediate_size: 16,
            qkv_bias: false,
        };
        FishModelConfig {
            kind: ModelKind::DualAr,
            slow: dims.clone(),
            fast: Some(dims),
            vocab_size: 2,
            num_codebooks: 3,
            codebook_size: 3,
            max_seq_len: 10,
            rope_base: 10000.,
            norm_eps: 1e-5,
            tie_word_embeddings: true,
            share_codebook_embeddings: true,
        }
    }

    fn prompt() -> EncodedConversation {
        EncodedConversation {
            channels: vec![vec![1, 0]; 4],
            vq_positions: vec![false; 2],
        }
    }

    #[test]
    fn source_order_prefill_slow_decode_and_prime_not_sample_step_zero() {
        let mut backend = ScriptedBackend { steps: Vec::new() };
        let mut trace = DualArTrace::default();
        let output = generate_dual_ar(
            &config(),
            &mut backend,
            &prompt(),
            2,
            1,
            0,
            SamplingParams::default(),
            &mut |logits, _, _| Ok(logits.iter().position(|v| *v == 1.).unwrap() as u32),
            &mut trace,
            &mut || false,
        )
        .unwrap();
        assert_eq!(output, [vec![1, 0], vec![0, 0], vec![1, 1], vec![1, 1]]);
        assert_eq!(
            backend.steps,
            [
                "prefill:2",
                "reset",
                "prime:11",
                "fast:1:0",
                "fast:2:1",
                "slow:3:[1, 0, 1, 1]",
                "reset",
                "prime:22",
                "fast:1:0",
                "fast:2:1"
            ]
        );
        assert_eq!(trace.slow_active_lengths, [2, 3]);
        assert_eq!(trace.fast_active_lengths, [1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn invalid_fast_code_is_rejected_before_submitting_later_codebooks() {
        let mut backend = ScriptedBackend { steps: Vec::new() };
        let mut trace = DualArTrace::default();
        let err = generate_dual_ar(
            &config(),
            &mut backend,
            &prompt(),
            2,
            1,
            0,
            SamplingParams::default(),
            &mut |logits, _, _| Ok(if logits.len() == 3 { 99 } else { 1 }),
            &mut trace,
            &mut || false,
        )
        .unwrap_err();
        assert!(matches!(err, FishError::InvalidShape(_)));
        assert_eq!(
            backend.steps,
            ["prefill:2", "reset", "prime:11", "fast:1:0"]
        );
        assert_eq!(trace.fast_active_lengths, [1]);
    }

    #[test]
    fn eos_on_first_step_still_completes_fast_before_stopping() {
        let mut backend = ScriptedBackend { steps: Vec::new() };
        let mut trace = DualArTrace::default();
        let result = generate_dual_ar(
            &config(),
            &mut backend,
            &prompt(),
            3,
            1,
            1,
            SamplingParams::default(),
            &mut |logits, _, _| Ok(logits.iter().position(|v| *v == 1.).unwrap() as u32),
            &mut trace,
            &mut || false,
        )
        .unwrap();
        assert!(result.iter().all(|v| v.len() == 1));
        assert_eq!(
            backend.steps,
            ["prefill:2", "reset", "prime:11", "fast:1:0", "fast:2:1"]
        );
    }

    #[test]
    fn cancellation_stops_before_later_fast_network() {
        let mut backend = ScriptedBackend { steps: Vec::new() };
        let mut cancelled_checks = 0;
        let mut trace = DualArTrace::default();
        let err = generate_dual_ar(
            &config(),
            &mut backend,
            &prompt(),
            2,
            1,
            0,
            SamplingParams::default(),
            &mut |logits, _, _| Ok(logits.iter().position(|v| *v == 1.).unwrap() as u32),
            &mut trace,
            &mut || {
                cancelled_checks += 1;
                cancelled_checks > 2
            },
        )
        .err()
        .unwrap();
        assert_eq!(err, FishError::InvalidTransition("synthesis cancelled"));
        assert_eq!(
            backend.steps,
            ["prefill:2", "reset", "prime:11", "fast:1:0"]
        );
    }

    #[test]
    fn input_failure_happens_before_any_backend_call() {
        let mut backend = ScriptedBackend { steps: Vec::new() };
        let mut bad = prompt();
        bad.channels[1].pop();
        let mut trace = DualArTrace::default();
        assert!(generate_dual_ar(
            &config(),
            &mut backend,
            &bad,
            2,
            1,
            0,
            SamplingParams::default(),
            &mut |_, _, _| Ok(0),
            &mut trace,
            &mut || false
        )
        .is_err());
        assert!(backend.steps.is_empty());
    }
}
