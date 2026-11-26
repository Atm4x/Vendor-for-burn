use crate::{
    config::{FishModelConfig, ModelKind},
    conversation::EncodedConversation,
    error::{FishError, Result},
    sampling::SamplingParams,
};

pub enum NaiveArForward {
    Host {
        semantic_logits: Vec<f32>,
        acoustic_logits: Vec<f32>,
    },
    DevicePending,
}

/// The Naive model emits every acoustic codebook from the same Slow AR forward.
/// A backend owns its KV state; this controller only owns the generation order.
pub trait NaiveArBackend {
    fn slow_prefill(&mut self, prompt: &EncodedConversation) -> Result<NaiveArForward>;
    fn slow_decode(&mut self, previous_column: &[u32], active_len: usize)
        -> Result<NaiveArForward>;

    fn sample_semantic(
        &mut self,
        forward: &NaiveArForward,
        params: SamplingParams,
        sample: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
    ) -> Result<u32> {
        match forward {
            NaiveArForward::Host {
                semantic_logits, ..
            } => sample(semantic_logits, &[], params),
            NaiveArForward::DevicePending => Err(FishError::InvalidTransition(
                "Naive backend must sample device logits itself",
            )),
        }
    }

    fn sample_acoustic(
        &mut self,
        forward: &NaiveArForward,
        codebook: usize,
        codebook_size: usize,
        history: &[usize],
        params: SamplingParams,
        sample: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
    ) -> Result<u32> {
        match forward {
            NaiveArForward::Host {
                acoustic_logits, ..
            } => {
                let start = codebook * codebook_size;
                sample(
                    &acoustic_logits[start..start + codebook_size],
                    history,
                    params,
                )
            }
            NaiveArForward::DevicePending => Err(FishError::InvalidTransition(
                "Naive backend must sample device logits itself",
            )),
        }
    }
}

pub fn naive_token_limit(
    config: &FishModelConfig,
    prompt_len: usize,
    requested: usize,
) -> Result<usize> {
    if prompt_len == 0 || prompt_len >= config.max_seq_len {
        return Err(FishError::InvalidPosition {
            position: prompt_len,
            capacity: config.max_seq_len,
        });
    }
    let remaining = config.max_seq_len - prompt_len;
    Ok(if requested == 0 {
        remaining
    } else {
        requested.min(remaining)
    })
}

pub fn generate_naive_ar<B, S>(
    config: &FishModelConfig,
    backend: &mut B,
    prompt: &EncodedConversation,
    max_new_tokens: usize,
    eos_id: u32,
    params: SamplingParams,
    sample: &mut S,
    is_cancelled: &mut dyn FnMut() -> bool,
) -> Result<Vec<Vec<u32>>>
where
    B: NaiveArBackend,
    S: FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
{
    config.validate()?;
    params.validate()?;
    if config.kind != ModelKind::Naive {
        return Err(FishError::Unsupported(
            "Naive generation requires Naive model",
        ));
    }
    let channels = config.num_codebooks + 1;
    if prompt.channels.len() != channels || prompt.channels[0].is_empty() {
        return Err(FishError::InvalidShape("invalid Naive prompt"));
    }
    let prompt_len = prompt.channels[0].len();
    if prompt.vq_positions.len() != prompt_len
        || prompt.channels.iter().any(|c| c.len() != prompt_len)
    {
        return Err(FishError::InvalidShape("invalid Naive prompt"));
    }
    let limit = naive_token_limit(config, prompt_len, max_new_tokens)?;
    let mut generated = vec![Vec::new(); channels];
    let semantic_params = SamplingParams {
        temperature: 0.1,
        top_p: 0.1,
        repetition_penalty: 1.0,
    };
    for step in 0..limit {
        if is_cancelled() {
            return Err(FishError::InvalidTransition("synthesis cancelled"));
        }
        let outputs = if step == 0 {
            backend.slow_prefill(prompt)?
        } else {
            let previous = generated
                .iter()
                .map(|c| *c.last().expect("complete previous column"))
                .collect::<Vec<_>>();
            backend.slow_decode(&previous, prompt_len + step)?
        };
        if let NaiveArForward::Host {
            semantic_logits,
            acoustic_logits,
        } = &outputs
        {
            if semantic_logits.len() != config.vocab_size
                || acoustic_logits.len() != config.num_codebooks * config.codebook_size
            {
                return Err(FishError::InvalidShape(
                    "Naive semantic or acoustic logits shape invalid",
                ));
            }
        }
        let semantic = backend.sample_semantic(&outputs, semantic_params, sample)?;
        if semantic as usize >= config.vocab_size {
            return Err(FishError::InvalidSampling(
                "Naive semantic token exceeds vocabulary",
            ));
        }
        let mut row = Vec::with_capacity(channels);
        row.push(semantic);
        for cb in 0..config.num_codebooks {
            if is_cancelled() {
                return Err(FishError::InvalidTransition("synthesis cancelled"));
            }
            let history = if step == 0 {
                Vec::new()
            } else {
                let mut prior = generated[cb + 1]
                    .iter()
                    .skip(1)
                    .rev()
                    .take(16)
                    .copied()
                    .collect::<Vec<_>>();
                prior.reverse();
                let mut prior = prior
                    .into_iter()
                    .map(|token| token as usize)
                    .collect::<Vec<_>>();
                prior.resize(16, 0);
                prior
            };
            let token = backend.sample_acoustic(
                &outputs,
                cb,
                config.codebook_size,
                &history,
                params,
                sample,
            )?;
            if token as usize >= config.codebook_size {
                return Err(FishError::InvalidSampling(
                    "Naive acoustic token exceeds codebook",
                ));
            }
            row.push(token);
        }
        for (dst, token) in generated.iter_mut().zip(row) {
            dst.push(token);
        }
        if semantic == eos_id {
            break;
        }
    }
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scripted {
        calls: Vec<String>,
    }

    impl NaiveArBackend for Scripted {
        fn slow_prefill(&mut self, prompt: &EncodedConversation) -> Result<NaiveArForward> {
            self.calls.push(format!("prefill:{}", prompt.length()));
            Ok(NaiveArForward::Host {
                semantic_logits: vec![0.; 4],
                acoustic_logits: vec![0.; 6],
            })
        }
        fn slow_decode(&mut self, previous: &[u32], active_len: usize) -> Result<NaiveArForward> {
            self.calls.push(format!("decode:{active_len}:{previous:?}"));
            Ok(NaiveArForward::Host {
                semantic_logits: vec![0.; 4],
                acoustic_logits: vec![0.; 6],
            })
        }
    }

    struct DeviceScripted {
        calls: Vec<String>,
        semantic_steps: usize,
    }

    impl NaiveArBackend for DeviceScripted {
        fn slow_prefill(&mut self, _: &EncodedConversation) -> Result<NaiveArForward> {
            self.calls.push("prefill".into());
            Ok(NaiveArForward::DevicePending)
        }
        fn slow_decode(&mut self, previous: &[u32], _: usize) -> Result<NaiveArForward> {
            self.calls.push(format!("decode:{previous:?}"));
            Ok(NaiveArForward::DevicePending)
        }
        fn sample_semantic(
            &mut self,
            _: &NaiveArForward,
            params: SamplingParams,
            _: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
        ) -> Result<u32> {
            assert_eq!(params.top_p, 0.1);
            assert_eq!(params.temperature, 0.1);
            assert_eq!(params.repetition_penalty, 1.0);
            self.semantic_steps += 1;
            Ok(if self.semantic_steps == 2 { 3 } else { 1 })
        }
        fn sample_acoustic(
            &mut self,
            _: &NaiveArForward,
            codebook: usize,
            _: usize,
            history: &[usize],
            _: SamplingParams,
            _: &mut dyn FnMut(&[f32], &[usize], SamplingParams) -> Result<u32>,
        ) -> Result<u32> {
            self.calls
                .push(format!("codebook:{codebook}:history:{history:?}"));
            Ok(codebook as u32)
        }
    }

    #[test]
    fn naive_device_forward_uses_backend_sampling_and_keeps_eos_column() {
        let mut c = FishModelConfig::synthetic_dual_ar_fixture();
        c.kind = ModelKind::Naive;
        c.fast = None;
        c.vocab_size = 4;
        c.num_codebooks = 2;
        c.codebook_size = 3;
        c.max_seq_len = 8;
        let prompt = EncodedConversation {
            channels: vec![vec![1], vec![0], vec![0]],
            vq_positions: vec![false],
        };
        let mut backend = DeviceScripted {
            calls: Vec::new(),
            semantic_steps: 0,
        };
        let out = generate_naive_ar(
            &c,
            &mut backend,
            &prompt,
            3,
            3,
            SamplingParams::default(),
            &mut |_, _, _| panic!("device-backed Naive must never sample host logits"),
            &mut || false,
        )
        .unwrap();
        assert_eq!(out, vec![vec![1, 3], vec![0, 0], vec![1, 1]]);
        assert_eq!(
            backend.calls,
            [
                "prefill",
                "codebook:0:history:[]",
                "codebook:1:history:[]",
                "decode:[1, 0, 1]",
                "codebook:0:history:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
                "codebook:1:history:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
            ]
        );
    }

    #[test]
    fn naive_shared_controller_preserves_single_forward_and_eos_column() {
        let mut c = FishModelConfig::synthetic_dual_ar_fixture();
        c.kind = ModelKind::Naive;
        c.fast = None;
        c.vocab_size = 4;
        c.num_codebooks = 2;
        c.codebook_size = 3;
        c.max_seq_len = 8;
        let prompt = EncodedConversation {
            channels: vec![vec![1], vec![0], vec![0]],
            vq_positions: vec![false],
        };
        let mut backend = Scripted { calls: Vec::new() };
        let out = generate_naive_ar(
            &c,
            &mut backend,
            &prompt,
            3,
            0,
            SamplingParams::default(),
            &mut |_, _, _| Ok(0),
            &mut || false,
        )
        .unwrap();
        assert_eq!(out, vec![vec![0], vec![0], vec![0]]);
        assert_eq!(backend.calls, ["prefill:1"]);
    }

    #[test]
    fn naive_shared_controller_passes_previous_column_and_checks_cancellation() {
        let mut c = FishModelConfig::synthetic_dual_ar_fixture();
        c.kind = ModelKind::Naive;
        c.fast = None;
        c.vocab_size = 4;
        c.num_codebooks = 2;
        c.codebook_size = 3;
        c.max_seq_len = 8;
        let prompt = EncodedConversation {
            channels: vec![vec![1], vec![0], vec![0]],
            vq_positions: vec![false],
        };
        let mut backend = Scripted { calls: Vec::new() };
        let out = generate_naive_ar(
            &c,
            &mut backend,
            &prompt,
            2,
            3,
            SamplingParams::default(),
            &mut |_, _, _| Ok(1),
            &mut || false,
        )
        .unwrap();
        assert_eq!(out, vec![vec![1; 2], vec![1; 2], vec![1; 2]]);
        assert_eq!(backend.calls, ["prefill:1", "decode:2:[1, 1, 1]"]);
        let mut polls = 0;
        let err = generate_naive_ar(
            &c,
            &mut backend,
            &prompt,
            2,
            3,
            SamplingParams::default(),
            &mut |_, _, _| Ok(1),
            &mut || {
                polls += 1;
                polls == 2
            },
        )
        .unwrap_err();
        assert_eq!(err, FishError::InvalidTransition("synthesis cancelled"));
    }
}
