use crate::config::FishModelConfig;
use crate::error::{FishError, Result};
use crate::kv::ArKvState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastStepKind {
    PrimeCacheAndDeriveInitialCode,
    SampleFromFastLogits,
}

pub fn initial_acoustic_code(
    semantic_id: u32,
    semantic_begin_id: u32,
    codebook_size: usize,
) -> Result<u32> {
    if codebook_size == 0 {
        return Err(FishError::InvalidConfig("empty acoustic codebook"));
    }
    let index = semantic_id.saturating_sub(semantic_begin_id);
    if index as usize >= codebook_size {
        return Err(FishError::InvalidShape(
            "semantic-derived code outside acoustic codebook",
        ));
    }
    Ok(index)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    NeedsPrefill,
    Ready,
    FastRunning,
    Finished,
}

#[derive(Debug, Clone)]
pub struct GenerationSession {
    pub kv: ArKvState,
    pub phase: Phase,
    pub num_codebooks: usize,
    pub prompt_len: usize,
    pub max_new_tokens: usize,
    pub eos_token: u32,
    emitted: Vec<Vec<u32>>,
    subsequent_history: Vec<Vec<u32>>,
}

impl GenerationSession {
    pub fn new(
        config: &FishModelConfig,
        prompt_len: usize,
        max_new_tokens: usize,
        eos_token: u32,
    ) -> Result<Self> {
        config.validate()?;
        if prompt_len == 0 || prompt_len >= config.max_seq_len {
            return Err(FishError::InvalidPosition {
                position: prompt_len,
                capacity: config.max_seq_len,
            });
        }
        let remaining = config.max_seq_len - prompt_len;
        let cap = if max_new_tokens == 0 {
            remaining
        } else {
            max_new_tokens.min(remaining)
        };
        Ok(Self {
            kv: ArKvState::new(config.max_seq_len, config.num_codebooks)?,
            phase: Phase::NeedsPrefill,
            num_codebooks: config.num_codebooks,
            prompt_len,
            max_new_tokens: cap,
            eos_token,
            emitted: Vec::new(),
            subsequent_history: Vec::new(),
        })
    }

    pub fn prefill(&mut self, prompt_len: usize) -> Result<()> {
        if self.phase != Phase::NeedsPrefill {
            return Err(FishError::InvalidTransition("prefill already started"));
        }
        if prompt_len != self.prompt_len {
            return Err(FishError::InvalidShape(
                "prefill length differs from session construction",
            ));
        }
        self.kv.prefill(prompt_len)?;
        self.phase = Phase::Ready;
        Ok(())
    }

    pub fn begin_semantic_step(&mut self) -> Result<()> {
        if self.phase != Phase::Ready {
            return Err(FishError::InvalidTransition(
                "step cannot start from current phase",
            ));
        }
        if self.emitted.len() >= self.max_new_tokens {
            return Err(FishError::InvalidTransition("generation limit reached"));
        }
        if !self.emitted.is_empty() {
            self.kv.next_slow()?;
        }
        self.kv.start_fast()?;
        self.phase = Phase::FastRunning;
        Ok(())
    }

    pub fn next_fast_codebook(&mut self) -> Result<usize> {
        if self.phase != Phase::FastRunning {
            return Err(FishError::InvalidTransition("not generating codebooks"));
        }
        let idx = self.kv.fast.active_len();
        self.kv.fast_step()?;
        Ok(idx)
    }

    pub fn next_fast_step(&mut self) -> Result<(usize, FastStepKind)> {
        let index = self.next_fast_codebook()?;
        let action = if index == 0 {
            FastStepKind::PrimeCacheAndDeriveInitialCode
        } else {
            FastStepKind::SampleFromFastLogits
        };
        Ok((index, action))
    }

    pub fn finish_step(&mut self, tokens: Vec<u32>) -> Result<()> {
        if self.phase != Phase::FastRunning || self.kv.fast.active_len() != self.num_codebooks {
            return Err(FishError::InvalidTransition(
                "Fast codebook steps incomplete",
            ));
        }
        if tokens.len() != 1 + self.num_codebooks {
            return Err(FishError::InvalidShape(
                "expected semantic + acoustic codebooks",
            ));
        }
        let eos = tokens[0] == self.eos_token;
        if !self.emitted.is_empty() {
            self.subsequent_history.push(tokens.clone());
        }
        self.emitted.push(tokens);
        self.phase = if eos || self.emitted.len() >= self.max_new_tokens {
            Phase::Finished
        } else {
            Phase::Ready
        };
        Ok(())
    }

    pub fn previous_window(&self, codebook: usize) -> Result<[u32; 16]> {
        if codebook >= 1 + self.num_codebooks {
            return Err(FishError::InvalidShape("invalid codebook channel"));
        }
        let mut result = [0_u32; 16];
        if self.subsequent_history.len() < 16 {
            for (i, t) in self.subsequent_history.iter().enumerate() {
                result[i] = t[codebook];
            }
        } else {
            for (i, t) in self
                .subsequent_history
                .iter()
                .rev()
                .take(16)
                .rev()
                .enumerate()
            {
                result[i] = t[codebook];
            }
        }
        Ok(result)
    }

    pub fn emitted(&self) -> &[Vec<u32>] {
        &self.emitted
    }
    pub fn reset_for_prompt(&mut self, prompt_len: usize, max_new_tokens: usize) -> Result<()> {
        if prompt_len == 0 || prompt_len >= self.kv.slow.capacity() {
            return Err(FishError::InvalidPosition {
                position: prompt_len,
                capacity: self.kv.slow.capacity(),
            });
        }
        self.reset();
        self.prompt_len = prompt_len;
        let remaining = self.kv.slow.capacity() - prompt_len;
        self.max_new_tokens = if max_new_tokens == 0 {
            remaining
        } else {
            max_new_tokens.min(remaining)
        };
        Ok(())
    }

    pub fn reset(&mut self) {
        self.kv.reset_request();
        self.phase = Phase::NeedsPrefill;
        self.emitted.clear();
        self.subsequent_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn session(max: usize) -> GenerationSession {
        GenerationSession::new(&FishModelConfig::synthetic_dual_ar_fixture(), 6, max, 4).unwrap()
    }
    fn generate_step(s: &mut GenerationSession, semantic: u32) {
        s.begin_semantic_step().unwrap();
        for i in 0..s.num_codebooks {
            assert_eq!(s.next_fast_codebook().unwrap(), i);
        }
        s.finish_step(vec![semantic; 9]).unwrap();
    }
    #[test]
    fn first_acoustic_code_clamps_negative_semantic_offset() {
        assert_eq!(initial_acoustic_code(100, 500, 1024).unwrap(), 0);
        assert_eq!(initial_acoustic_code(500, 500, 1024).unwrap(), 0);
        assert_eq!(initial_acoustic_code(501, 500, 1024).unwrap(), 1);
        assert_eq!(initial_acoustic_code(1523, 500, 1024).unwrap(), 1023);
        assert!(initial_acoustic_code(1524, 500, 1024).is_err());
        assert!(initial_acoustic_code(1, 1, 0).is_err());
    }

    #[test]
    fn normal_dual_ar_primes_first_fast_step_without_sampling() {
        let mut session = session(3);
        session.prefill(6).unwrap();
        session.begin_semantic_step().unwrap();
        assert_eq!(
            session.next_fast_step().unwrap(),
            (0, FastStepKind::PrimeCacheAndDeriveInitialCode)
        );
        for step in 1..session.num_codebooks {
            assert_eq!(
                session.next_fast_step().unwrap(),
                (step, FastStepKind::SampleFromFastLogits)
            );
        }
        assert!(session.next_fast_step().is_err());
    }

    #[test]
    fn prefill_decode_then_eos() {
        let mut s = session(5);
        s.prefill(6).unwrap();
        generate_step(&mut s, 10);
        assert_eq!(s.kv.slow.active_len(), 6);
        generate_step(&mut s, 4);
        assert_eq!(s.phase, Phase::Finished);
        assert_eq!(s.kv.slow.active_len(), 7);
    }
    #[test]
    fn eos_on_first_prefill_sample() {
        let mut s = session(5);
        s.prefill(6).unwrap();
        generate_step(&mut s, 4);
        assert_eq!(s.emitted().len(), 1);
        assert!(s.begin_semantic_step().is_err());
    }
    #[test]
    fn history_does_not_contain_prefill_prediction() {
        let mut s = session(8);
        s.prefill(6).unwrap();
        generate_step(&mut s, 99);
        assert_eq!(s.previous_window(0).unwrap(), [0; 16]);
        generate_step(&mut s, 20);
        assert_eq!(s.previous_window(0).unwrap()[0], 20);
    }
    #[test]
    fn prevent_partial_fast_step() {
        let mut s = session(5);
        s.prefill(6).unwrap();
        s.begin_semantic_step().unwrap();
        s.next_fast_codebook().unwrap();
        assert!(s.finish_step(vec![3; 9]).is_err());
    }
    #[test]
    fn per_request_reset() {
        let mut s = session(5);
        s.prefill(6).unwrap();
        generate_step(&mut s, 10);
        s.reset_for_prompt(2, 5).unwrap();
        assert_eq!(s.phase, Phase::NeedsPrefill);
        assert_eq!(s.kv.slow.active_len(), 0);
        s.prefill(2).unwrap();
        generate_step(&mut s, 11);
        assert_eq!(s.kv.slow.active_len(), 2);
        assert_eq!(s.emitted().len(), 1);
    }
    #[test]
    fn prompt_length_cannot_drift() {
        let mut s = session(5);
        assert!(s.prefill(2).is_err());
        s.prefill(6).unwrap();
    }
    #[test]
    fn limit_is_clamped_to_capacity() {
        let c = FishModelConfig::synthetic_dual_ar_fixture();
        let s = GenerationSession::new(&c, 255, 100, 4).unwrap();
        assert_eq!(s.max_new_tokens, 1);
    }
}
