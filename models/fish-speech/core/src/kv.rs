use crate::error::{FishError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCursor {
    capacity: usize,
    active_len: usize,
    initialized: bool,
}

impl KvCursor {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(FishError::InvalidConfig("KV capacity must be positive"));
        }
        Ok(Self {
            capacity,
            active_len: 0,
            initialized: false,
        })
    }

    pub fn reset(&mut self) {
        self.active_len = 0;
        self.initialized = false;
    }

    pub fn write_contiguous(&mut self, start: usize, count: usize) -> Result<usize> {
        if count == 0 {
            return Err(FishError::InvalidPosition {
                position: start,
                capacity: self.capacity,
            });
        }
        let end = start.checked_add(count).ok_or(FishError::InvalidPosition {
            position: start,
            capacity: self.capacity,
        })?;
        if start != self.active_len || end > self.capacity {
            return Err(FishError::InvalidPosition {
                position: end,
                capacity: self.capacity,
            });
        }
        self.active_len = end;
        self.initialized = true;
        Ok(end)
    }

    pub fn active_len(&self) -> usize {
        self.active_len
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn initialized(&self) -> bool {
        self.initialized
    }
    pub fn mask_key_allowed(&self, query_pos: usize, key_pos: usize) -> bool {
        self.initialized
            && query_pos < self.active_len
            && key_pos < self.active_len
            && key_pos <= query_pos
    }
}

#[derive(Debug, Clone)]
pub struct ArKvState {
    pub slow: KvCursor,
    pub fast: KvCursor,
    pub slow_steps: usize,
}

impl ArKvState {
    pub fn new(slow_capacity: usize, codebooks: usize) -> Result<Self> {
        Ok(Self {
            slow: KvCursor::new(slow_capacity)?,
            fast: KvCursor::new(codebooks)?,
            slow_steps: 0,
        })
    }

    pub fn prefill(&mut self, prompt_len: usize) -> Result<()> {
        if self.slow.initialized() {
            return Err(FishError::InvalidTransition("prefill already performed"));
        }
        self.slow.write_contiguous(0, prompt_len)?;
        Ok(())
    }

    pub fn next_slow(&mut self) -> Result<()> {
        if !self.slow.initialized() {
            return Err(FishError::InvalidTransition("Slow prefill missing"));
        }
        let pos = self.slow.active_len();
        self.slow.write_contiguous(pos, 1)?;
        self.slow_steps += 1;
        Ok(())
    }

    pub fn start_fast(&mut self) -> Result<()> {
        if !self.slow.initialized() {
            return Err(FishError::InvalidTransition("Slow cache not initialized"));
        }
        self.fast.reset();
        Ok(())
    }

    pub fn fast_step(&mut self) -> Result<usize> {
        let pos = self.fast.active_len();
        self.fast.write_contiguous(pos, 1)
    }

    pub fn reset_request(&mut self) {
        self.slow.reset();
        self.fast.reset();
        self.slow_steps = 0;
    }
}

pub fn reference_attention_1d(
    query: f32,
    keys: &[f32],
    values: &[f32],
    active_len: usize,
) -> Result<f32> {
    if active_len == 0 || active_len > keys.len() || active_len > values.len() {
        return Err(FishError::InvalidShape("active prefix outside input"));
    }
    let scores: Vec<f32> = keys[..active_len].iter().map(|k| query * k).collect();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|v| (v - max).exp()).collect();
    let z: f32 = exp.iter().sum();
    Ok(exp
        .iter()
        .zip(values[..active_len].iter())
        .map(|(s, v)| s * v)
        .sum::<f32>()
        / z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_then_decode_positions() {
        let mut state = ArKvState::new(16, 8).unwrap();
        state.prefill(7).unwrap();
        assert_eq!(state.slow.active_len(), 7);
        state.next_slow().unwrap();
        assert_eq!(state.slow.active_len(), 8);
        state.next_slow().unwrap();
        assert_eq!(state.slow.active_len(), 9);
    }
    #[test]
    fn reject_gaps_or_overflow() {
        let mut k = KvCursor::new(4).unwrap();
        assert!(k.write_contiguous(1, 1).is_err());
        k.write_contiguous(0, 4).unwrap();
        assert!(k.write_contiguous(4, 1).is_err());
    }
    #[test]
    fn reject_empty_prefill() {
        assert!(ArKvState::new(8, 8).unwrap().prefill(0).is_err());
    }
    #[test]
    fn causal_mask_uses_actual_positions() {
        let mut k = KvCursor::new(1024).unwrap();
        k.write_contiguous(0, 10).unwrap();
        assert!(k.mask_key_allowed(9, 9));
        assert!(!k.mask_key_allowed(9, 10));
        assert!(!k.mask_key_allowed(4, 5));
    }
    #[test]
    fn poisoned_tail_is_never_read() {
        let keys = [0.1, 0.2, f32::NAN, f32::NAN];
        let values = [1.0, 2.0, f32::NAN, f32::NAN];
        let safe = reference_attention_1d(0.5, &keys, &values, 2).unwrap();
        let baseline = reference_attention_1d(0.5, &keys[..2], &values[..2], 2).unwrap();
        assert!((safe - baseline).abs() < 1e-7);
        assert!(safe.is_finite());
    }
    #[test]
    fn fast_resets_each_slow_step() {
        let mut state = ArKvState::new(12, 3).unwrap();
        state.prefill(4).unwrap();
        state.start_fast().unwrap();
        for i in 1..=3 {
            assert_eq!(state.fast_step().unwrap(), i);
        }
        assert!(state.fast_step().is_err());
        state.next_slow().unwrap();
        state.start_fast().unwrap();
        assert_eq!(state.fast_step().unwrap(), 1);
        assert_eq!(state.slow.active_len(), 5);
    }
    #[test]
    fn reset_request_has_no_cache_visibility() {
        let mut state = ArKvState::new(10, 2).unwrap();
        state.prefill(9).unwrap();
        state.fast_step().unwrap();
        state.reset_request();
        assert_eq!(state.slow.active_len(), 0);
        assert_eq!(state.fast.active_len(), 0);
        assert!(!state.slow.mask_key_allowed(0, 0));
        state.prefill(1).unwrap();
        assert!(!state.slow.mask_key_allowed(0, 1));
    }
}
