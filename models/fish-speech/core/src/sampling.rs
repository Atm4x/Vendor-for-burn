use crate::error::{FishError, Result};

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.7,
            repetition_penalty: 1.5,
        }
    }
}

impl SamplingParams {
    pub fn validate(self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 || self.temperature >= 2.0 {
            return Err(FishError::InvalidSampling(
                "temperature must be within (0,2)",
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(FishError::InvalidSampling("top_p must be within (0,1]"));
        }
        if !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
            || self.repetition_penalty >= 2.0
        {
            return Err(FishError::InvalidSampling(
                "repetition_penalty must be within (0,2)",
            ));
        }
        Ok(())
    }
}

pub fn logits_to_probs(
    logits: &[f32],
    history: &[usize],
    params: SamplingParams,
) -> Result<Vec<f32>> {
    params.validate()?;
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err(FishError::InvalidSampling(
            "logits must be nonempty and finite",
        ));
    }
    let mut adjusted = logits.to_vec();
    let mut seen = vec![false; adjusted.len()];
    for &idx in history {
        let already = *seen.get(idx).ok_or(FishError::InvalidSampling(
            "history token exceeds vocabulary",
        ))?;
        if already {
            continue;
        }
        seen[idx] = true;
        let current = adjusted.get_mut(idx).ok_or(FishError::InvalidSampling(
            "history token exceeds vocabulary",
        ))?;
        if *current < 0.0 {
            *current *= params.repetition_penalty;
        } else {
            *current /= params.repetition_penalty;
        }
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| adjusted[b].total_cmp(&adjusted[a]).then_with(|| a.cmp(&b)));
    let max = adjusted[indices[0]];
    let original_exp: Vec<f64> = indices
        .iter()
        .map(|&i| ((adjusted[i] - max) as f64).exp())
        .collect();
    let original_sum: f64 = original_exp.iter().sum();
    let mut cumulative = 0.0_f64;
    let mut keep = vec![false; logits.len()];
    for (rank, &idx) in indices.iter().enumerate() {
        cumulative += original_exp[rank] / original_sum;
        keep[idx] = rank == 0 || cumulative <= params.top_p as f64;
    }
    let temperature = params.temperature.max(1.0e-5);
    let scaled_max = indices
        .iter()
        .filter(|&&i| keep[i])
        .map(|&i| adjusted[i] / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights = vec![0.0_f64; logits.len()];
    for (i, w) in weights.iter_mut().enumerate() {
        if keep[i] {
            *w = ((adjusted[i] / temperature - scaled_max) as f64).exp();
        }
    }
    let total: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|v| (v / total) as f32).collect())
}

#[derive(Debug, Clone)]
pub struct SessionRng {
    state: u64,
}

impl SessionRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545f4914f6cdd1d)
    }
    pub fn uniform_open_01(&mut self) -> f64 {
        let raw = self.next_u64() >> 11;
        ((raw as f64) + 0.5) / ((1_u64 << 53) as f64)
    }
    pub fn sample_exponential_race(&mut self, probs: &[f32]) -> Result<usize> {
        if probs.is_empty() || probs.iter().any(|&p| p < 0.0 || !p.is_finite()) {
            return Err(FishError::InvalidSampling("probabilities invalid"));
        }
        let mut best = f64::NEG_INFINITY;
        let mut winner = 0;
        for (i, &p) in probs.iter().enumerate() {
            let exp = -self.uniform_open_01().ln();
            let score = p as f64 / exp;
            if score > best {
                best = score;
                winner = i;
            }
        }
        if best <= 0.0 {
            return Err(FishError::InvalidSampling("all probabilities zero"));
        }
        Ok(winner)
    }
}

pub fn greedy(probs: &[f32]) -> Result<usize> {
    probs
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(i, _)| i)
        .ok_or(FishError::InvalidSampling(
            "empty or nonfinite probabilities",
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_logits_same_uniform_no_filter() {
        let p = logits_to_probs(
            &[0.0, 0.0, 0.0],
            &[],
            SamplingParams {
                top_p: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(p.iter().all(|x| (*x - 1.0 / 3.0).abs() < 1e-6));
    }
    #[test]
    fn fish_top_p_does_not_shift_cutoff() {
        let p = logits_to_probs(
            &[2.0, 1.0, 0.0],
            &[],
            SamplingParams {
                top_p: 0.7,
                temperature: 1.0,
                repetition_penalty: 1.0,
            },
        )
        .unwrap();
        assert_eq!(p, [1.0, 0.0, 0.0]);
    }
    #[test]
    fn keep_top_one_even_small_p() {
        let p = logits_to_probs(
            &[0.0, 0.0],
            &[],
            SamplingParams {
                top_p: 0.01,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(p, [1.0, 0.0]);
    }
    #[test]
    fn negative_penalty_multiplies_positive_divides() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            repetition_penalty: 1.5,
        };
        let p = logits_to_probs(&[2.0, -1.0], &[0, 1], params).unwrap();
        let expected = (2.0_f32 / 1.5 - (-1.5)).exp();
        assert!((p[0] - expected / (1.0 + expected)).abs() < 1e-6);
    }
    #[test]
    fn penalties_applied_before_top_p() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.6,
            repetition_penalty: 1.5,
        };
        let p = logits_to_probs(&[3.0, 2.5], &[0], params).unwrap();
        assert_eq!(p, [0.0, 1.0]);
    }
    #[test]
    fn duplicate_history_token_is_penalized_once() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            repetition_penalty: 1.5,
        };
        let once = logits_to_probs(&[2.0, 1.0], &[0], params).unwrap();
        let repeated = logits_to_probs(&[2.0, 1.0], &[0, 0, 0], params).unwrap();
        assert_eq!(once, repeated);
    }
    #[test]
    fn reject_invalid_inputs() {
        assert!(logits_to_probs(&[], &[], SamplingParams::default()).is_err());
        assert!(logits_to_probs(&[1.0], &[10], SamplingParams::default()).is_err());
        assert!(logits_to_probs(&[f32::NAN], &[], SamplingParams::default()).is_err());
    }
    #[test]
    fn deterministic_seed() {
        let a: Vec<usize> = {
            let mut r = SessionRng::new(123);
            (0..20)
                .map(|_| r.sample_exponential_race(&[0.2, 0.8]).unwrap())
                .collect()
        };
        let b: Vec<usize> = {
            let mut r = SessionRng::new(123);
            (0..20)
                .map(|_| r.sample_exponential_race(&[0.2, 0.8]).unwrap())
                .collect()
        };
        assert_eq!(a, b);
    }
    #[test]
    fn exponential_race_distribution() {
        let mut r = SessionRng::new(42);
        let hits = (0..20_000)
            .filter(|_| r.sample_exponential_race(&[0.2, 0.8]).unwrap() == 1)
            .count();
        assert!((hits as f32 / 20_000.0 - 0.8).abs() < 0.02);
    }
    #[test]
    fn sum_to_one() {
        let p = logits_to_probs(
            &[0.1, -2.0, 1.1, 3.2, 0.7],
            &[1, 2, 0],
            SamplingParams {
                top_p: 0.99,
                ..Default::default()
            },
        )
        .unwrap();
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
