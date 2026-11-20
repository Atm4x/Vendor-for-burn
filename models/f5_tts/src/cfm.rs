use crate::{F5Error, Result};

const EPSS_5: &[u8] = &[0, 2, 4, 8, 16, 32];
const EPSS_6: &[u8] = &[0, 2, 4, 6, 8, 16, 32];
const EPSS_7: &[u8] = &[0, 2, 4, 6, 8, 16, 24, 32];
const EPSS_10: &[u8] = &[0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32];
const EPSS_12: &[u8] = &[0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32];
const EPSS_16: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub steps: usize,
    pub cfg_strength: f32,
    pub sway_sampling_coef: Option<f32>,
    pub empirically_pruned_steps: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            steps: 32,
            cfg_strength: 2.0,
            sway_sampling_coef: Some(-1.0),
            empirically_pruned_steps: true,
        }
    }
}

impl SamplingConfig {
    pub fn validate(&self) -> Result<()> {
        if !(1..=256).contains(&self.steps) {
            return Err(F5Error::InvalidConfiguration(
                "NFE steps must be in 1..=256",
            ));
        }
        if !self.cfg_strength.is_finite() || self.cfg_strength < 0.0 {
            return Err(F5Error::InvalidConfiguration(
                "CFG strength must be finite and nonnegative",
            ));
        }
        if self.sway_sampling_coef.is_some_and(|v| !v.is_finite()) {
            return Err(F5Error::InvalidConfiguration(
                "Sway coefficient must be finite",
            ));
        }
        Ok(())
    }

    pub fn timeline(&self) -> Result<Vec<f32>> {
        self.validate()?;
        let epss = if self.empirically_pruned_steps {
            match self.steps {
                5 => EPSS_5,
                6 => EPSS_6,
                7 => EPSS_7,
                10 => EPSS_10,
                12 => EPSS_12,
                16 => EPSS_16,
                _ => &[],
            }
        } else {
            &[]
        };
        let mut timeline: Vec<f32> = if epss.is_empty() {
            (0..=self.steps)
                .map(|index| index as f32 / self.steps as f32)
                .collect()
        } else {
            epss.iter().map(|&step| f32::from(step) / 32.0).collect()
        };
        if let Some(coefficient) = self.sway_sampling_coef {
            for time in &mut timeline {
                let t = *time;
                *time = t + coefficient * ((core::f32::consts::FRAC_PI_2 * t).cos() - 1.0 + t);
            }
        }
        if timeline.windows(2).any(|window| window[1] <= window[0]) {
            return Err(F5Error::InvalidConfiguration(
                "Sampling timeline must be strictly increasing",
            ));
        }
        Ok(timeline)
    }
}

pub fn classifier_free_guidance(conditional: f32, unconditional: f32, strength: f32) -> f32 {
    conditional + (conditional - unconditional) * strength
}

pub fn euler_step(state: &mut [f32], velocity: &[f32], delta: f32) -> Result<()> {
    if state.len() != velocity.len() {
        return Err(F5Error::InvalidShape(
            "State and velocity have different tensor sizes",
        ));
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(F5Error::InvalidConfiguration(
            "Euler delta must be positive and finite",
        ));
    }
    for (value, velocity) in state.iter_mut().zip(velocity) {
        *value += velocity * delta;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epss_reproduces_original_f5_schedule() {
        let config = SamplingConfig {
            steps: 5,
            sway_sampling_coef: None,
            ..SamplingConfig::default()
        };
        assert_eq!(
            config.timeline().unwrap(),
            vec![0.0, 0.0625, 0.125, 0.25, 0.5, 1.0]
        );
    }

    #[test]
    fn sway_keeps_endpoints_and_increases_time() {
        let timeline = SamplingConfig::default().timeline().unwrap();
        assert_eq!(timeline[0], 0.0);
        assert!((timeline[32] - 1.0).abs() < 1e-6);
        assert!(timeline.windows(2).all(|item| item[0] < item[1]));
    }

    #[test]
    fn euler_and_cfg_match_reference_equation() {
        let mut x = [0.5, -1.0];
        let v = [
            classifier_free_guidance(3.0, 1.0, 2.0),
            classifier_free_guidance(2.0, 4.0, 2.0),
        ];
        euler_step(&mut x, &v, 0.25).unwrap();
        assert_eq!(x, [2.25, -1.5]);
    }

    #[test]
    fn invalid_timeline_is_rejected() {
        let config = SamplingConfig {
            sway_sampling_coef: Some(100.0),
            ..SamplingConfig::default()
        };
        assert!(config.timeline().is_err());
    }
}
