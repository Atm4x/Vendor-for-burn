use crate::{F5Error, Result};

pub struct StandardNormal {
    state: u64,
    spare: Option<f32>,
}

impl StandardNormal {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0x9e37_79b9_7f4a_7c15 {
                0x2545_f491_4f6c_dd1d
            } else {
                seed ^ 0x9e37_79b9_7f4a_7c15
            },
            spare: None,
        }
    }

    fn uniform(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let value = self.state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ((value >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    pub fn sample(&mut self) -> f32 {
        if let Some(sample) = self.spare.take() {
            return sample;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = 2.0 * core::f64::consts::PI * self.uniform();
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }

    pub fn sample_vec(&mut self, count: usize) -> Result<Vec<f32>> {
        if count == 0 || count > 65_536 * 100 {
            return Err(F5Error::InvalidShape("Invalid Gaussian noise length"));
        }
        Ok((0..count).map(|_| self.sample()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_stream_is_deterministic_across_chunks() {
        let mut a = StandardNormal::new(41);
        let mut b = StandardNormal::new(41);
        let x = a.sample_vec(7).unwrap();
        let y = a.sample_vec(5).unwrap();
        assert_eq!([x, y].concat(), b.sample_vec(12).unwrap());
        assert!(a.sample_vec(0).is_err());
    }
}
