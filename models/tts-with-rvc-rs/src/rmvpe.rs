use burn::tensor::{activation, Device, Tensor};

use crate::checkpoint::RvcCheckpoint;

pub const F0_BINS: usize = 360;
const CENTS_START: f64 = 1997.3794084376191;
const CENTS_STEP: f64 = 20.0;

/// Final RMVPE inference head after the bidirectional GRU.
pub struct RmvpeHead {
    weight: Tensor<2>,
    bias: Tensor<1>,
}

impl RmvpeHead {
    pub fn load(checkpoint: &RvcCheckpoint, device: &Device) -> Result<Self, String> {
        let weight = checkpoint.tensor::<2>("fc.1.weight", device)?;
        let bias = checkpoint.tensor::<1>("fc.1.bias", device)?;
        if weight.dims() != [F0_BINS, 512] || bias.dims() != [F0_BINS] {
            return Err("RMVPE output projection must have shape (360, 512) and bias (360)".into());
        }
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, gru_output: Tensor<3>) -> Tensor<3> {
        let [batch, time, channels] = gru_output.dims();
        assert_eq!(channels, 512, "RMVPE bidirectional GRU must emit 512 features");
        let projected = gru_output
            .reshape([batch * time, 512])
            .matmul(self.weight.clone().transpose())
            + self.bias.clone().reshape([1, F0_BINS]);
        activation::sigmoid(projected).reshape([batch, time, F0_BINS])
    }
}

/// Reproduces RMVPE's 9-bin weighted local-centroid F0 decoder, per frame.
/// `salience` is the 360-class sigmoid output from RMVPE.
pub fn salience_to_f0(salience: &[f32], threshold: f32) -> Result<Vec<f32>, String> {
    if salience.len() % F0_BINS != 0 || !threshold.is_finite() || threshold < 0.0 {
        return Err("RMVPE requires complete 360-bin frames and a finite nonnegative threshold".into());
    }
    let mut output = Vec::with_capacity(salience.len() / F0_BINS);
    for frame in salience.chunks_exact(F0_BINS) {
        if frame.iter().any(|value| !value.is_finite()) {
            return Err("RMVPE salience contains a non-finite value".into());
        }
        let mut center = 0;
        for index in 1..F0_BINS {
            if frame[index] > frame[center] {
                center = index;
            }
        }
        if frame[center] <= threshold {
            output.push(0.0);
            continue;
        }
        let start = center.saturating_sub(4);
        let end = (center + 4).min(F0_BINS - 1);
        let mut weighted = 0.0_f64;
        let mut total = 0.0_f64;
        for index in start..=end {
            let value = frame[index] as f64;
            weighted += value * (CENTS_START + CENTS_STEP * index as f64);
            total += value;
        }
        let cents = if total > 0.0 { weighted / total } else { 0.0 };
        output.push(if cents == 0.0 { 0.0 } else { (10.0 * 2.0_f64.powf(cents / 1200.0)) as f32 });
    }
    Ok(output)
}
