use crate::checkpoint::Checkpoint;

use super::decode::{GroupConfig, GroupPredictions, InstrumentGroup, decode_groups};
use super::instruments::INSTRUMENTS;

struct Dense {
    width_in: usize,
    width_out: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
}

impl Dense {
    fn load(checkpoint: &Checkpoint, prefix: &str) -> Result<Self, String> {
        let (shape, weight) = checkpoint.values(&format!("{prefix}.weight"))?;
        let (bias_shape, bias) = checkpoint.values(&format!("{prefix}.bias"))?;
        if shape.len() != 2 || bias_shape != [shape[0]] {
            return Err(format!("invalid Grouper track-head layer {prefix}"));
        }
        Ok(Self {
            width_in: shape[1], width_out: shape[0],
            weights: weight.into_iter().map(f64::from).collect(),
            bias: bias.into_iter().map(f64::from).collect(),
        })
    }

    fn forward(&self, input: &[f64]) -> Result<Vec<f64>, String> {
        if input.len() != self.width_in { return Err("Grouper track-head input width mismatch".into()); }
        Ok((0..self.width_out).map(|j| self.bias[j] + input.iter().enumerate()
            .map(|(i, &x)| self.weights[j * self.width_in + i] * x).sum::<f64>()).collect())
    }
}

fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t
        - 0.284_496_736) * t + 0.254_829_592) * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

fn gelu(x: f64) -> f64 { 0.5 * x * (1.0 + erf(x / 2.0f64.sqrt())) }

pub fn track_stats(notes: &[(f64, f64, u8)], indices: &[usize]) -> Result<[f64; 6], String> {
    if indices.is_empty() || indices.iter().any(|&i| i >= notes.len()) {
        return Err("track requires valid note indices".into());
    }
    let values = indices.iter().map(|&i| f64::from(notes[i].2) / 64.0 - 1.0).collect::<Vec<_>>();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut polyphonic = 0.0;
    if indices.len() <= 3000 {
        for &a in indices {
            let (on, off, _) = notes[a];
            let count = indices.iter().filter(|&&b| {
                notes[b].1.max(notes[b].0 + 0.02).min(off.max(on + 0.02))
                    > notes[b].0.max(on) + 0.03
            }).count();
            if count > 1 { polyphonic += 1.0; }
        }
        polyphonic /= n;
    }
    Ok([mean, std, min, max, n.ln_1p() / 5.0, polyphonic])
}

pub struct TrackHead { layers: [Dense; 3] }

impl TrackHead {
    pub fn load(checkpoint: &Checkpoint) -> Result<Self, String> {
        let layers = [
            Dense::load(checkpoint, "heads.th.0")?,
            Dense::load(checkpoint, "heads.th.2")?,
            Dense::load(checkpoint, "heads.th.4")?,
        ];
        if layers[0].width_out != layers[1].width_in
            || layers[1].width_out != layers[2].width_in
            || layers[2].width_out != INSTRUMENTS.len()
        { return Err("Grouper track-head linear dimensions disagree".into()); }
        Ok(Self { layers })
    }

    pub fn predict(&self, hidden: &[Vec<f32>], notes: &[(f64, f64, u8)], indices: &[usize])
        -> Result<Vec<f64>, String>
    {
        if indices.is_empty() || indices.iter().any(|&i| i >= hidden.len()) {
            return Err("invalid Grouper track-head note references".into());
        }
        let d = hidden[indices[0]].len();
        if self.layers[0].width_in != d + 6 || indices.iter().any(|&i| hidden[i].len() != d) {
            return Err("Grouper track-head state width mismatch".into());
        }
        let mut mean = vec![0.0f64; d];
        for &i in indices {
            for (sum, &value) in mean.iter_mut().zip(&hidden[i]) { *sum += f64::from(value); }
        }
        for x in &mut mean { *x /= indices.len() as f64; }
        mean.extend_from_slice(&track_stats(notes, indices)?);
        for (i, layer) in self.layers.iter().enumerate() {
            mean = layer.forward(&mean)?;
            if i != self.layers.len() - 1 { for value in &mut mean { *value = gelu(*value); } }
        }
        Ok(mean)
    }

    pub fn decode(
        &self, predictions: &GroupPredictions, notes: &[(f64, f64, u8)], config: GroupConfig,
    ) -> Result<Vec<InstrumentGroup>, String> {
        let hidden = predictions.hidden.as_ref().ok_or("Grouper track-head requires hidden states")?;
        decode_groups(predictions, notes, config, Some(&|ids| self.predict(hidden, notes, ids)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn monophonic_register_statistics_match_upstream() {
        let notes = [(0.0, 0.2, 60), (0.4, 0.6, 64), (0.8, 1.0, 67)];
        let stats = track_stats(&notes, &[0, 1, 2]).unwrap();
        assert!((stats[0] - (191.0 / 3.0 / 64.0 - 1.0)).abs() < 1e-7);
        assert_eq!(stats[5], 0.0);
    }
}
