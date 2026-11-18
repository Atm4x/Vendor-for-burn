use anyhow::Result;
use burn::tensor::{Device, Tensor, Int, module::{embedding, linear}};
use crate::{store::TensorStore, transformer::TransformerStack};

pub struct ScalarPredictor {
    symbol_weight: Tensor<2>,
    speaker_weight: Tensor<2>,
    stack: TransformerStack,
    lin_weight: Tensor<2>,
    lin_bias: Tensor<1>,
}

impl ScalarPredictor {
    fn load(weights: &TensorStore, prefix: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            symbol_weight: weights.float(&format!("{prefix}.embedding.weight"), device)?,
            speaker_weight: weights.float(&format!("{prefix}.speaker_embedding.weight"), device)?,
            stack: TransformerStack::load(weights, &format!("{prefix}.transformer"), 4, device)?,
            lin_weight: weights.float::<2>(&format!("{prefix}.lin.weight"), device)?.transpose(),
            lin_bias: weights.float(&format!("{prefix}.lin.bias"), device)?,
        })
    }

    pub fn duration(weights: &TensorStore, device: &Device) -> Result<Self> {
        Self::load(weights, "dur_predictor.dur_pred", device)
    }

    pub fn pitch(weights: &TensorStore, device: &Device) -> Result<Self> {
        Self::load(weights, "pitch_predictor.pitch_pred", device)
    }

    /// Computes the original logarithmic durations or pitch values as `[B, L]`.
    pub fn forward(&self, tokens: Tensor<2, Int>, speaker: Tensor<2, Int>, mask: Option<Tensor<2>>, alpha: f32) -> Tensor<2> {
        let [batch, len] = tokens.dims();
        let channel = self.symbol_weight.dims()[1];
        let symbol = embedding(self.symbol_weight.clone(), tokens);
        let speaker = embedding(self.speaker_weight.clone(), speaker).reshape([batch, 1, channel]);
        let hidden = symbol + speaker.repeat_dim(1, len);
        let hidden = self.stack.forward(hidden, mask);
        linear(hidden, self.lin_weight.clone(), Some(self.lin_bias.clone())).reshape([batch, len]) / alpha
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use serde::Deserialize;
    use std::{fs, path::Path};

    #[derive(Deserialize)]
    struct Inputs {
        tokens: Vec<i64>,
        speaker: Vec<i64>,
    }

    fn check(label: &str, actual: Vec<f32>, expected: Vec<f32>, tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        let max_abs = actual.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("{label} max absolute difference: {max_abs:.9}");
        assert!(max_abs < tolerance, "{label}: error {max_abs} exceeds {tolerance}");
    }

    #[test]
    fn reference_duration_pitch_and_transformer() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = Path::new(&dir);
        let device = Device::flex();
        let weights = TensorStore::open(&dir.join("tts.safetensors")).unwrap();
        let reference = TensorStore::open(&dir.join("acoustic-parity.safetensors")).unwrap();
        let json = fs::read_to_string(dir.join("acoustic-input.json")).unwrap();
        let input: Inputs = serde_json::from_str(&json).unwrap();
        let len = input.tokens.len();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(input.tokens, [1, len]), &device);
        let speaker = Tensor::<2, Int>::from_data(TensorData::new(input.speaker, [1, 1]), &device);
        let dur = ScalarPredictor::duration(&weights, &device).unwrap();
        let pitch = ScalarPredictor::pitch(&weights, &device).unwrap();
        let durations = dur.forward(tokens.clone(), speaker.clone(), None, 1.0).into_data().try_to_vec::<f32>().unwrap();
        let pitches = pitch.forward(tokens, speaker, None, 1.0).into_data().try_to_vec::<f32>().unwrap();
        check("duration", durations, reference.float_data("duration_log").unwrap(), 2e-3);
        check("pitch", pitches, reference.float_data("pitch").unwrap(), 2e-3);
        let block = super::super::transformer::TransformerBlock::load(&weights, "dur_predictor.dur_pred.transformer.layers.0", &device).unwrap();
        let input = reference.float("predictor_pos", &device).unwrap();
        let actual = block.forward(input, None).into_data().try_to_vec::<f32>().unwrap();
        check("predictor_block0", actual, reference.float_data("predictor_block0").unwrap(), 1e-3);
    }
}
