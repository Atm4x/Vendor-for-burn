use anyhow::{Result, bail};
use burn::tensor::{Device, Tensor, TensorData, Int, ops::{ConvOptions, AvgPoolOptions, PadMode}, module::{embedding, conv1d, linear, layer_norm, avg_pool1d}};

use crate::{store::TensorStore, transformer::{TransformerBlock, TransformerStack}, predictors::ScalarPredictor};

pub struct LengthRegulator {
    device: Device,
}

impl LengthRegulator {
    pub fn new(device: &Device) -> Self { Self { device: device.clone() } }

    /// PyTorch repeat_interleave with durations truncated after adding 0.5,
    /// right-padding unequal batch sequences with zero vectors.
    pub fn expand(&self, hidden: Tensor<3>, durations: Tensor<2>) -> Result<Tensor<3>> {
        let [batch, len, channels] = hidden.dims();
        if durations.dims() != [batch, len] { bail!("Invalid durations shape"); }
        let values = durations.into_data().try_to_vec::<f32>()?;
        let mut batch_indices = Vec::with_capacity(batch);
        let mut max_len = 0usize;
        for b in 0..batch {
            let mut indices = Vec::new();
            for t in 0..len {
                let duration = (values[b * len + t].max(0.0) + 0.5) as usize;
                if duration > 100_000 { bail!("Excessive token duration"); }
                indices.extend(std::iter::repeat_n(t as i64, duration));
            }
            max_len = max_len.max(indices.len());
            batch_indices.push(indices);
        }
        if max_len == 0 || max_len > 1_000_000 { bail!("Invalid expanded sequence length"); }
        let mut all_indices = Vec::with_capacity(batch * max_len);
        let mut all_mask = Vec::with_capacity(batch * max_len);
        for indices in &batch_indices {
            all_indices.extend(indices.iter().copied());
            all_mask.extend(std::iter::repeat_n(1.0f32, indices.len()));
            all_indices.extend(std::iter::repeat_n(0i64, max_len - indices.len()));
            all_mask.extend(std::iter::repeat_n(0.0f32, max_len - indices.len()));
        }
        let gather = Tensor::<3, Int>::from_data(
            TensorData::new(all_indices, [batch, max_len, 1]), &self.device).repeat_dim(2, channels);
        let valid = Tensor::<3>::from_data(
            TensorData::new(all_mask, [batch, max_len, 1]), &self.device);
        Ok(hidden.gather(1, gather) * valid)
    }
}

pub struct FastPitchDecoder {
    pos_encoding: Tensor<3>,
    pos_scale: Tensor<1>,
    pre: TransformerBlock,
    shorten: Vec<TransformerBlock>,
    up_proj: Tensor<2>,
    up_bias: Tensor<1>,
    post: TransformerBlock,
    norm_weight: Tensor<1>,
    norm_bias: Tensor<1>,
}

impl FastPitchDecoder {
    fn load(weights: &TensorStore, device: &Device) -> Result<Self> {
        let p = "tacotron.decoder";
        Ok(Self {
            pos_encoding: weights.float(&format!("{p}.pos_encoder.pe"), device)?,
            pos_scale: weights.float(&format!("{p}.pos_encoder.scale"), device)?,
            pre: TransformerBlock::load(weights, &format!("{p}.pre_vanilla_layers.0"), device)?,
            shorten: (0..2).map(|i| TransformerBlock::load(weights, &format!("{p}.shorten_layers.{i}"), device)).collect::<Result<_>>()?,
            up_proj: weights.float::<2>(&format!("{p}.upsample.proj.weight"), device)?.transpose(),
            up_bias: weights.float(&format!("{p}.upsample.proj.bias"), device)?,
            post: TransformerBlock::load(weights, &format!("{p}.post_vanilla_layers.0"), device)?,
            norm_weight: weights.float(&format!("{p}.norm.weight"), device)?,
            norm_bias: weights.float(&format!("{p}.norm.bias"), device)?,
        })
    }

    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch, len, channels] = input.dims();
        let pos = self.pos_encoding.clone().slice([0..len, 0..1, 0..channels]).swap_dims(0, 1);
        let mut x = input + pos * self.pos_scale.clone().reshape([1, 1, 1]);
        x = self.pre.forward(x, None);
        let pad = (3 - (len % 3)) % 3;
        if pad != 0 {
            x = x.pad([(0, pad), (0, 0)], PadMode::Constant(0.0));
        }
        let padded = len + pad;
        let residual = x.clone();
        x = avg_pool1d(x.swap_dims(1, 2), AvgPoolOptions::new([3])).swap_dims(1, 2);
        x = self.shorten[1].forward(x, None);
        x = linear(x, self.up_proj.clone(), Some(self.up_bias.clone()))
            .reshape([batch, padded, channels]);
        x = (x + residual).slice([0..batch, 0..len, 0..channels]);
        x = self.post.forward(x, None);
        layer_norm(x, self.norm_weight.clone(), Some(self.norm_bias.clone()), 1e-5)
    }
}

pub struct AcousticModel {
    embedding: Tensor<2>,
    speaker_embedding: Tensor<2>,
    encoder: TransformerStack,
    pitch_proj_weight: Tensor<3>,
    pitch_proj_bias: Tensor<1>,
    energy_proj_weight: Tensor<3>,
    energy_proj_bias: Tensor<1>,
    regulator: LengthRegulator,
    decoder: FastPitchDecoder,
    mel_weight: Tensor<2>,
    mel_bias: Tensor<1>,
}

impl AcousticModel {
    pub fn load(weights: &TensorStore, device: &Device) -> Result<Self> {
        Ok(Self {
            embedding: weights.float("tacotron.embedding.weight", device)?,
            speaker_embedding: weights.float("tacotron.speaker_embedding.weight", device)?,
            encoder: TransformerStack::load(weights, "tacotron.encoder", 4, device)?,
            pitch_proj_weight: weights.float("tacotron.pitch_proj.weight", device)?,
            pitch_proj_bias: weights.float("tacotron.pitch_proj.bias", device)?,
            energy_proj_weight: weights.float("tacotron.energy_proj.weight", device)?,
            energy_proj_bias: weights.float("tacotron.energy_proj.bias", device)?,
            regulator: LengthRegulator::new(device),
            decoder: FastPitchDecoder::load(weights, device)?,
            mel_weight: weights.float::<2>("tacotron.lin.weight", device)?.transpose(),
            mel_bias: weights.float("tacotron.lin.bias", device)?,
        })
    }

    pub fn encode(&self, symbols: Tensor<2, Int>, mask: Option<Tensor<2>>) -> Tensor<3> {
        self.encoder.forward(embedding(self.embedding.clone(), symbols), mask)
    }

    /// `durations` uses `[B, L]`; pitch and optional energy use `[B, 1, L]`.
    /// Output mel is `[B, 192, T]`, exactly as the original Silero Tacotron.
    pub fn forward(&self, symbols: Tensor<2, Int>, speakers: Tensor<2, Int>,
        mask: Option<Tensor<2>>, durations: Tensor<2>, pitch: Tensor<3>, energy: Option<Tensor<3>>) -> Result<Tensor<3>> {
        let [batch, len] = symbols.dims();
        if pitch.dims() != [batch, 1, len] { bail!("Expected pitch [batch, 1, symbols]"); }
        let speaker = embedding(self.speaker_embedding.clone(), speakers).reshape([batch, 1, 128]);
        let mut encoded = self.encode(symbols, mask) + speaker.repeat_dim(1, len);
        let pitch_projection = conv1d(pitch, self.pitch_proj_weight.clone(), Some(self.pitch_proj_bias.clone()),
            ConvOptions::new([1], [1], [1], 1)).swap_dims(1, 2);
        encoded = encoded + pitch_projection;
        if let Some(energy) = energy {
            if energy.dims() != [batch, 1, len] { bail!("Expected energy [batch, 1, symbols]"); }
            encoded = encoded + conv1d(energy, self.energy_proj_weight.clone(), Some(self.energy_proj_bias.clone()),
                ConvOptions::new([1], [1], [1], 1)).swap_dims(1, 2);
        }
        let expanded = self.regulator.expand(encoded, durations)?;
        let decoded = self.decoder.forward(expanded);
        Ok(linear(decoded, self.mel_weight.clone(), Some(self.mel_bias.clone())).swap_dims(1, 2))
    }
}

pub struct TextToMel {
    pub duration: ScalarPredictor,
    pub pitch: ScalarPredictor,
    pub acoustic: AcousticModel,
    device: Device,
}

impl TextToMel {
    pub fn load(weights: &TensorStore, device: &Device) -> Result<Self> {
        Ok(Self {
            duration: ScalarPredictor::duration(weights, device)?,
            pitch: ScalarPredictor::pitch(weights, device)?,
            acoustic: AcousticModel::load(weights, device)?,
            device: device.clone(),
        })
    }

    /// Implements the default one-speaker acoustic inference without optional
    /// SSML, pause overrides, rate modifications, or explicit pitch coefficients.
    pub fn infer(&self, symbols: &[i64], speaker: i64) -> Result<(Tensor<3>, Vec<f32>)> {
        if symbols.is_empty() { bail!("Cannot infer empty text"); }
        let len = symbols.len();
        let input = Tensor::<2, Int>::from_data(TensorData::new(symbols.to_vec(), [1, len]), &self.device);
        let speaker = Tensor::<2, Int>::from_data(TensorData::new(vec![speaker], [1, 1]), &self.device);
        let log_duration = self.duration.forward(input.clone(), speaker.clone(), None, 1.0);
        let mut duration: Vec<f32> = (log_duration.exp() - 1.0).clamp_min(0.0).round().into_data().try_to_vec()?;
        duration[0] = duration[0].min(5.0);
        if len >= 2 {
            duration[len - 1] = duration[len - 1].min(7.0);
            duration[len - 2] = 13.0;
        }
        if len >= 3 { duration[len - 3] = duration[len - 3].min(13.0); }
        let pitch = self.pitch.forward(input.clone(), speaker.clone(), None, 1.0).reshape([1, 1, len]);
        let durations = Tensor::<2>::from_data(TensorData::new(duration.clone(), [1, len]), &self.device);
        Ok((self.acoustic.forward(input, speaker, None, durations, pitch, None)?, duration))
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::{fs, path::Path};

    #[derive(Deserialize)]
    struct Inputs { tokens: Vec<i64>, speaker: Vec<i64>, lengths: Vec<f32> }

    fn max_abs(got: Vec<f32>, expected: Vec<f32>, label: &str) -> f32 {
        assert_eq!(got.len(), expected.len(), "mismatched {label} tensor length");
        let diff = got.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("{label}: max_absolute_difference={diff:.9}");
        diff
    }

    #[test]
    fn reference_encoder_and_acoustic_mel() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = Path::new(&dir);
        let device = Device::flex();
        let weights = TensorStore::open(&dir.join("tts.safetensors")).unwrap();
        let parity = TensorStore::open(&dir.join("acoustic-parity.safetensors")).unwrap();
        let input: Inputs = serde_json::from_str(&fs::read_to_string(dir.join("acoustic-input.json")).unwrap()).unwrap();
        let len = input.tokens.len();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(input.tokens, [1, len]), &device);
        let speaker = Tensor::<2, Int>::from_data(TensorData::new(input.speaker, [1, 1]), &device);
        let durations = Tensor::<2>::from_data(TensorData::new(input.lengths, [1, len]), &device);
        let model = AcousticModel::load(&weights, &device).unwrap();
        let encoded = model.encode(tokens.clone(), None).into_data().try_to_vec::<f32>().unwrap();
        assert!(max_abs(encoded, parity.float_data("acoustic_encoded").unwrap(), "encoder") < 1e-3);
        let pitch = parity.float("pitch", &device).unwrap();
        let speaker_emb = embedding(model.speaker_embedding.clone(), speaker.clone()).reshape([1, 1, 128]).repeat_dim(1, len);
        let pitch_emb = conv1d(pitch.clone(), model.pitch_proj_weight.clone(), Some(model.pitch_proj_bias.clone()), ConvOptions::new([1], [1], [1], 1)).swap_dims(1, 2);
        let mixed = model.encode(tokens.clone(), None) + speaker_emb + pitch_emb;
        max_abs(mixed.clone().into_data().try_to_vec().unwrap(), parity.float_data("mixed").unwrap(), "mixed");
        let expanded = model.regulator.expand(mixed, durations.clone()).unwrap();
        max_abs(expanded.clone().into_data().try_to_vec().unwrap(), parity.float_data("expanded").unwrap(), "expanded");
        let d = &model.decoder;
        let pos = d.pos_encoding.clone().slice([0..21, 0..1, 0..128]).swap_dims(0,1);
        let positioned = expanded + pos * d.pos_scale.clone().reshape([1, 1, 1]);
        max_abs(positioned.clone().into_data().try_to_vec().unwrap(), parity.float_data("decoder_pos").unwrap(), "decoder_pos");
        let pre = d.pre.forward(positioned, None);
        max_abs(pre.clone().into_data().try_to_vec().unwrap(), parity.float_data("decoder_pre").unwrap(), "decoder_pre");
        let down = avg_pool1d(pre.clone().swap_dims(1, 2), AvgPoolOptions::new([3])).swap_dims(1, 2);
        max_abs(down.clone().into_data().try_to_vec().unwrap(), parity.float_data("decoder_down").unwrap(), "decoder_down");
        let short0 = d.shorten[0].forward(down.clone(), None);
        max_abs(short0.clone().into_data().try_to_vec().unwrap(), parity.float_data("decoder_short0").unwrap(), "decoder_short0");
        let short1 = d.shorten[1].forward(down, None);
        max_abs(short1.clone().into_data().try_to_vec().unwrap(), parity.float_data("decoder_short1").unwrap(), "decoder_short1");
        let up = linear(short1, d.up_proj.clone(), Some(d.up_bias.clone())).reshape([1,21,128]);
        max_abs(up.clone().into_data().try_to_vec().unwrap(), parity.float_data("decoder_up").unwrap(), "decoder_up");
        let post_input = pre + up;
        max_abs(post_input.clone().into_data().try_to_vec().unwrap(), parity.float_data("decoder_post_input").unwrap(), "decoder_post_input");
        let post = d.post.forward(post_input, None);
        max_abs(post.clone().into_data().try_to_vec().unwrap(), parity.float_data("decoder_post").unwrap(), "decoder_post");
        let final_x = layer_norm(post, d.norm_weight.clone(), Some(d.norm_bias.clone()), 1e-5);
        max_abs(final_x.clone().into_data().try_to_vec().unwrap(), parity.float_data("decoder_final").unwrap(), "decoder_final");
        let got = model.forward(tokens, speaker, None, durations, pitch, None).unwrap().into_data().try_to_vec::<f32>().unwrap();
        assert!(max_abs(got, parity.float_data("mel").unwrap(), "mel") < 1e-3);
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod full_inference_tests {
    use super::*;
    use std::{fs, path::Path};

    #[test]
    fn reference_default_token_to_mel() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let dir = Path::new(&dir);
        let device = Device::flex();
        let weights = TensorStore::open(&dir.join("tts.safetensors")).unwrap();
        let parity = TensorStore::open(&dir.join("acoustic-parity.safetensors")).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("acoustic-input.json")).unwrap()).unwrap();
        let ids: Vec<i64> = serde_json::from_value(json["tokens"].clone()).unwrap();
        let model = TextToMel::load(&weights, &device).unwrap();
        let (mel, durations) = model.infer(&ids, 0).unwrap();
        let expected_durations = parity.float_data("full_duration").unwrap();
        assert_eq!(durations, expected_durations);
        let expected_mel = parity.float_data("full_mel").unwrap();
        let got = mel.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(got.len(), expected_mel.len());
        let max_abs = got.iter().zip(expected_mel.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        println!("complete token-to-mel parity max_abs={max_abs:.9}");
        assert!(max_abs < 2e-3);
    }
}
