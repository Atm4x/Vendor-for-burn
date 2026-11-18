//! Silero V5 ConvNeXt vocoder as native Burn tensor operations.

use anyhow::Result;
use burn::tensor::{Tensor, activation::gelu, module::{conv1d, layer_norm, linear}, ops::ConvOptions};
use crate::store::TensorStore;
use burn::tensor::Device;

struct ConvNeXtBlock {
    dwconv_weight: Tensor<3>,
    dwconv_bias: Tensor<1>,
    norm_weight: Tensor<1>,
    norm_bias: Tensor<1>,
    pw1_weight: Tensor<2>,
    pw1_bias: Tensor<1>,
    pw2_weight: Tensor<2>,
    pw2_bias: Tensor<1>,
    gamma: Tensor<1>,
}

impl ConvNeXtBlock {
    fn load(weights: &TensorStore, id: usize, device: &Device) -> Result<Self> {
        let p = format!("vocoder.backbone.convnext.{id}");
        Ok(Self {
            dwconv_weight: weights.float(&format!("{p}.dwconv.weight"), device)?,
            dwconv_bias: weights.float(&format!("{p}.dwconv.bias"), device)?,
            norm_weight: weights.float(&format!("{p}.norm.weight"), device)?,
            norm_bias: weights.float(&format!("{p}.norm.bias"), device)?,
            pw1_weight: weights.float::<2>(&format!("{p}.pwconv1.weight"), device)?.transpose(),
            pw1_bias: weights.float(&format!("{p}.pwconv1.bias"), device)?,
            pw2_weight: weights.float::<2>(&format!("{p}.pwconv2.weight"), device)?.transpose(),
            pw2_bias: weights.float(&format!("{p}.pwconv2.bias"), device)?,
            gamma: weights.float(&format!("{p}.gamma"), device)?,
        })
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let channels = self.dwconv_weight.dims()[0];
        let x = conv1d(input.clone(), self.dwconv_weight.clone(), Some(self.dwconv_bias.clone()),
            ConvOptions::new([1], [3], [1], channels));
        let x = layer_norm(x.swap_dims(1,2), self.norm_weight.clone(), Some(self.norm_bias.clone()), 1e-6);
        let x = linear(x, self.pw1_weight.clone(), Some(self.pw1_bias.clone()));
        let x = gelu(x);
        let x = linear(x, self.pw2_weight.clone(), Some(self.pw2_bias.clone()));
        (x * self.gamma.clone().reshape([1,1,channels])).swap_dims(1,2) + input
    }
}

pub struct Vocoder {
    embed_weight: Tensor<3>,
    embed_bias: Tensor<1>,
    norm_weight: Tensor<1>,
    norm_bias: Tensor<1>,
    blocks: Vec<ConvNeXtBlock>,
    final_norm_weight: Tensor<1>,
    final_norm_bias: Tensor<1>,
    head_weight: Tensor<2>,
    head_bias: Tensor<1>,
}

impl Vocoder {
    pub fn load(weights: &TensorStore, device: &Device) -> Result<Self> {
        let p = "vocoder.backbone";
        let blocks = (0..8).map(|id| ConvNeXtBlock::load(weights, id, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_weight: weights.float(&format!("{p}.embed.weight"), device)?,
            embed_bias: weights.float(&format!("{p}.embed.bias"), device)?,
            norm_weight: weights.float(&format!("{p}.norm.weight"), device)?,
            norm_bias: weights.float(&format!("{p}.norm.bias"), device)?,
            blocks,
            final_norm_weight: weights.float(&format!("{p}.final_layer_norm.weight"), device)?,
            final_norm_bias: weights.float(&format!("{p}.final_layer_norm.bias"), device)?,
            head_weight: weights.float::<2>("vocoder.head.out.weight", device)?.transpose(),
            head_bias: weights.float("vocoder.head.out.bias", device)?,
        })
    }

    /// Returns `(magnitude, cos(phase), sin(phase))` in `[B, 1201, T]`.
    /// This is the complete JitV4 backbone and head, before Silero's ISTFT/PQMF.
    pub fn forward(&self, mel: Tensor<3>) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let x = conv1d(mel, self.embed_weight.clone(), Some(self.embed_bias.clone()),
            ConvOptions::new([1], [3], [1], 1));
        let mut x = layer_norm(x.swap_dims(1,2), self.norm_weight.clone(), Some(self.norm_bias.clone()), 1e-6)
            .swap_dims(1,2);
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = layer_norm(x.swap_dims(1,2), self.final_norm_weight.clone(),
            Some(self.final_norm_bias.clone()), 1e-6);
        let spectrum = linear(x, self.head_weight.clone(), Some(self.head_bias.clone()))
            .swap_dims(1,2);
        let [batch, channels, frames] = spectrum.dims();
        let bins = channels / 2;
        let mag = spectrum.clone().slice([0..batch, 0..bins, 0..frames]).exp().clamp_max(100.0);
        let phase = spectrum.slice([0..batch, bins..channels, 0..frames]);
        (mag, phase.clone().cos(), phase.sin())
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn actual_silero_v5_spectral_head_parity() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else { return };
        let device = Device::flex();
        let weights = TensorStore::open(&Path::new(&dir).join("tts.safetensors")).unwrap();
        let reference = TensorStore::open(&Path::new(&dir).join("parity.safetensors")).unwrap();
        let model = Vocoder::load(&weights, &device).unwrap();
        let (mag, c, s) = model.forward(reference.float("mel", &device).unwrap());
        for (name, prediction) in [("mag", mag), ("cos", c), ("sin", s)] {
            let expected = reference.float::<3>(name, &device).unwrap()
                .into_data().try_to_vec::<f32>().unwrap();
            let got = prediction.into_data().try_to_vec::<f32>().unwrap();
            let max_abs = got.iter().zip(expected.iter())
                .map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max);
            println!("{name}: max_abs={max_abs:.9}");
            assert!(max_abs < 1.0e-3, "{name} max_abs={max_abs:.9}");
        }
    }
}
