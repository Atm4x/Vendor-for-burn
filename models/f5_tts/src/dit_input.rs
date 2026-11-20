use burn::{
    module::Param,
    nn::{
        Embedding, EmbeddingConfig, Linear, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        Device, Int, Tensor, TensorData,
        activation::{gelu, mish},
    },
};

use crate::{F5Error, Result, checkpoint::Checkpoint, dit::load_linear};

fn conv1d(
    checkpoint: &Checkpoint,
    prefix: &str,
    dim: usize,
    groups: usize,
    kernel: usize,
    device: &Device,
) -> Result<Conv1d> {
    if groups == 0 || !dim.is_multiple_of(groups) || kernel == 0 || kernel.is_multiple_of(2) {
        return Err(F5Error::InvalidConfiguration(
            "Invalid F5 Conv1d dimensions",
        ));
    }
    let mut module = Conv1dConfig::new(dim, dim, kernel)
        .with_groups(groups)
        .with_padding(PaddingConfig1d::Explicit(kernel / 2, kernel / 2))
        .init(device);
    let weights = checkpoint.read_f32(&format!("{prefix}.weight"), &[dim, dim / groups, kernel])?;
    let bias = checkpoint.read_f32(&format!("{prefix}.bias"), &[dim])?;
    module.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(weights, [dim, dim / groups, kernel]),
        device,
    ));
    module.bias = Some(Param::from_tensor(Tensor::from_data(
        TensorData::new(bias, [dim]),
        device,
    )));
    Ok(module)
}

pub struct TextConvNeXt {
    depthwise: Conv1d,
    gamma: Tensor<3>,
    beta: Tensor<3>,
    norm_gamma: Tensor<3>,
    norm_beta: Tensor<3>,
    in_linear: Linear,
    out_linear: Linear,
}

impl TextConvNeXt {
    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        index: usize,
        dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("transformer.text_embed.text_blocks.{index}");
        let intermediates = dim * 2;
        let norm_gamma = Tensor::from_data(
            TensorData::new(
                checkpoint.read_f32(&format!("{prefix}.norm.weight"), &[dim])?,
                [1, 1, dim],
            ),
            device,
        );
        let norm_beta = Tensor::from_data(
            TensorData::new(
                checkpoint.read_f32(&format!("{prefix}.norm.bias"), &[dim])?,
                [1, 1, dim],
            ),
            device,
        );
        Ok(Self {
            depthwise: conv1d(checkpoint, &format!("{prefix}.dwconv"), dim, dim, 7, device)?,
            gamma: Tensor::from_data(
                TensorData::new(
                    checkpoint.read_f32(&format!("{prefix}.grn.gamma"), &[1, 1, intermediates])?,
                    [1, 1, intermediates],
                ),
                device,
            ),
            beta: Tensor::from_data(
                TensorData::new(
                    checkpoint.read_f32(&format!("{prefix}.grn.beta"), &[1, 1, intermediates])?,
                    [1, 1, intermediates],
                ),
                device,
            ),
            norm_gamma,
            norm_beta,
            in_linear: load_linear(
                checkpoint,
                &format!("{prefix}.pwconv1"),
                dim,
                intermediates,
                device,
            )?,
            out_linear: load_linear(
                checkpoint,
                &format!("{prefix}.pwconv2"),
                intermediates,
                dim,
                device,
            )?,
        })
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let residual = x.clone();
        let x = self.depthwise.forward(x.swap_dims(1, 2)).swap_dims(1, 2);
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let x = centered.clone() / (centered.powf_scalar(2.0).mean_dim(2) + 1e-6).sqrt();
        let x = x * self.norm_gamma.clone() + self.norm_beta.clone();
        let x = gelu(self.in_linear.forward(x));
        let response = x.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let normalized = response.clone() / (response.mean_dim(2) + 1e-6);
        let x = x.clone() + self.gamma.clone() * (x * normalized) + self.beta.clone();
        residual + self.out_linear.forward(x)
    }
}

pub struct TextEmbedding {
    lookup: Embedding,
    convnext: Vec<TextConvNeXt>,
    dim: usize,
    vocab_size: usize,
}

impl TextEmbedding {
    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        vocab_size: usize,
        dim: usize,
        conv_layers: usize,
        device: &Device,
    ) -> Result<Self> {
        if vocab_size == 0 || dim == 0 || !dim.is_multiple_of(2) {
            return Err(F5Error::InvalidConfiguration(
                "Invalid F5 text embedding dimensions",
            ));
        }
        let mut lookup = EmbeddingConfig::new(vocab_size + 1, dim).init(device);
        let weights = checkpoint.read_f32(
            "transformer.text_embed.text_embed.weight",
            &[vocab_size + 1, dim],
        )?;
        lookup.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(weights, [vocab_size + 1, dim]),
            device,
        ));
        let mut convnext = Vec::with_capacity(conv_layers);
        for i in 0..conv_layers {
            convnext.push(TextConvNeXt::from_checkpoint(checkpoint, i, dim, device)?);
        }
        Ok(Self {
            lookup,
            convnext,
            dim,
            vocab_size,
        })
    }

    pub fn forward(
        &self,
        text: Tensor<2, Int>,
        seq_len: usize,
        drop_text: bool,
    ) -> Result<Tensor<3>> {
        let [batch, tokens] = text.dims();
        if seq_len == 0 || seq_len > 8192 {
            return Err(F5Error::InvalidConfiguration(
                "Invalid F5 text sequence length",
            ));
        }
        let device = text.device();
        let mut text = text.slice([0..batch, 0..tokens.min(seq_len)]) + 1;
        if tokens < seq_len {
            text = Tensor::cat(
                vec![
                    text,
                    Tensor::<2, Int>::zeros([batch, seq_len - tokens], &device),
                ],
                1,
            );
        }
        let pad = text.clone().equal_elem(0);
        if drop_text {
            text = Tensor::<2, Int>::zeros([batch, seq_len], &device);
        }
        let mut output = self.lookup.forward(text);
        let frequencies: Vec<f32> = (0..seq_len)
            .flat_map(|position| {
                (0..self.dim).map(move |channel| {
                    let half = self.dim / 2;
                    let angle = position as f32
                        * (-(10_000.0_f32.ln()) * ((channel % half) * 2) as f32 / self.dim as f32)
                            .exp();
                    if channel < half {
                        angle.cos()
                    } else {
                        angle.sin()
                    }
                })
            })
            .collect();
        output = output
            + Tensor::<3>::from_data(
                TensorData::new(frequencies, [1, seq_len, self.dim]),
                &device,
            );
        let pad = pad.unsqueeze_dim::<3>(2).expand([batch, seq_len, self.dim]);
        output = output.mask_fill(pad.clone(), 0.0);
        for convnext in &self.convnext {
            output = convnext.forward(output).mask_fill(pad.clone(), 0.0);
        }
        Ok(output)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

pub struct InputEmbedding {
    projection: Linear,
    conv1: Conv1d,
    conv2: Conv1d,
    dim: usize,
    mel_channels: usize,
    text_dim: usize,
}

impl InputEmbedding {
    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        dim: usize,
        mel_channels: usize,
        text_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        if dim == 0 || !dim.is_multiple_of(16) || mel_channels == 0 || text_dim == 0 {
            return Err(F5Error::InvalidConfiguration(
                "Invalid input embedding dimensions",
            ));
        }
        Ok(Self {
            projection: load_linear(
                checkpoint,
                "transformer.input_embed.proj",
                2 * mel_channels + text_dim,
                dim,
                device,
            )?,
            conv1: conv1d(
                checkpoint,
                "transformer.input_embed.conv_pos_embed.conv1d.0",
                dim,
                16,
                31,
                device,
            )?,
            conv2: conv1d(
                checkpoint,
                "transformer.input_embed.conv_pos_embed.conv1d.2",
                dim,
                16,
                31,
                device,
            )?,
            dim,
            mel_channels,
            text_dim,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<3>,
        cond: Tensor<3>,
        text: Tensor<3>,
        drop_audio_cond: bool,
    ) -> Result<Tensor<3>> {
        let [batch, seq, channels] = x.dims();
        if channels != self.mel_channels
            || cond.dims() != [batch, seq, channels]
            || text.dims() != [batch, seq, self.text_dim]
        {
            return Err(F5Error::InvalidShape(
                "Input embedding requires aligned mel, conditioning and text frames",
            ));
        }
        let audio_condition = if drop_audio_cond {
            Tensor::zeros([batch, seq, channels], &x.device())
        } else {
            cond
        };
        let projected = self
            .projection
            .forward(Tensor::cat(vec![x, audio_condition, text], 2));
        debug_assert_eq!(projected.dims()[2], self.dim);
        let positional = self.conv1.forward(projected.clone().swap_dims(1, 2));
        let positional = mish(positional);
        let positional = mish(self.conv2.forward(positional)).swap_dims(1, 2);
        Ok(projected + positional)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cubecl-cpu")]
    #[test]
    fn input_embedding_matches_pytorch_convnext_and_position_convolutions() {
        use std::path::Path;
        let device = Device::cpu();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let checkpoint = Checkpoint::open(fixtures.join("dit_input_small.safetensors")).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/dit_input_expected.json"))
                .unwrap();
        let floats = |key: &str| {
            fixture[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let tokens: Vec<i64> = fixture["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(tokens, [1, 3]), &device);
        let text = TextEmbedding::from_checkpoint(&checkpoint, 5, 8, 1, &device)
            .unwrap()
            .forward(tokens, 4, false)
            .unwrap();
        let text_actual = text.clone().into_data().try_to_vec::<f32>().unwrap();
        let text_error = text_actual
            .iter()
            .zip(floats("text"))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            text_error < 2e-4,
            "F5 text ConvNeXt differs from PyTorch max_abs={text_error}"
        );
        let mel = Tensor::<3>::from_data(TensorData::new(floats("mel"), [1, 4, 2]), &device);
        let cond = Tensor::<3>::from_data(TensorData::new(floats("cond"), [1, 4, 2]), &device);
        let network = InputEmbedding::from_checkpoint(&checkpoint, 16, 2, 8, &device).unwrap();
        let output = network
            .forward(mel, cond, text, false)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let input_error = output
            .iter()
            .zip(floats("input"))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            input_error < 2e-4,
            "F5 input grouped convolutions differ from PyTorch max_abs={input_error}"
        );
    }
}
