use burn::{
    module::Param,
    nn::{
        Linear, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Device, Tensor, TensorData, activation::gelu},
};

use crate::{F5Error, Result, checkpoint::Checkpoint, dit::load_linear, fft::IStft};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocosConfig {
    pub mel_channels: usize,
    pub dim: usize,
    pub intermediate_dim: usize,
    pub layers: usize,
}

impl VocosConfig {
    pub const MEL_24KHZ: Self = Self {
        mel_channels: 100,
        dim: 512,
        intermediate_dim: 1536,
        layers: 8,
    };

    fn validate(&self) -> Result<()> {
        if self.mel_channels != 100
            || self.dim == 0
            || self.intermediate_dim == 0
            || self.layers == 0
            || self.layers > 32
        {
            return Err(F5Error::InvalidConfiguration("Invalid Vocos architecture"));
        }
        Ok(())
    }
}

fn load_conv(
    ckpt: &Checkpoint,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    groups: usize,
    device: &Device,
) -> Result<Conv1d> {
    let mut module = Conv1dConfig::new(input, output, kernel)
        .with_padding(PaddingConfig1d::Explicit(kernel / 2, kernel / 2))
        .with_groups(groups)
        .init(device);
    let weights = ckpt.read_f32(
        &format!("{prefix}.weight"),
        &[output, input / groups, kernel],
    )?;
    let bias = ckpt.read_f32(&format!("{prefix}.bias"), &[output])?;
    module.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(weights, [output, input / groups, kernel]),
        device,
    ));
    module.bias = Some(Param::from_tensor(Tensor::from_data(
        TensorData::new(bias, [output]),
        device,
    )));
    Ok(module)
}

#[derive(Debug)]
struct Norm {
    gamma: Tensor<3>,
    beta: Tensor<3>,
}

impl Norm {
    fn from_checkpoint(
        ckpt: &Checkpoint,
        prefix: &str,
        dim: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gamma: Tensor::from_data(
                TensorData::new(
                    ckpt.read_f32(&format!("{prefix}.weight"), &[dim])?,
                    [1, 1, dim],
                ),
                device,
            ),
            beta: Tensor::from_data(
                TensorData::new(
                    ckpt.read_f32(&format!("{prefix}.bias"), &[dim])?,
                    [1, 1, dim],
                ),
                device,
            ),
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered / (variance + 1e-6).sqrt() * self.gamma.clone() + self.beta.clone()
    }
}

struct VocosConvNeXt {
    depthwise: Conv1d,
    norm: Norm,
    expand: Linear,
    gamma: Tensor<3>,
    beta: Tensor<3>,
    project: Linear,
}

impl VocosConvNeXt {
    fn from_checkpoint(
        ckpt: &Checkpoint,
        index: usize,
        config: VocosConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("backbone.convnext.{index}");
        let dim = config.dim;
        let hidden = config.intermediate_dim;
        Ok(Self {
            depthwise: load_conv(ckpt, &format!("{prefix}.dwconv"), dim, dim, 7, dim, device)?,
            norm: Norm::from_checkpoint(ckpt, &format!("{prefix}.norm"), dim, device)?,
            expand: load_linear(ckpt, &format!("{prefix}.pwconv1"), dim, hidden, device)?,
            gamma: Tensor::from_data(
                TensorData::new(
                    ckpt.read_f32(&format!("{prefix}.grn.gamma"), &[1, 1, hidden])?,
                    [1, 1, hidden],
                ),
                device,
            ),
            beta: Tensor::from_data(
                TensorData::new(
                    ckpt.read_f32(&format!("{prefix}.grn.beta"), &[1, 1, hidden])?,
                    [1, 1, hidden],
                ),
                device,
            ),
            project: load_linear(ckpt, &format!("{prefix}.pwconv2"), hidden, dim, device)?,
        })
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let x = self
            .depthwise
            .forward(input.clone().swap_dims(1, 2))
            .swap_dims(1, 2);
        let x = gelu(self.expand.forward(self.norm.forward(x)));
        let response = x.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let norm = response.clone() / (response.mean_dim(2) + 1e-6);
        let x = x.clone() + self.gamma.clone() * x * norm + self.beta.clone();
        input + self.project.forward(x)
    }
}

pub struct VocosDecoder {
    config: VocosConfig,
    embed: Conv1d,
    norm: Norm,
    blocks: Vec<VocosConvNeXt>,
    final_norm: Norm,
    head: Linear,
    istft: IStft,
}

impl VocosDecoder {
    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        config: VocosConfig,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        let embed = load_conv(
            checkpoint,
            "backbone.embed",
            config.mel_channels,
            config.dim,
            7,
            1,
            device,
        )?;
        let norm = Norm::from_checkpoint(checkpoint, "backbone.norm", config.dim, device)?;
        let blocks = (0..config.layers)
            .map(|i| VocosConvNeXt::from_checkpoint(checkpoint, i, config, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config,
            embed,
            norm,
            blocks,
            final_norm: Norm::from_checkpoint(
                checkpoint,
                "backbone.final_layer_norm",
                config.dim,
                device,
            )?,
            head: load_linear(checkpoint, "head.out", config.dim, 1026, device)?,
            istft: IStft::new(),
        })
    }

    pub fn spectrogram_parameters(&self, mel: Tensor<3>) -> Result<Tensor<3>> {
        let [batch, frames, channels] = mel.dims();
        if batch != 1 || frames == 0 || channels != self.config.mel_channels {
            return Err(F5Error::InvalidShape("Vocos expects [1, frames, 100] mel"));
        }
        let mut x = self
            .norm
            .forward(self.embed.forward(mel.swap_dims(1, 2)).swap_dims(1, 2));
        for block in &self.blocks {
            x = block.forward(x);
        }
        Ok(self.head.forward(self.final_norm.forward(x)))
    }

    /// Returns a mono 24 kHz waveform; this is a single host transfer after the GPU network.
    pub fn decode(&self, mel: Tensor<3>, restoration_gain: f32) -> Result<Vec<f32>> {
        if !restoration_gain.is_finite() || restoration_gain < 0.0 {
            return Err(F5Error::InvalidConfiguration(
                "Invalid Vocos restoration gain",
            ));
        }
        let head = self.spectrogram_parameters(mel)?;
        let [_, frames, bands] = head.dims();
        if bands != 1026 {
            return Err(F5Error::InvalidShape(
                "Vocos head requires 1026 spectral parameters",
            ));
        }
        let values = head
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| F5Error::ModelExecution(format!("Vocos head download failed: {e}")))?;
        let mut real = vec![0.0; frames * 513];
        let mut imag = vec![0.0; frames * 513];
        for frame in 0..frames {
            for bin in 0..513 {
                let base = frame * 1026;
                let magnitude = values[base + bin].exp();
                let phase = values[base + 513 + bin];
                if !magnitude.is_finite() || !phase.is_finite() {
                    return Err(F5Error::ModelExecution(
                        "Vocos emitted invalid spectrum".to_string(),
                    ));
                }
                real[frame * 513 + bin] = magnitude * phase.cos();
                imag[frame * 513 + bin] = magnitude * phase.sin();
            }
        }
        let mut wave = self.istft.reconstruct(&real, &imag)?;
        for sample in &mut wave {
            *sample *= restoration_gain;
        }
        Ok(wave)
    }
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn vocos_backbone_head_and_waveform_match_pytorch() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/vocos_decoder_small.safetensors");
        let checkpoint = Checkpoint::open(path).unwrap();
        let device = Device::cpu();
        let config = VocosConfig {
            mel_channels: 100,
            dim: 8,
            intermediate_dim: 16,
            layers: 2,
        };
        let model = VocosDecoder::from_checkpoint(&checkpoint, config, &device).unwrap();
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/vocos_decoder_expected.json"
        ))
        .unwrap();
        let values = |name: &str| -> Vec<f32> {
            expected[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };
        let mel = Tensor::from_data(TensorData::new(values("mel"), [1, 9, 100]), &device);
        let head = model
            .spectrogram_parameters(mel.clone())
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let head_expected = values("head");
        let head_max_abs = head
            .iter()
            .zip(head_expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            head_max_abs < 2e-4,
            "Vocos head parity max_abs={head_max_abs}"
        );
        let wave = model.decode(mel, 1.0).unwrap();
        let wave_expected = values("wave");
        assert_eq!(wave.len(), wave_expected.len());
        let wave_max_abs = wave
            .iter()
            .zip(wave_expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            wave_max_abs < 2e-4,
            "Vocos waveform parity max_abs={wave_max_abs}"
        );
    }
}
