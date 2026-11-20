use burn::{
    module::Param,
    nn::{
        Linear, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{
        Bool, Device, Tensor, TensorData,
        activation::{gelu, mish, softmax},
    },
};

use crate::{
    F5Error, Result,
    checkpoint::Checkpoint,
    cross_lingual::{SPEED_CLASSES, speed_from_label},
    dit::{DiTAttention, RotaryPositions, load_linear},
    mel::{MEL_CHANNELS, VocosMel},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeedTransformerConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub ff_mult: usize,
    pub position_groups: usize,
}

impl SpeedTransformerConfig {
    pub const CROSS_LINGUAL: Self = Self {
        dim: 512,
        depth: 6,
        heads: 8,
        ff_mult: 4,
        position_groups: 16,
    };

    fn validate(self) -> Result<()> {
        if self.dim == 0
            || self.depth == 0
            || self.depth > 32
            || self.ff_mult == 0
            || self.heads == 0
            || !self.dim.is_multiple_of(self.heads)
            || !(self.dim / self.heads).is_multiple_of(2)
            || self.position_groups == 0
            || !self.dim.is_multiple_of(self.position_groups)
            || self.dim.checked_mul(self.ff_mult).is_none()
        {
            return Err(F5Error::InvalidConfiguration(
                "Invalid Cross-Lingual SpeedTransformer architecture",
            ));
        }
        Ok(())
    }
}

struct Norm {
    gamma: Tensor<3>,
    beta: Tensor<3>,
    epsilon: f32,
}

impl Norm {
    fn from_checkpoint(
        ckpt: &Checkpoint,
        name: &str,
        dim: usize,
        epsilon: f32,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gamma: Tensor::from_data(
                TensorData::new(
                    ckpt.read_f32(&format!("{name}.weight"), &[dim])?,
                    [1, 1, dim],
                ),
                device,
            ),
            beta: Tensor::from_data(
                TensorData::new(ckpt.read_f32(&format!("{name}.bias"), &[dim])?, [1, 1, dim]),
                device,
            ),
            epsilon,
        })
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let mean = input.clone().mean_dim(2);
        let centered = input - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered / (variance + self.epsilon).sqrt() * self.gamma.clone() + self.beta.clone()
    }
}

fn load_conv(
    ckpt: &Checkpoint,
    prefix: &str,
    dim: usize,
    kernel: usize,
    groups: usize,
    device: &Device,
) -> Result<Conv1d> {
    let mut layer = Conv1dConfig::new(dim, dim, kernel)
        .with_padding(PaddingConfig1d::Explicit(kernel / 2, kernel / 2))
        .with_groups(groups)
        .init(device);
    layer.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(
            ckpt.read_f32(&format!("{prefix}.weight"), &[dim, dim / groups, kernel])?,
            [dim, dim / groups, kernel],
        ),
        device,
    ));
    layer.bias = Some(Param::from_tensor(Tensor::from_data(
        TensorData::new(ckpt.read_f32(&format!("{prefix}.bias"), &[dim])?, [dim]),
        device,
    )));
    Ok(layer)
}

struct SpeedLayer {
    attention: DiTAttention,
    norm1: Norm,
    norm2: Norm,
    ff1: Linear,
    ff2: Linear,
}

impl SpeedLayer {
    fn from_checkpoint(
        ckpt: &Checkpoint,
        index: usize,
        config: SpeedTransformerConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("speed_transformer.transformer_blocks.{index}");
        Ok(Self {
            attention: DiTAttention::from_checkpoint(
                ckpt,
                &format!("{prefix}.attn"),
                config.dim,
                config.heads,
                device,
            )?,
            norm1: Norm::from_checkpoint(ckpt, &format!("{prefix}.ln1"), config.dim, 1e-6, device)?,
            norm2: Norm::from_checkpoint(ckpt, &format!("{prefix}.ln2"), config.dim, 1e-6, device)?,
            ff1: load_linear(
                ckpt,
                &format!("{prefix}.ff.ff.0.0"),
                config.dim,
                config.dim * config.ff_mult,
                device,
            )?,
            ff2: load_linear(
                ckpt,
                &format!("{prefix}.ff.ff.2"),
                config.dim * config.ff_mult,
                config.dim,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        x: Tensor<3>,
        rope: &RotaryPositions,
        valid: Tensor<3, Bool>,
    ) -> Result<Tensor<3>> {
        let attention =
            self.attention
                .forward_masked(self.norm1.forward(x.clone()), rope, valid.clone())?;
        let x = x + attention;
        Ok(x.clone()
            + self
                .ff2
                .forward(gelu_tanh(self.ff1.forward(self.norm2.forward(x)))))
    }
}

fn gelu_tanh(x: Tensor<3>) -> Tensor<3> {
    let polynomial = x.clone() + x.clone().powf_scalar(3.0) * 0.044_715;
    x.clone() * ((polynomial * (2.0_f32 / core::f32::consts::PI).sqrt()).tanh() + 1.0) * 0.5
}

pub struct SpeedTransformer {
    config: SpeedTransformerConfig,
    mel_proj: Linear,
    position1: Conv1d,
    position2: Conv1d,
    layers: Vec<SpeedLayer>,
    pool1: Linear,
    pool2: Linear,
    classifier_norm: Norm,
    classifier1: Linear,
    classifier2: Linear,
}

impl SpeedTransformer {
    pub fn from_checkpoint(
        ckpt: &Checkpoint,
        config: SpeedTransformerConfig,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        let dim = config.dim;
        let mut layers = Vec::with_capacity(config.depth);
        for index in 0..config.depth {
            layers.push(SpeedLayer::from_checkpoint(ckpt, index, config, device)?);
        }
        Ok(Self {
            config,
            mel_proj: load_linear(
                ckpt,
                "speed_transformer.mel_proj",
                MEL_CHANNELS,
                dim,
                device,
            )?,
            position1: load_conv(
                ckpt,
                "speed_transformer.conv_layer.conv1d.0",
                dim,
                31,
                config.position_groups,
                device,
            )?,
            position2: load_conv(
                ckpt,
                "speed_transformer.conv_layer.conv1d.2",
                dim,
                31,
                config.position_groups,
                device,
            )?,
            layers,
            pool1: load_linear(ckpt, "speed_transformer.pool.0", dim, dim, device)?,
            pool2: load_linear(ckpt, "speed_transformer.pool.2", dim, 1, device)?,
            classifier_norm: Norm::from_checkpoint(
                ckpt,
                "speed_transformer.classifier.0",
                dim,
                1e-5,
                device,
            )?,
            classifier1: load_linear(ckpt, "speed_transformer.classifier.1", dim, dim, device)?,
            classifier2: load_linear(
                ckpt,
                "speed_transformer.classifier.3",
                dim,
                SPEED_CLASSES,
                device,
            )?,
        })
    }

    /// Consumes [batch, frames, 100] Vocos mel and unpadded frame counts.
    pub fn forward(&self, mel: Tensor<3>, lengths: &[usize]) -> Result<Tensor<2>> {
        let [batch, frames, channels] = mel.dims();
        if batch == 0
            || frames == 0
            || frames > 8192
            || channels != MEL_CHANNELS
            || lengths.len() != batch
            || lengths.iter().any(|&n| n == 0 || n > frames)
        {
            return Err(F5Error::InvalidShape(
                "SpeedTransformer needs valid [B, T, 100] mel and lengths",
            ));
        }
        let device = mel.device();
        let mask: Vec<bool> = lengths
            .iter()
            .flat_map(|&n| (0..frames).map(move |i| i < n))
            .collect();
        let valid = Tensor::<2, Bool>::from_data(TensorData::new(mask, [batch, frames]), &device);
        let mask3 = valid.clone().unsqueeze_dim::<3>(2);
        let mut x = self
            .mel_proj
            .forward(mel)
            .mask_fill(mask3.clone().bool_not(), 0.0);
        let y = mish(self.position1.forward(x.swap_dims(1, 2)))
            .swap_dims(1, 2)
            .mask_fill(mask3.clone().bool_not(), 0.0);
        x = mish(self.position2.forward(y.swap_dims(1, 2)))
            .swap_dims(1, 2)
            .mask_fill(mask3.clone().bool_not(), 0.0);
        let rope = RotaryPositions::new(frames, self.config.dim / self.config.heads, &device)?;
        for layer in &self.layers {
            x = layer.forward(x, &rope, mask3.clone())?;
        }
        let scores = self
            .pool2
            .forward(self.pool1.forward(x.clone()).tanh())
            .mask_fill(mask3.bool_not(), -f32::MAX);
        let pooled = (x * softmax(scores, 1)).sum_dim(1);
        let pooled = self.classifier_norm.forward(pooled);
        let logits = self
            .classifier2
            .forward(gelu(self.classifier1.forward(pooled)));
        Ok(logits.squeeze_dim::<2>(1))
    }

    pub fn predict_speeds(&self, mel: Tensor<3>, lengths: &[usize]) -> Result<Vec<f32>> {
        let logits = self.forward(mel, lengths)?;
        let [batch, classes] = logits.dims();
        if classes != SPEED_CLASSES {
            return Err(F5Error::InvalidShape(
                "SpeedTransformer classifier must have 32 classes",
            ));
        }
        let values = logits.into_data().try_to_vec::<f32>().map_err(|error| {
            F5Error::ModelExecution(format!("SpeedTransformer logits download: {error}"))
        })?;
        values
            .chunks_exact(SPEED_CLASSES)
            .take(batch)
            .map(|row| speed_from_label(crate::cross_lingual::best_speed_class(row)?))
            .collect()
    }
}

pub struct CrossLingualSpeedPredictor {
    mel: VocosMel,
    network: SpeedTransformer,
}

impl CrossLingualSpeedPredictor {
    pub fn from_checkpoint(ckpt: &Checkpoint, device: &Device) -> Result<Self> {
        Ok(Self {
            mel: VocosMel::new(),
            network: SpeedTransformer::from_checkpoint(
                ckpt,
                SpeedTransformerConfig::CROSS_LINGUAL,
                device,
            )?,
        })
    }

    pub fn predict_mel(&self, mel: Tensor<3>, lengths: &[usize]) -> Result<Vec<f32>> {
        self.network.predict_speeds(mel, lengths)
    }

    pub fn predict_audio(&self, mono_24khz: &[f32], device: &Device) -> Result<f32> {
        let frames = self.mel.extract(mono_24khz)?;
        let values = Tensor::<3>::from_data(
            TensorData::new(frames.values, [1, frames.frames, MEL_CHANNELS]),
            device,
        );
        Ok(self.network.predict_speeds(values, &[frames.frames])?[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_speed_network_configs() {
        assert!(
            SpeedTransformerConfig {
                heads: 3,
                ..SpeedTransformerConfig::CROSS_LINGUAL
            }
            .validate()
            .is_err()
        );
        assert!(
            SpeedTransformerConfig {
                position_groups: 0,
                ..SpeedTransformerConfig::CROSS_LINGUAL
            }
            .validate()
            .is_err()
        );
    }

    #[cfg(feature = "cubecl-cpu")]
    #[test]
    fn masked_speed_transformer_matches_independent_torch() {
        use std::path::Path;
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let spec: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/speed_predictor_expected.json"
        ))
        .unwrap();
        let read = |key: &str| -> Vec<f32> {
            spec[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };
        let device = Device::cpu();
        let checkpoint =
            Checkpoint::open(fixture_dir.join("speed_predictor_small.safetensors")).unwrap();
        let config = SpeedTransformerConfig {
            dim: spec["dim"].as_u64().unwrap() as usize,
            depth: spec["depth"].as_u64().unwrap() as usize,
            heads: spec["heads"].as_u64().unwrap() as usize,
            ff_mult: spec["ff_mult"].as_u64().unwrap() as usize,
            position_groups: spec["position_groups"].as_u64().unwrap() as usize,
        };
        eprintln!("speed parity: load checkpoint");
        let model = SpeedTransformer::from_checkpoint(&checkpoint, config, &device).unwrap();
        eprintln!("speed parity: checkpoint loaded");
        let batch = spec["batch"].as_u64().unwrap() as usize;
        let frames = spec["frames"].as_u64().unwrap() as usize;
        let lengths: Vec<usize> = spec["lengths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let mel =
            Tensor::<3>::from_data(TensorData::new(read("mel"), [batch, frames, 100]), &device);
        eprintln!("speed parity: forward");
        let logits = model
            .forward(mel, &lengths)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        eprintln!("speed parity: forward complete");
        let target = read("logits");
        let worst = logits
            .iter()
            .zip(&target)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("SpeedTransformer PyTorch parity max_abs={worst:.8}");
        assert!(
            worst < 1e-3,
            "SpeedTransformer PyTorch parity max_abs={worst}"
        );
        for (row, expected) in logits
            .chunks_exact(SPEED_CLASSES)
            .zip(spec["classes"].as_array().unwrap())
        {
            assert_eq!(
                crate::cross_lingual::best_speed_class(row).unwrap(),
                expected.as_u64().unwrap() as usize
            );
        }
    }
}
