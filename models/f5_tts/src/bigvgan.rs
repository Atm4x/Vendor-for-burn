//! Native BigVGAN-v2 inference. All convolutions and anti-aliased periodic
//! activations execute on the selected Burn backend, without a Python worker.
use burn::{
    module::Param,
    nn::{
        PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig},
    },
    tensor::{Device, Tensor, TensorData, ops::PadMode},
};

use crate::{F5Error, Result, checkpoint::Checkpoint};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BigVganConfig {
    pub num_mels: usize,
    pub initial_channels: usize,
    pub upsample_rates: [usize; 6],
    pub upsample_kernels: [usize; 6],
    pub resblock_kernels: [usize; 3],
    pub resblock_dilations: [usize; 3],
    pub snake_logscale: bool,
}

impl BigVganConfig {
    pub const V2_24KHZ_100BAND_256X: Self = Self {
        num_mels: 100,
        initial_channels: 1536,
        upsample_rates: [4, 4, 2, 2, 2, 2],
        upsample_kernels: [8, 8, 4, 4, 4, 4],
        resblock_kernels: [3, 7, 11],
        resblock_dilations: [1, 3, 5],
        snake_logscale: true,
    };

    pub fn validate(self) -> Result<()> {
        if self.num_mels != 100
            || self.initial_channels == 0
            || self.initial_channels % 64 != 0
            || self.upsample_rates.iter().product::<usize>() != 256
            || self.upsample_rates.iter().any(|&r| r == 0)
            || self
                .upsample_kernels
                .iter()
                .zip(self.upsample_rates)
                .any(|(&k, r)| k < r || (k - r) % 2 != 0)
            || self.resblock_kernels.iter().any(|&k| k % 2 != 1 || k == 0)
            || self.resblock_dilations.contains(&0)
        {
            return Err(F5Error::InvalidConfiguration(
                "Invalid BigVGAN-v2 24kHz config",
            ));
        }
        Ok(())
    }
}

fn normalized_weight(ckpt: &Checkpoint, key: &str, shape: [usize; 3]) -> Result<Vec<f32>> {
    let full = format!("{key}.weight");
    if ckpt.tensors().contains_key(&full) {
        return ckpt.read_f32(&full, &shape);
    }
    let data = ckpt.read_f32(&format!("{key}.weight_v"), &shape)?;
    let gains = ckpt.read_f32(&format!("{key}.weight_g"), &[shape[0], 1, 1])?;
    let mut weights = data;
    let per_channel = shape[1] * shape[2];
    for (i, gain) in gains.iter().enumerate() {
        let span = &mut weights[i * per_channel..(i + 1) * per_channel];
        let norm = span
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            return Err(F5Error::InvalidCheckpoint(format!(
                "BigVGAN zero weight norm: {key}"
            )));
        }
        for weight in span {
            *weight *= (*gain as f64 / norm) as f32;
        }
    }
    Ok(weights)
}

fn load_conv(
    ckpt: &Checkpoint,
    key: &str,
    input: usize,
    output: usize,
    kernel: usize,
    dilation: usize,
    bias: bool,
    device: &Device,
) -> Result<Conv1d> {
    let pad = (kernel - 1) * dilation / 2;
    let mut conv = Conv1dConfig::new(input, output, kernel)
        .with_padding(PaddingConfig1d::Explicit(pad, pad))
        .with_dilation(dilation)
        .with_bias(bias)
        .init(device);
    conv.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(
            normalized_weight(ckpt, key, [output, input, kernel])?,
            [output, input, kernel],
        ),
        device,
    ));
    if bias {
        conv.bias = Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(ckpt.read_f32(&format!("{key}.bias"), &[output])?, [output]),
            device,
        )));
    }
    Ok(conv)
}

fn load_upsample(
    ckpt: &Checkpoint,
    key: &str,
    input: usize,
    output: usize,
    rate: usize,
    kernel: usize,
    device: &Device,
) -> Result<ConvTranspose1d> {
    let mut conv = ConvTranspose1dConfig::new([input, output], kernel)
        .with_stride(rate)
        .with_padding((kernel - rate) / 2)
        .init(device);
    conv.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(
            normalized_weight(ckpt, key, [input, output, kernel])?,
            [input, output, kernel],
        ),
        device,
    ));
    conv.bias = Some(Param::from_tensor(Tensor::from_data(
        TensorData::new(ckpt.read_f32(&format!("{key}.bias"), &[output])?, [output]),
        device,
    )));
    Ok(conv)
}

fn bessel_i0(x: f64) -> f64 {
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for k in 1..60 {
        term *= (x * x / 4.0) / (k * k) as f64;
        sum += term;
        if term < sum * 1e-15 {
            break;
        }
    }
    sum
}

fn kaiser_sinc_filter(cutoff: f64, half_width: f64, size: usize) -> Vec<f32> {
    let half = size / 2;
    let a = 2.285 * (half - 1) as f64 * std::f64::consts::PI * 4.0 * half_width + 7.95;
    let beta = if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    };
    let denom = bessel_i0(beta);
    let mut result = Vec::with_capacity(size);
    for index in 0..size {
        let pos = index as f64 / (size - 1) as f64;
        let window = bessel_i0(beta * (1.0 - (2.0 * pos - 1.0).powi(2)).max(0.0).sqrt()) / denom;
        let time = index as f64 - half as f64 + if size % 2 == 0 { 0.5 } else { 0.0 };
        let v = 2.0
            * cutoff
            * window
            * if time == 0.0 {
                1.0
            } else {
                (2.0 * cutoff * std::f64::consts::PI * time).sin()
                    / (2.0 * cutoff * std::f64::consts::PI * time)
            };
        result.push(v as f32);
    }
    let sum = result.iter().map(|&v| v as f64).sum::<f64>();
    result
        .iter_mut()
        .for_each(|x| *x = (*x as f64 / sum) as f32);
    result
}

struct PeriodicActivation {
    alpha: Tensor<3>,
    beta: Tensor<3>,
    up: ConvTranspose1d,
    down: Conv1d,
}

impl PeriodicActivation {
    fn from_checkpoint(
        ckpt: &Checkpoint,
        key: &str,
        channels: usize,
        logscale: bool,
        device: &Device,
    ) -> Result<Self> {
        let mut alpha = ckpt.read_f32(&format!("{key}.act.alpha"), &[channels])?;
        let mut beta = ckpt.read_f32(&format!("{key}.act.beta"), &[channels])?;
        if logscale {
            alpha.iter_mut().for_each(|v| *v = v.exp());
            beta.iter_mut().for_each(|v| *v = v.exp());
        }
        let filter = kaiser_sinc_filter(0.25, 0.3, 12);
        let mut up = ConvTranspose1dConfig::new([channels, channels], 12)
            .with_groups(channels)
            .with_stride(2)
            .with_bias(false)
            .init(device);
        let mut down = Conv1dConfig::new(channels, channels, 12)
            .with_groups(channels)
            .with_stride(2)
            .with_bias(false)
            .with_padding(PaddingConfig1d::Valid)
            .init(device);
        let taps = (0..channels)
            .flat_map(|_| filter.iter().copied())
            .collect::<Vec<_>>();
        up.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(taps.clone(), [channels, 1, 12]),
            device,
        ));
        down.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(taps, [channels, 1, 12]),
            device,
        ));
        Ok(Self {
            alpha: Tensor::from_data(TensorData::new(alpha, [1, channels, 1]), device),
            beta: Tensor::from_data(TensorData::new(beta, [1, channels, 1]), device),
            up,
            down,
        })
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let len = input.dims()[2];
        let up = self.up.forward(input.pad([(5, 5)], PadMode::Edge)) * 2.0;
        let signal = up.slice([0..1, 0..self.alpha.dims()[1], 15..15 + len * 2]);
        let sine = (signal.clone() * self.alpha.clone()).sin();
        let activated = signal + sine.clone() * sine / (self.beta.clone() + 1e-9);
        self.down.forward(activated.pad([(5, 6)], PadMode::Edge))
    }
}

struct AmpResBlock {
    layers: Vec<(PeriodicActivation, Conv1d, PeriodicActivation, Conv1d)>,
}

impl AmpResBlock {
    fn from_checkpoint(
        ckpt: &Checkpoint,
        key: &str,
        channels: usize,
        kernel: usize,
        dilations: &[usize; 3],
        logscale: bool,
        device: &Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(dilations.len());
        for (i, &dilation) in dilations.iter().enumerate() {
            layers.push((
                PeriodicActivation::from_checkpoint(
                    ckpt,
                    &format!("{key}.activations.{}", i * 2),
                    channels,
                    logscale,
                    device,
                )?,
                load_conv(
                    ckpt,
                    &format!("{key}.convs1.{i}"),
                    channels,
                    channels,
                    kernel,
                    dilation,
                    true,
                    device,
                )?,
                PeriodicActivation::from_checkpoint(
                    ckpt,
                    &format!("{key}.activations.{}", i * 2 + 1),
                    channels,
                    logscale,
                    device,
                )?,
                load_conv(
                    ckpt,
                    &format!("{key}.convs2.{i}"),
                    channels,
                    channels,
                    kernel,
                    1,
                    true,
                    device,
                )?,
            ));
        }
        Ok(Self { layers })
    }

    fn forward(&self, mut signal: Tensor<3>) -> Tensor<3> {
        for (act1, conv1, act2, conv2) in &self.layers {
            let branch = conv2.forward(act2.forward(conv1.forward(act1.forward(signal.clone()))));
            signal = signal + branch;
        }
        signal
    }
}

struct BigVganStage {
    up: ConvTranspose1d,
    residual: Vec<AmpResBlock>,
}

pub struct BigVganDecoder {
    pub config: BigVganConfig,
    pre: Conv1d,
    stages: Vec<BigVganStage>,
    post_activation: PeriodicActivation,
    post: Conv1d,
}

impl BigVganDecoder {
    pub fn from_checkpoint(
        ckpt: &Checkpoint,
        config: BigVganConfig,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        let pre = load_conv(
            ckpt,
            "conv_pre",
            config.num_mels,
            config.initial_channels,
            7,
            1,
            true,
            device,
        )?;
        let mut stages = Vec::with_capacity(6);
        let mut channels = config.initial_channels;
        for stage in 0..6 {
            let output = channels / 2;
            let up = load_upsample(
                ckpt,
                &format!("ups.{stage}.0"),
                channels,
                output,
                config.upsample_rates[stage],
                config.upsample_kernels[stage],
                device,
            )?;
            let mut residual = Vec::with_capacity(3);
            for kernel_idx in 0..3 {
                residual.push(AmpResBlock::from_checkpoint(
                    ckpt,
                    &format!("resblocks.{}", stage * 3 + kernel_idx),
                    output,
                    config.resblock_kernels[kernel_idx],
                    &config.resblock_dilations,
                    config.snake_logscale,
                    device,
                )?);
            }
            stages.push(BigVganStage { up, residual });
            channels = output;
        }
        Ok(Self {
            config,
            pre,
            stages,
            post_activation: PeriodicActivation::from_checkpoint(
                ckpt,
                "activation_post",
                channels,
                config.snake_logscale,
                device,
            )?,
            post: load_conv(ckpt, "conv_post", channels, 1, 7, 1, false, device)?,
        })
    }

    /// [1, frames, 100] -> [1, 1, frames * 256], as with the official generator.
    pub fn forward(&self, mel: Tensor<3>) -> Result<Tensor<3>> {
        let [batch, frames, channels] = mel.dims();
        if batch != 1 || frames == 0 || channels != self.config.num_mels {
            return Err(F5Error::InvalidShape(
                "BigVGAN expects [1, frames, 100] mel",
            ));
        }
        let mut signal = self.pre.forward(mel.swap_dims(1, 2));
        for stage in &self.stages {
            signal = stage.up.forward(signal);
            let residuals = stage
                .residual
                .iter()
                .map(|r| r.forward(signal.clone()))
                .collect::<Vec<_>>();
            signal = residuals[0].clone();
            for residual in residuals.iter().skip(1) {
                signal = signal + residual.clone();
            }
            signal = signal / stage.residual.len() as f32;
        }
        let result = self
            .post
            .forward(self.post_activation.forward(signal))
            .clamp(-1.0, 1.0);
        if result.dims() != [1, 1, frames * 256] {
            return Err(F5Error::ModelExecution(
                "BigVGAN upsample output shape mismatch".into(),
            ));
        }
        Ok(result)
    }

    pub fn decode(&self, mel: Tensor<3>, gain: f32) -> Result<Vec<f32>> {
        if !gain.is_finite() || gain < 0.0 {
            return Err(F5Error::InvalidConfiguration(
                "Invalid BigVGAN restoration gain",
            ));
        }
        let mut pcm = self
            .forward(mel)?
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| F5Error::ModelExecution(format!("BigVGAN waveform download: {e}")))?;
        pcm.iter_mut().for_each(|v| *v *= gain);
        Ok(pcm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_v2_shape_and_kernels() {
        let config = BigVganConfig::V2_24KHZ_100BAND_256X;
        config.validate().unwrap();
        assert_eq!(config.upsample_rates.iter().product::<usize>(), 256);
        assert_eq!(config.initial_channels / 64, 24);
    }

    #[test]
    fn kaiser_antialias_kernel_is_normalized_and_symmetric() {
        let filter = kaiser_sinc_filter(0.25, 0.3, 12);
        assert!((filter.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for i in 0..6 {
            assert!((filter[i] - filter[11 - i]).abs() < 1e-7);
        }
    }
}
