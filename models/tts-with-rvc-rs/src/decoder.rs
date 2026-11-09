use burn::tensor::{activation, module, ops::{ConvOptions, ConvTransposeOptions}, Device, Tensor, TensorData};
use serde::Deserialize;

use crate::checkpoint::RvcCheckpoint;

struct Conv1d {
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    stride: usize,
    dilation: usize,
    padding: usize,
}

impl Conv1d {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, stride: usize, dilation: usize, padding: usize, device: &Device) -> Result<Self, String> {
        if stride == 0 || dilation == 0 {
            return Err(format!("{prefix}: stride/dilation must be positive"));
        }
        let weight = if checkpoint.contains(&format!("{prefix}.weight_v")) {
            checkpoint.weight_norm_conv1d(prefix, device)?
        } else {
            checkpoint.tensor::<3>(&format!("{prefix}.weight"), device)?
        };
        let bias = checkpoint.contains(&format!("{prefix}.bias"))
            .then(|| checkpoint.tensor::<1>(&format!("{prefix}.bias"), device))
            .transpose()?;
        if let Some(bias) = &bias {
            if bias.dims()[0] != weight.dims()[0] {
                return Err(format!("{prefix}: bias and weight output channels differ"));
            }
        }
        Ok(Self {weight, bias, stride, dilation, padding})
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        module::conv1d(input, self.weight.clone(), self.bias.clone(),
            ConvOptions::new_with_padding([self.stride], [(self.padding, self.padding)], [self.dilation], 1))
    }
}

struct TransposedConv1d {
    weight: Tensor<3>,
    bias: Tensor<1>,
    stride: usize,
    padding: usize,
}

impl TransposedConv1d {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, stride: usize, device: &Device) -> Result<Self, String> {
        if stride == 0 { return Err(format!("{prefix}: invalid stride")); }
        let weight = checkpoint.weight_norm_conv1d(prefix, device)?;
        let bias = checkpoint.tensor::<1>(&format!("{prefix}.bias"), device)?;
        let [_, output_channels, kernel] = weight.dims();
        if kernel < stride || (kernel - stride) % 2 != 0 || bias.dims()[0] != output_channels {
            return Err(format!("{prefix}: unsupported ConvTranspose1d geometry"));
        }
        Ok(Self {weight, bias, stride, padding: (kernel - stride) / 2})
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        module::conv_transpose1d(input, self.weight.clone(), Some(self.bias.clone()),
            ConvTransposeOptions::new([self.stride], [self.padding], [0], [1], 1))
    }
}

pub struct ResBlock1 {
    first: Vec<Conv1d>,
    second: Vec<Conv1d>,
    channels: usize,
}

impl ResBlock1 {
    pub fn load(checkpoint: &RvcCheckpoint, index: usize, dilations: [usize; 3], device: &Device) -> Result<Self, String> {
        let prefix = format!("dec.resblocks.{index}");
        let mut first = Vec::with_capacity(dilations.len());
        let mut second = Vec::with_capacity(dilations.len());
        let mut channels = None;
        for (layer, dilation) in dilations.into_iter().enumerate() {
            for (family, dilation, target) in [("convs1", dilation, &mut first), ("convs2", 1, &mut second)] {
                let name = format!("{prefix}.{family}.{layer}");
                let shape = checkpoint.tensor_f32(&format!("{name}.weight_v"))?.0;
                if shape.len() != 3 || shape[0] != shape[1] || shape[2] % 2 == 0 {
                    return Err(format!("{name}: expected same-channel odd-kernel Conv1d"));
                }
                if channels.get_or_insert(shape[0]) != &shape[0] {
                    return Err(format!("{name}: inconsistent channel count"));
                }
                target.push(Conv1d::load(checkpoint, &name, 1, dilation, (shape[2] - 1) * dilation / 2, device)?);
            }
        }
        Ok(Self {first, second, channels: channels.unwrap()})
    }

    pub fn forward(&self, mut x: Tensor<3>) -> Tensor<3> {
        assert_eq!(x.dims()[1], self.channels);
        for (first, second) in self.first.iter().zip(&self.second) {
            let h = second.forward(activation::leaky_relu(first.forward(activation::leaky_relu(x.clone(), 0.1)), 0.1));
            x = x + h;
        }
        x
    }
}

pub struct DecoderStage {
    upsample: TransposedConv1d,
    noise: Conv1d,
    residual: Vec<ResBlock1>,
}

impl DecoderStage {
    pub fn load(checkpoint: &RvcCheckpoint, stage: usize, stride: usize, future_stride: usize, resblock_kernels: usize, dilations: [usize; 3], device: &Device) -> Result<Self, String> {
        if future_stride == 0 || resblock_kernels == 0 {
            return Err("decoder stage requires a positive source stride and residual count".into());
        }
        let upsample = TransposedConv1d::load(checkpoint, &format!("dec.ups.{stage}"), stride, device)?;
        let name = format!("dec.noise_convs.{stage}");
        let noise_shape = checkpoint.tensor_f32(&format!("{name}.weight"))?.0;
        if noise_shape.len() != 3 || noise_shape[0] != upsample.bias.dims()[0] || noise_shape[1] != 1 {
            return Err(format!("{name}: source channel dimensions do not match upsampling"));
        }
        let noise = Conv1d::load(checkpoint, &name, future_stride, 1, future_stride / 2, device)?;
        let mut residual = Vec::with_capacity(resblock_kernels);
        for i in 0..resblock_kernels {
            let rb = ResBlock1::load(checkpoint, stage * resblock_kernels + i, dilations, device)?;
            if rb.channels != upsample.bias.dims()[0] {
                return Err(format!("stage {stage}: residual channels don't match upsampling"));
            }
            residual.push(rb);
        }
        Ok(Self { upsample, noise, residual })
    }

    pub fn upsample_only(&self, input: Tensor<3>) -> Tensor<3> {
        self.upsample.forward(activation::leaky_relu(input, 0.1))
    }

    pub fn forward(&self, input: Tensor<3>, harmonic_source: Tensor<3>) -> Tensor<3> {
        let upsampled = self.upsample_only(input);
        let source = self.noise.forward(harmonic_source);
        assert_eq!(upsampled.dims(), source.dims(), "harmonic source must align with upsampled features");
        let mut result = Tensor::<3>::zeros(upsampled.dims(), &upsampled.device());
        let prepared = upsampled + source;
        for block in &self.residual {
            result = result + block.forward(prepared.clone());
        }
        result / self.residual.len() as f32
    }

    pub fn residual_count(&self) -> usize { self.residual.len() }
}

#[derive(Debug, Deserialize)]
pub struct DecoderStageFixture {
    pub stage: usize,
    pub stride: usize,
    pub future_stride: usize,
    pub num_kernels: usize,
    pub dilations: [usize; 3],
    pub input_shape: [usize; 3],
    pub source_shape: [usize; 3],
    pub input: Vec<f32>,
    pub harmonic_source: Vec<f32>,
    pub expected_upsampled: Vec<f32>,
    pub expected_output: Vec<f32>,
}

impl DecoderStageFixture {
    pub fn tensors(&self, device: &Device) -> (Tensor<3>, Tensor<3>) {
        let input = Tensor::from_data(TensorData::new(self.input.clone(), self.input_shape), device);
        let source = Tensor::from_data(TensorData::new(self.harmonic_source.clone(), self.source_shape), device);
        (input, source)
    }

    pub fn load(&self, checkpoint: &RvcCheckpoint, device: &Device) -> Result<DecoderStage, String> {
        DecoderStage::load(checkpoint, self.stage, self.stride, self.future_stride, self.num_kernels, self.dilations, device)
    }
}
