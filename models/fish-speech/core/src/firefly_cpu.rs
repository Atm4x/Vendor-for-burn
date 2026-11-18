use crate::audio::{fish_conv_output_len, fish_conv_padding, fish_transconv_output_len};
use crate::error::{FishError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct AudioTensor {
    pub batch: usize,
    pub channels: usize,
    pub frames: usize,
    pub data: Vec<f32>,
}

impl AudioTensor {
    pub fn new(batch: usize, channels: usize, frames: usize, data: Vec<f32>) -> Result<Self> {
        let size = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(frames))
            .ok_or(FishError::InvalidShape("audio tensor extent overflow"))?;
        if batch == 0 || channels == 0 || frames == 0 || data.len() != size {
            return Err(FishError::InvalidShape("invalid NCT audio tensor"));
        }
        Ok(Self {
            batch,
            channels,
            frames,
            data,
        })
    }

    #[inline]
    fn index(&self, batch: usize, channel: usize, frame: usize) -> usize {
        (batch * self.channels + channel) * self.frames + frame
    }

    fn checked_shape(&self, channels: usize, frames: usize) -> Result<usize> {
        self.batch
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(frames))
            .ok_or(FishError::InvalidShape("audio operation extent overflow"))
    }
}

#[derive(Debug, Clone)]
pub struct Conv1dKernel {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    /// PyTorch Conv1d layout: [out_channels, in_channels/groups, kernel_size].
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl Conv1dKernel {
    pub fn view(&self) -> Conv1dView<'_> {
        Conv1dView {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            dilation: self.dilation,
            groups: self.groups,
            weights: &self.weights,
            bias: self.bias.as_deref(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.view().validate()
    }

    pub fn forward(&self, input: &AudioTensor) -> Result<AudioTensor> {
        self.view().forward(input)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Conv1dView<'a> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub weights: &'a [f32],
    pub bias: Option<&'a [f32]>,
}

impl Conv1dView<'_> {
    pub fn validate(&self) -> Result<()> {
        if self.in_channels == 0
            || self.out_channels == 0
            || self.kernel_size == 0
            || self.stride == 0
            || self.dilation == 0
            || self.groups == 0
            || self.in_channels % self.groups != 0
            || self.out_channels % self.groups != 0
        {
            return Err(FishError::InvalidShape(
                "invalid grouped Conv1d configuration",
            ));
        }
        let expected = self
            .out_channels
            .checked_mul(self.in_channels / self.groups)
            .and_then(|n| n.checked_mul(self.kernel_size))
            .ok_or(FishError::InvalidShape("Conv1d weights extent overflow"))?;
        if self.weights.len() != expected
            || self
                .bias
                .as_ref()
                .is_some_and(|b| b.len() != self.out_channels)
        {
            return Err(FishError::InvalidShape(
                "Conv1d weights or bias shape mismatch",
            ));
        }
        Ok(())
    }

    pub fn forward(&self, input: &AudioTensor) -> Result<AudioTensor> {
        self.validate()?;
        if input.channels != self.in_channels {
            return Err(FishError::InvalidShape(
                "Conv1d input channel count mismatch",
            ));
        }
        let length =
            fish_conv_output_len(input.frames, self.kernel_size, self.stride, self.dilation)?;
        let (pad_left, _) =
            fish_conv_padding(input.frames, self.kernel_size, self.stride, self.dilation)?;
        let mut output = vec![0.0; input.checked_shape(self.out_channels, length)?];
        let in_per_group = self.in_channels / self.groups;
        let out_per_group = self.out_channels / self.groups;
        for b in 0..input.batch {
            for out_ch in 0..self.out_channels {
                let group = out_ch / out_per_group;
                for t in 0..length {
                    let mut accum = self.bias.as_ref().map_or(0.0, |bias| bias[out_ch]);
                    for local_in in 0..in_per_group {
                        let in_ch = group * in_per_group + local_in;
                        for k in 0..self.kernel_size {
                            let padded_t = t * self.stride + k * self.dilation;
                            if padded_t < pad_left {
                                continue;
                            }
                            let original_t = padded_t - pad_left;
                            if original_t >= input.frames {
                                continue;
                            }
                            let w_index = (out_ch * in_per_group + local_in) * self.kernel_size + k;
                            accum += input.data[input.index(b, in_ch, original_t)]
                                * self.weights[w_index];
                        }
                    }
                    output[(b * self.out_channels + out_ch) * length + t] = accum;
                }
            }
        }
        AudioTensor::new(input.batch, self.out_channels, length, output)
    }
}

#[derive(Debug, Clone)]
pub struct TransConv1dKernel {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    /// PyTorch ConvTranspose1d layout: [in_channels, out_channels, kernel_size], groups=1.
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl TransConv1dKernel {
    pub fn view(&self) -> TransConv1dView<'_> {
        TransConv1dView {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            dilation: self.dilation,
            weights: &self.weights,
            bias: self.bias.as_deref(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.view().validate()
    }

    pub fn forward(&self, input: &AudioTensor) -> Result<AudioTensor> {
        self.view().forward(input)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TransConv1dView<'a> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub weights: &'a [f32],
    pub bias: Option<&'a [f32]>,
}

impl TransConv1dView<'_> {
    pub fn validate(&self) -> Result<()> {
        if self.in_channels == 0
            || self.out_channels == 0
            || self.kernel_size == 0
            || self.stride == 0
            || self.dilation == 0
        {
            return Err(FishError::InvalidShape(
                "invalid ConvTranspose1d configuration",
            ));
        }
        let expected = self
            .in_channels
            .checked_mul(self.out_channels)
            .and_then(|n| n.checked_mul(self.kernel_size))
            .ok_or(FishError::InvalidShape("ConvTranspose1d weights overflow"))?;
        if self.weights.len() != expected
            || self
                .bias
                .as_ref()
                .is_some_and(|b| b.len() != self.out_channels)
        {
            return Err(FishError::InvalidShape(
                "ConvTranspose1d weights or bias mismatch",
            ));
        }
        Ok(())
    }

    pub fn forward(&self, input: &AudioTensor) -> Result<AudioTensor> {
        self.validate()?;
        if input.channels != self.in_channels {
            return Err(FishError::InvalidShape(
                "ConvTranspose1d input channels mismatch",
            ));
        }
        let out_len =
            fish_transconv_output_len(input.frames, self.kernel_size, self.stride, self.dilation)?;
        let raw_len = (input.frames - 1)
            .checked_mul(self.stride)
            .and_then(|n| n.checked_add((self.kernel_size - 1).checked_mul(self.dilation)?))
            .and_then(|n| n.checked_add(1))
            .ok_or(FishError::InvalidShape(
                "ConvTranspose1d raw length overflow",
            ))?;
        let trim = self.kernel_size - self.stride;
        if raw_len < trim || raw_len - trim != out_len {
            return Err(FishError::InvalidShape(
                "ConvTranspose1d trim length mismatch",
            ));
        }
        let mut output = vec![0.0; input.checked_shape(self.out_channels, out_len)?];
        for b in 0..input.batch {
            for out_ch in 0..self.out_channels {
                for t in 0..out_len {
                    output[(b * self.out_channels + out_ch) * out_len + t] =
                        self.bias.as_ref().map_or(0.0, |bias| bias[out_ch]);
                }
            }
            for in_ch in 0..self.in_channels {
                for t in 0..input.frames {
                    let x = input.data[input.index(b, in_ch, t)];
                    for k in 0..self.kernel_size {
                        let raw_t = t * self.stride + k * self.dilation;
                        if raw_t >= out_len {
                            continue;
                        }
                        for out_ch in 0..self.out_channels {
                            let wi = (in_ch * self.out_channels + out_ch) * self.kernel_size + k;
                            output[(b * self.out_channels + out_ch) * out_len + raw_t] +=
                                x * self.weights[wi];
                        }
                    }
                }
            }
        }
        AudioTensor::new(input.batch, self.out_channels, out_len, output)
    }
}

pub fn layer_norm_channels_first(
    input: &AudioTensor,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Result<AudioTensor> {
    if !epsilon.is_finite()
        || epsilon <= 0.0
        || weight.len() != input.channels
        || bias.len() != input.channels
    {
        return Err(FishError::InvalidShape("invalid layer-norm parameters"));
    }
    let mut output = input.clone();
    for b in 0..input.batch {
        for t in 0..input.frames {
            let mean = (0..input.channels)
                .map(|c| input.data[input.index(b, c, t)])
                .sum::<f32>()
                / input.channels as f32;
            let variance = (0..input.channels)
                .map(|c| (input.data[input.index(b, c, t)] - mean).powi(2))
                .sum::<f32>()
                / input.channels as f32;
            let inv = (variance + epsilon).sqrt().recip();
            for c in 0..input.channels {
                let idx = input.index(b, c, t);
                output.data[idx] = (input.data[idx] - mean) * inv * weight[c] + bias[c];
            }
        }
    }
    Ok(output)
}

fn erf_approx(value: f64) -> f64 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    sign * (1.0 - p * (-x * x).exp())
}

pub fn gelu(x: f32) -> f32 {
    (0.5 * x as f64 * (1.0 + erf_approx(x as f64 * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
pub struct ConvNextBlock {
    pub depthwise: Conv1dKernel,
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub pointwise1_weight: Vec<f32>,
    pub pointwise1_bias: Vec<f32>,
    pub pointwise2_weight: Vec<f32>,
    pub pointwise2_bias: Vec<f32>,
    pub gamma: Option<Vec<f32>>,
    pub epsilon: f32,
}

impl ConvNextBlock {
    pub fn forward(&self, input: &AudioTensor) -> Result<AudioTensor> {
        let channels = input.channels;
        let hidden = self.pointwise1_bias.len();
        if hidden == 0
            || self.depthwise.in_channels != channels
            || self.depthwise.out_channels != channels
            || self.depthwise.groups != channels
            || self.norm_weight.len() != channels
            || self.norm_bias.len() != channels
            || self.pointwise1_weight.len() != channels * hidden
            || self.pointwise2_weight.len() != channels * hidden
            || self.pointwise2_bias.len() != channels
            || self.gamma.as_ref().is_some_and(|g| g.len() != channels)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
        {
            return Err(FishError::InvalidShape(
                "invalid ConvNeXt block parameter shapes",
            ));
        }
        let dw = self.depthwise.forward(input)?;
        if dw.frames != input.frames {
            return Err(FishError::InvalidShape("ConvNeXt depthwise length changed"));
        }
        let mut output = input.clone();
        let mut normalized = vec![0.0; channels];
        let mut intermediate = vec![0.0; hidden];
        for b in 0..input.batch {
            for t in 0..input.frames {
                let mean = (0..channels)
                    .map(|c| dw.data[dw.index(b, c, t)])
                    .sum::<f32>()
                    / channels as f32;
                let variance = (0..channels)
                    .map(|c| (dw.data[dw.index(b, c, t)] - mean).powi(2))
                    .sum::<f32>()
                    / channels as f32;
                let inv = (variance + self.epsilon).sqrt().recip();
                for (c, v) in normalized.iter_mut().enumerate() {
                    *v = ((dw.data[dw.index(b, c, t)] - mean) * inv) * self.norm_weight[c]
                        + self.norm_bias[c];
                }
                for (h, v) in intermediate.iter_mut().enumerate() {
                    let mut sum = self.pointwise1_bias[h];
                    for (c, &norm) in normalized.iter().enumerate() {
                        sum += self.pointwise1_weight[h * channels + c] * norm;
                    }
                    *v = gelu(sum);
                }
                for c in 0..channels {
                    let mut sum = self.pointwise2_bias[c];
                    for (h, &intermediate_value) in intermediate.iter().enumerate() {
                        sum += self.pointwise2_weight[c * hidden + h] * intermediate_value;
                    }
                    let scale = self.gamma.as_ref().map_or(1.0, |g| g[c]);
                    output.data[input.index(b, c, t)] += sum * scale;
                }
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct ResBlock1 {
    pub pairs: Vec<(Conv1dKernel, Conv1dKernel)>,
}

impl ResBlock1 {
    pub fn forward(&self, input: &AudioTensor) -> Result<AudioTensor> {
        if self.pairs.is_empty() {
            return Err(FishError::InvalidShape(
                "ResBlock1 must contain convolution pairs",
            ));
        }
        let mut x = input.clone();
        for (first, second) in &self.pairs {
            let activated = AudioTensor {
                data: x.data.iter().copied().map(silu).collect(),
                ..x.clone()
            };
            let intermediate = first.forward(&activated)?;
            let activated = AudioTensor {
                data: intermediate.data.iter().copied().map(silu).collect(),
                ..intermediate
            };
            let residual = second.forward(&activated)?;
            if residual.batch != x.batch
                || residual.channels != x.channels
                || residual.frames != x.frames
            {
                return Err(FishError::InvalidShape("ResBlock1 residual shape mismatch"));
            }
            for (a, b) in x.data.iter_mut().zip(&residual.data) {
                *a += b;
            }
        }
        Ok(x)
    }
}

pub fn parallel_resblocks_mean(input: &AudioTensor, branches: &[ResBlock1]) -> Result<AudioTensor> {
    if branches.is_empty() {
        return Err(FishError::InvalidShape("ParallelBlock requires branches"));
    }
    let mut out = AudioTensor {
        data: vec![0.0; input.data.len()],
        ..input.clone()
    };
    for branch in branches {
        let y = branch.forward(input)?;
        if y.data.len() != out.data.len() {
            return Err(FishError::InvalidShape("ParallelBlock shape mismatch"));
        }
        for (dst, val) in out.data.iter_mut().zip(&y.data) {
            *dst += val / branches.len() as f32;
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct HifiGanDecoder {
    pub pre: Conv1dKernel,
    pub upsample: Vec<TransConv1dKernel>,
    pub residuals: Vec<Vec<ResBlock1>>,
    pub post: Conv1dKernel,
}

impl HifiGanDecoder {
    pub fn forward(&self, input: &AudioTensor) -> Result<AudioTensor> {
        if self.upsample.is_empty() || self.upsample.len() != self.residuals.len() {
            return Err(FishError::InvalidShape(
                "HiFiGAN upsample and residual stages mismatch",
            ));
        }
        let mut x = self.pre.forward(input)?;
        for (up, blocks) in self.upsample.iter().zip(&self.residuals) {
            let act = AudioTensor {
                data: x.data.iter().copied().map(silu).collect(),
                ..x
            };
            x = up.forward(&act)?;
            x = parallel_resblocks_mean(&x, blocks)?;
        }
        let act = AudioTensor {
            data: x.data.iter().copied().map(silu).collect(),
            ..x
        };
        let mut out = self.post.forward(&act)?;
        if out.channels != 1 {
            return Err(FishError::InvalidShape("HiFiGAN output must be mono"));
        }
        for sample in &mut out.data {
            *sample = sample.tanh();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_input(batch: usize, channels: usize, frames: usize) -> AudioTensor {
        let n = batch * channels * frames;
        AudioTensor::new(
            batch,
            channels,
            frames,
            (0..n)
                .map(|i| ((i * 19 + 3) % 29) as f32 / 11.0 - 1.2)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn reject_bad_weight_layouts() {
        let x = example_input(1, 2, 5);
        let wrong = Conv1dKernel {
            in_channels: 2,
            out_channels: 1,
            kernel_size: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            weights: vec![0.0; 5],
            bias: None,
        };
        assert!(wrong.forward(&x).is_err());
    }

    #[test]
    fn convolution_identity_kernel() {
        let x = example_input(2, 2, 5);
        let kernel = Conv1dKernel {
            in_channels: 2,
            out_channels: 2,
            kernel_size: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            weights: vec![1.0, 0.0, 0.0, 1.0],
            bias: None,
        };
        assert_eq!(kernel.forward(&x).unwrap(), x);
    }

    #[test]
    fn transposed_identity_upsample() {
        let x = example_input(1, 1, 4);
        let kernel = TransConv1dKernel {
            in_channels: 1,
            out_channels: 1,
            kernel_size: 2,
            stride: 2,
            dilation: 1,
            weights: vec![1.0, 1.0],
            bias: None,
        };
        let out = kernel.forward(&x).unwrap();
        assert_eq!(out.frames, 8);
        for i in 0..4 {
            assert_eq!(&out.data[i * 2..i * 2 + 2], &[x.data[i], x.data[i]]);
        }
    }

    #[test]
    fn invalid_convnext_detected() {
        let x = example_input(1, 2, 4);
        let block = ConvNextBlock {
            depthwise: Conv1dKernel {
                in_channels: 2,
                out_channels: 2,
                kernel_size: 3,
                stride: 1,
                dilation: 1,
                groups: 2,
                weights: vec![0.0; 6],
                bias: None,
            },
            norm_weight: vec![1.0; 2],
            norm_bias: vec![0.0; 2],
            pointwise1_weight: vec![0.0; 4],
            pointwise1_bias: vec![0.0; 2],
            pointwise2_weight: vec![0.0; 3],
            pointwise2_bias: vec![0.0; 2],
            gamma: None,
            epsilon: 1e-6,
        };
        assert!(block.forward(&x).is_err());
    }
}
