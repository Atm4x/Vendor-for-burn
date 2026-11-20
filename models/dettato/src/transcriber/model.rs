use crate::{checkpoint::Checkpoint, hcqt::Features};
use burn::tensor::{Device, Tensor, TensorData, activation, module, ops::ConvOptions};

pub(crate) struct BatchNorm {
    scale: Tensor<1>,
    shift: Tensor<1>,
}

impl BatchNorm {
    pub(crate) fn load(ckpt: &Checkpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let (_, weights) = ckpt.values(&format!("{prefix}.weight"))?;
        let (_, biases) = ckpt.values(&format!("{prefix}.bias"))?;
        let (_, means) = ckpt.values(&format!("{prefix}.running_mean"))?;
        let (_, variances) = ckpt.values(&format!("{prefix}.running_var"))?;
        let len = weights.len();
        if biases.len() != len || means.len() != len || variances.len() != len {
            return Err(format!("{prefix}: inconsistent batch norm channel counts"));
        }
        let mut scale = Vec::with_capacity(len);
        let mut shift = Vec::with_capacity(len);
        for i in 0..len {
            let s = weights[i] / (variances[i] + 1e-5).sqrt();
            scale.push(s);
            shift.push(biases[i] - means[i] * s);
        }
        Ok(Self {
            scale: Tensor::from_data(TensorData::new(scale, [len]), device),
            shift: Tensor::from_data(TensorData::new(shift, [len]), device),
        })
    }

    pub(crate) fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let n = self.scale.dims()[0];
        input * self.scale.clone().reshape([1, n, 1, 1])
            + self.shift.clone().reshape([1, n, 1, 1])
    }
}

pub(crate) struct Conv2d {
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

impl Conv2d {
    pub(crate) fn load(
        ckpt: &Checkpoint,
        key: &str,
        device: &Device,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Self, String> {
        let weight = ckpt.tensor::<4>(&format!("{key}.weight"), device)?;
        let bias = ckpt.has(&format!("{key}.bias"))
            .then(|| ckpt.tensor::<1>(&format!("{key}.bias"), device))
            .transpose()?;
        let [out, input, _, _] = weight.dims();
        if groups == 0 || out % groups != 0 || input == 0 {
            return Err(format!("{key}: invalid grouped convolution"));
        }
        if bias.as_ref().is_some_and(|b| b.dims()[0] != out) {
            return Err(format!("{key}: invalid bias dimension"));
        }
        Ok(Self { weight, bias, stride, padding, dilation, groups })
    }

    pub(crate) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        module::conv2d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            ConvOptions::new(self.stride, self.padding, self.dilation, self.groups),
        )
    }
}

pub struct Stem {
    input_norm: BatchNorm,
    first: Conv2d,
    first_norm: BatchNorm,
    second: Conv2d,
    second_norm: BatchNorm,
    down: Conv2d,
    output_norm: BatchNorm,
}

impl Stem {
    pub fn load(ckpt: &Checkpoint, device: &Device, time_stride: usize) -> Result<Self, String> {
        if time_stride == 0 { return Err("invalid stem time stride".into()); }
        let kt = if time_stride == 2 { 3 } else { 1 };
        Ok(Self {
            input_norm: BatchNorm::load(ckpt, "stem.inorm", device)?,
            first: Conv2d::load(ckpt, "stem.c1", device, [1, 1], [1, 1], [1, 1], 1)?,
            first_norm: BatchNorm::load(ckpt, "stem.n1", device)?,
            second: Conv2d::load(ckpt, "stem.c2", device, [1, 1], [1, 1], [1, 1], 1)?,
            second_norm: BatchNorm::load(ckpt, "stem.n2", device)?,
            down: Conv2d::load(ckpt, "stem.down", device, [time_stride, 3], [kt / 2, 1], [1, 1], 1)?,
            output_norm: BatchNorm::load(ckpt, "stem.n3", device)?,
        })
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.input_norm.forward(x);
        let x = activation::gelu(self.first_norm.forward(self.first.forward(x)));
        let x = activation::gelu(self.second_norm.forward(self.second.forward(x)));
        self.output_norm.forward(self.down.forward(x))
    }

    pub fn features(&self, features: &Features, device: &Device) -> Result<Tensor<4>, String> {
        if features.frames == 0 { return Err("empty HCQT frames".into()); }
        let input = Tensor::from_data(
            TensorData::new(features.data.clone(), [1, features.channels, features.frames, features.bins]),
            device,
        );
        Ok(self.forward(input))
    }
}

pub struct PitchMix {
    taps: Tensor<2>,
    offsets: Vec<isize>,
}

impl PitchMix {
    pub fn load(
        ckpt: &Checkpoint,
        key: &str,
        device: &Device,
        offsets: Vec<isize>,
    ) -> Result<Self, String> {
        let taps = ckpt.tensor::<2>(&format!("{key}.taps"), device)?;
        if taps.dims()[1] != offsets.len() {
            return Err(format!("{key}: number of offsets differs from taps"));
        }
        Ok(Self { taps, offsets })
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [b, channels, time, pitches] = x.dims();
        let device = x.device();
        let mut y = Tensor::<4>::zeros([b, channels, time, pitches], &device);
        for (k, &offset) in self.offsets.iter().enumerate() {
            if offset.unsigned_abs() >= pitches { continue; }
            let shifted = if offset > 0 {
                let n = offset as usize;
                Tensor::cat(vec![
                    x.clone().narrow(3, n, pitches - n),
                    Tensor::zeros([b, channels, time, n], &device),
                ], 3)
            } else if offset < 0 {
                let n = offset.unsigned_abs();
                Tensor::cat(vec![
                    Tensor::zeros([b, channels, time, n], &device),
                    x.clone().narrow(3, 0, pitches - n),
                ], 3)
            } else {
                x.clone()
            };
            let tap = self.taps.clone().narrow(1, k, 1).reshape([1, channels, 1, 1]);
            y = y + shifted * tap;
        }
        y
    }
}

pub struct LaneBlock {
    pitch_mix: PitchMix,
    depthwise: Conv2d,
    norm: BatchNorm,
    expand: Conv2d,
    project: Conv2d,
    gamma: Tensor<1>,
}

impl LaneBlock {
    pub fn load(
        ckpt: &Checkpoint,
        key: &str,
        device: &Device,
        dilation: usize,
        offsets: Vec<isize>,
    ) -> Result<Self, String> {
        if dilation == 0 { return Err("invalid lane block dilation".into()); }
        let gamma = ckpt.tensor::<1>(&format!("{key}.gamma"), device)?;
        let channels = gamma.dims()[0];
        let depthwise_weight = ckpt.tensor::<4>(&format!("{key}.dw_t.weight"), device)?;
        let [output, input, kernel_t, kernel_p] = depthwise_weight.dims();
        if output != channels || input != 1 || kernel_p != 1 || kernel_t % 2 == 0 {
            return Err(format!("{key}: invalid depthwise kernel"));
        }
        Ok(Self {
            pitch_mix: PitchMix::load(ckpt, &format!("{key}.pmix"), device, offsets)?,
            depthwise: Conv2d::load(
                ckpt, &format!("{key}.dw_t"), device,
                [1, 1], [(kernel_t / 2) * dilation, 0], [dilation, 1], channels,
            )?,
            norm: BatchNorm::load(ckpt, &format!("{key}.norm"), device)?,
            expand: Conv2d::load(ckpt, &format!("{key}.pw1"), device, [1, 1], [0, 0], [1, 1], 1)?,
            project: Conv2d::load(ckpt, &format!("{key}.pw2"), device, [1, 1], [0, 0], [1, 1], 1)?,
            gamma,
        })
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let t = self.depthwise.forward(x.clone());
        let t = t.clone() + self.pitch_mix.forward(t);
        let t = self.norm.forward(t);
        let t = self.project.forward(activation::gelu(self.expand.forward(t)));
        let c = self.gamma.dims()[0];
        x + t * self.gamma.clone().reshape([1, c, 1, 1])
    }
}
