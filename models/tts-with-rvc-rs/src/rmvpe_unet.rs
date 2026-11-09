use burn::tensor::{activation, module, ops::{AvgPoolOptions, ConvOptions}, Device, Tensor};
use crate::checkpoint::RvcCheckpoint;

struct BatchNorm2d {
    weight: Tensor<1>,
    bias: Tensor<1>,
    mean: Tensor<1>,
    variance: Tensor<1>,
}

impl BatchNorm2d {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let weight = checkpoint.tensor::<1>(&format!("{prefix}.weight"), device)?;
        let bias = checkpoint.tensor::<1>(&format!("{prefix}.bias"), device)?;
        let mean = checkpoint.tensor::<1>(&format!("{prefix}.running_mean"), device)?;
        let variance = checkpoint.tensor::<1>(&format!("{prefix}.running_var"), device)?;
        let channels = weight.dims();
        if channels[0] == 0 || bias.dims() != channels || mean.dims() != channels || variance.dims() != channels {
            return Err(format!("{prefix}: incompatible BatchNorm2d parameter dimensions"));
        }
        Ok(Self { weight, bias, mean, variance })
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let channels = self.weight.dims()[0];
        assert_eq!(input.dims()[1], channels, "BatchNorm2d channel mismatch");
        let shape = [1, channels, 1, 1];
        (input - self.mean.clone().reshape(shape))
            / (self.variance.clone().reshape(shape) + 1e-5).sqrt()
            * self.weight.clone().reshape(shape)
            + self.bias.clone().reshape(shape)
    }
}

struct Conv2d {
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    padding: usize,
}

impl Conv2d {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, kernel: usize, device: &Device) -> Result<Self, String> {
        let weight = checkpoint.tensor::<4>(&format!("{prefix}.weight"), device)?;
        let shape = weight.dims();
        if shape[0] == 0 || shape[1] == 0 || shape[2] != kernel || shape[3] != kernel || kernel % 2 == 0 {
            return Err(format!("{prefix}: unsupported Conv2d kernel dimensions {shape:?}"));
        }
        let bias = checkpoint.contains(&format!("{prefix}.bias"))
            .then(|| checkpoint.tensor::<1>(&format!("{prefix}.bias"), device))
            .transpose()?;
        if bias.as_ref().is_some_and(|bias| bias.dims()[0] != shape[0]) {
            return Err(format!("{prefix}: Conv2d bias channels mismatch"));
        }
        Ok(Self { weight, bias, padding: kernel / 2 })
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        module::conv2d(input, self.weight.clone(), self.bias.clone(),
            ConvOptions::new([1, 1], [self.padding, self.padding], [1, 1], 1))
    }
}

struct ConvBlockRes {
    conv1: Conv2d,
    norm1: BatchNorm2d,
    conv2: Conv2d,
    norm2: BatchNorm2d,
    shortcut: Option<Conv2d>,
}

impl ConvBlockRes {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, device: &Device) -> Result<Self, String> {
        let conv1 = Conv2d::load(checkpoint, &format!("{prefix}.conv.0"), 3, device)?;
        let norm1 = BatchNorm2d::load(checkpoint, &format!("{prefix}.conv.1"), device)?;
        let conv2 = Conv2d::load(checkpoint, &format!("{prefix}.conv.3"), 3, device)?;
        let norm2 = BatchNorm2d::load(checkpoint, &format!("{prefix}.conv.4"), device)?;
        let input_channels = conv1.weight.dims()[1];
        let hidden_channels = conv1.weight.dims()[0];
        if conv2.weight.dims()[1] != hidden_channels || conv2.weight.dims()[0] != hidden_channels ||
            norm1.weight.dims()[0] != hidden_channels || norm2.weight.dims()[0] != hidden_channels {
            return Err(format!("{prefix}: residual convolution channel mismatch"));
        }
        let shortcut_prefix = format!("{prefix}.shortcut");
        let shortcut = checkpoint.contains(&format!("{shortcut_prefix}.weight"))
            .then(|| Conv2d::load(checkpoint, &shortcut_prefix, 1, device))
            .transpose()?;
        if let Some(ref shortcut) = shortcut {
            if shortcut.weight.dims()[0] != hidden_channels || shortcut.weight.dims()[1] != input_channels {
                return Err(format!("{prefix}: shortcut channel mismatch"));
            }
        } else if input_channels != hidden_channels {
            return Err(format!("{prefix}: required shortcut missing"));
        }
        Ok(Self { conv1, norm1, conv2, norm2, shortcut })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let residual = match &self.shortcut {
            Some(shortcut) => shortcut.forward(x.clone()),
            None => x.clone(),
        };
        let y = activation::relu(self.norm1.forward(self.conv1.forward(x)));
        activation::relu(self.norm2.forward(self.conv2.forward(y))) + residual
    }
}

pub struct RmvpeEncoderStage {
    input_norm: Option<BatchNorm2d>,
    blocks: Vec<ConvBlockRes>,
    pool: [usize; 2],
}

pub struct RmvpeEncoderOutput {
    pub skip: Tensor<4>,
    pub pooled: Tensor<4>,
}

impl RmvpeEncoderStage {
    pub fn load(checkpoint: &RvcCheckpoint, stage: usize, device: &Device) -> Result<Self, String> {
        if stage >= 5 { return Err("RMVPE U-Net contains only five encoder stages".into()); }
        let input_norm = if stage == 0 {
            Some(BatchNorm2d::load(checkpoint, "unet.encoder.bn", device)?)
        } else { None };
        let prefix = format!("unet.encoder.layers.{stage}.conv");
        let blocks = load_blocks(checkpoint, &prefix, device)?;
        Ok(Self { input_norm, blocks, pool: [2, 2] })
    }

    pub fn forward(&self, input: Tensor<4>) -> RmvpeEncoderOutput {
        let mut x = match &self.input_norm { Some(norm) => norm.forward(input), None => input };
        for block in &self.blocks { x = block.forward(x); }
        let pooled = module::avg_pool2d(x.clone(), AvgPoolOptions::new(self.pool));
        RmvpeEncoderOutput { skip: x, pooled }
    }

    pub fn blocks(&self) -> usize { self.blocks.len() }
}

fn load_blocks(checkpoint: &RvcCheckpoint, prefix: &str, device: &Device) -> Result<Vec<ConvBlockRes>, String> {
    let mut blocks = Vec::new();
    for index in 0..32 {
        let name = format!("{prefix}.{index}");
        if !checkpoint.contains(&format!("{name}.conv.0.weight")) { break; }
        let block = ConvBlockRes::load(checkpoint, &name, device)?;
        if let Some(previous) = blocks.last() {
            let previous: &ConvBlockRes = previous;
            if previous.conv2.weight.dims()[0] != block.conv1.weight.dims()[1] {
                return Err(format!("{name}: input channels differ from previous residual block output"));
            }
        }
        blocks.push(block);
    }
    if blocks.is_empty() || checkpoint.contains(&format!("{prefix}.{}.conv.0.weight", blocks.len())) {
        return Err(format!("{prefix}: no blocks found or excessive block count"));
    }
    Ok(blocks)
}

pub struct RmvpeIntermediate {
    layers: Vec<Vec<ConvBlockRes>>,
}

impl RmvpeIntermediate {
    pub fn load(checkpoint: &RvcCheckpoint, levels: usize, device: &Device) -> Result<Self, String> {
        if !(1..=4).contains(&levels) {
            return Err("RMVPE U-Net Intermediate must have 1..=4 levels".into());
        }
        let mut layers = Vec::with_capacity(levels);
        for level in 0..levels {
            let prefix = format!("unet.intermediate.layers.{level}.conv");
            let blocks = load_blocks(checkpoint, &prefix, device)?;
            if let Some(previous) = layers.last() {
                let previous: &Vec<ConvBlockRes> = previous;
                if previous.last().unwrap().conv2.weight.dims()[0] != blocks[0].conv1.weight.dims()[1] {
                    return Err(format!("RMVPE Intermediate level {level}: channel discontinuity"));
                }
            }
            layers.push(blocks);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, mut input: Tensor<4>) -> Tensor<4> {
        for layer in &self.layers {
            for block in layer { input = block.forward(input); }
        }
        input
    }

    pub fn forward_with_levels(&self, mut input: Tensor<4>) -> Vec<Tensor<4>> {
        let mut trace = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            for block in layer { input = block.forward(input); }
            trace.push(input.clone());
        }
        trace
    }

    pub fn levels(&self) -> usize { self.layers.len() }
}
