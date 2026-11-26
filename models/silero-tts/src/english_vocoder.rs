use anyhow::{Result, ensure};
use burn::tensor::{Device, Tensor, activation::leaky_relu, module::{conv1d, conv_transpose1d}, ops::{ConvOptions, ConvTransposeOptions}};
use crate::store::TensorStore;

struct Conv {
    weight: Tensor<3>,
    bias: Tensor<1>,
    padding: usize,
    dilation: usize,
}

impl Conv {
    fn load(store: &TensorStore, name: &str, channels: [usize; 2], kernel: usize, dilation: usize, device: &Device) -> Result<Self> {
        let weight = store.float::<3>(&format!("{name}.weight"), device)?;
        let bias = store.float::<1>(&format!("{name}.bias"), device)?;
        ensure!(weight.dims() == [channels[1], channels[0], kernel] && bias.dims() == [channels[1]], "Invalid convolution shape: {name}");
        Ok(Self { weight, bias, padding: (kernel - 1) * dilation / 2, dilation })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        conv1d(x, self.weight.clone(), Some(self.bias.clone()), ConvOptions::new([1], [self.padding], [self.dilation], 1))
    }
}

struct ResidualBlock([Conv; 2]);
impl ResidualBlock {
    fn load(store: &TensorStore, name: &str, channels: usize, kernel: usize, device: &Device) -> Result<Self> {
        Ok(Self([Conv::load(store, &format!("{name}.convs.0"), [channels; 2], kernel, 1, device)?,
            Conv::load(store, &format!("{name}.convs.1"), [channels; 2], kernel, 3, device)?]))
    }
    fn forward(&self, mut x: Tensor<3>) -> Tensor<3> {
        for conv in &self.0 { x = conv.forward(leaky_relu(x.clone(), 0.1)) + x; }
        x
    }
}

enum ResidualStage {
    Burn([ResidualBlock; 3]),
    #[cfg(feature = "reference-flex")]
    Flex(crate::flex_english_conv::FlexEnglishResidualStage),
}
impl ResidualStage {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        match self {
            Self::Burn([a, b, c]) => (a.forward(x.clone()) + b.forward(x.clone()) + c.forward(x)) / 3.0,
            #[cfg(feature = "reference-flex")]
            Self::Flex(stage) => stage.forward(x),
        }
    }
}

struct Stage {
    weight: Tensor<3>,
    bias: Tensor<1>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    blocks: ResidualStage,
}

impl Stage {
    fn load(store: &TensorStore, suffix: &str, index: usize, input: usize, output: usize, stride: usize, device: &Device) -> Result<Self> {
        let name = format!("vocoder.ups{suffix}.{index}");
        let weight = store.float::<3>(&format!("{name}.weight"), device)?;
        let bias = store.float::<1>(&format!("{name}.bias"), device)?;
        ensure!(weight.dims() == [input, output, 2 * stride] && bias.dims() == [output], "Invalid upsampling shape: {name}");
        let block = |j, k| ResidualBlock::load(store, &format!("vocoder.resblocks{suffix}.{}", index * 3 + j), output, k, device);
        let blocks = [block(0, 3)?, block(1, 7)?, block(2, 11)?];
        #[cfg(feature = "reference-flex")]
        let blocks = if device == &Device::flex() && std::env::var_os("SILERO_EN_BASELINE_CONV").is_none() {
            let convert = |ResidualBlock([a, b]): ResidualBlock| -> Result<_> {
                use crate::flex_english_conv::FlexEnglishConv;
                Ok([
                    FlexEnglishConv::new(a.weight, a.bias, a.dilation)?,
                    FlexEnglishConv::new(b.weight, b.bias, b.dilation)?,
                ])
            };
            let [a, b, c] = blocks;
            ResidualStage::Flex(crate::flex_english_conv::FlexEnglishResidualStage([
                convert(a)?, convert(b)?, convert(c)?,
            ]))
        } else {
            ResidualStage::Burn(blocks)
        };
        #[cfg(not(feature = "reference-flex"))]
        let blocks = ResidualStage::Burn(blocks);
        Ok(Self { weight, bias, stride, padding: stride.div_ceil(2), output_padding: stride % 2, blocks })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = conv_transpose1d(leaky_relu(x, 0.1), self.weight.clone(), Some(self.bias.clone()),
            ConvTransposeOptions::new([self.stride], [self.padding], [self.output_padding], [1], 1));
        self.blocks.forward(x)
    }
}

struct RateGraph { pre: Conv, stages: Vec<Stage>, post: Conv }
impl RateGraph {
    fn load(store: &TensorStore, rate: u32, device: &Device) -> Result<Self> {
        let (pre, suffix, channels, strides, post) = match rate {
            8000 => (0, "_8", &[256, 128, 64][..], &[5, 5][..], "conv_post"),
            24000 => (1, "", &[512, 256, 128, 64][..], &[3, 5, 5][..], "conv_post"),
            48000 => (1, "_u", &[512, 256, 128, 64, 64][..], &[3, 5, 5, 2][..], "conv_post_u"),
            _ => anyhow::bail!("Invalid English vocoder rate"),
        };
        let pre = Conv::load(store, &format!("vocoder.conv_pre.{pre}"), [128, channels[0]], 7, 1, device)?;
        let stages = strides.iter().enumerate().map(|(i, &stride)| Stage::load(store, suffix, i, channels[i], channels[i + 1], stride, device)).collect::<Result<_>>()?;
        let post = Conv::load(store, &format!("vocoder.{post}"), [64, 4], 7, 1, device)?;
        Ok(Self { pre, stages, post })
    }
    fn forward(&self, mel: Tensor<3>) -> Tensor<3> {
        let mut x = self.pre.forward(mel);
        for stage in &self.stages { x = stage.forward(x); }
        self.post.forward(leaky_relu(x, 0.01))
    }
}

pub struct EnglishVocoder {
    rates: [RateGraph; 3],
    synthesis: Tensor<3>,
    updown: Tensor<3>,
}
impl EnglishVocoder {
    pub fn load(store: &TensorStore, device: &Device) -> Result<Self> {
        let synthesis = store.float::<3>("vocoder.pqmf_layer.G", device)?;
        let updown = store.float::<3>("vocoder.pqmf_layer.updown_filter", device)?;
        ensure!(synthesis.dims() == [1, 4, 63] && updown.dims() == [4, 4, 4], "Invalid English PQMF shapes");
        Ok(Self { rates: [RateGraph::load(store, 8000, device)?, RateGraph::load(store, 24000, device)?, RateGraph::load(store, 48000, device)?], synthesis, updown: updown * 4.0 })
    }
    pub fn forward(&self, mel: Tensor<3>, rate: u32) -> Result<Tensor<2>> {
        let [batch, channels, frames] = mel.dims();
        ensure!(channels == 128 && frames > 0, "English vocoder expects [batch, 128, frames]");
        let index = match rate { 8000 => 0, 24000 => 1, 48000 => 2, _ => anyhow::bail!("Unsupported English sample rate: {rate}") };
        let x = self.rates[index].forward(mel);
        let x = conv_transpose1d(x, self.updown.clone(), None, ConvTransposeOptions::new([4], [0], [0], [1], 1));
        let pcm = conv1d(x, self.synthesis.clone(), None, ConvOptions::new([1], [31], [1], 1));
        let limit: f32 = pcm.clone().abs().max().into_scalar();
        ensure!(limit.is_finite(), "Non-finite English PCM");
        let pcm = if limit > 1.0 { pcm.tanh() } else { pcm };
        let length = frames * (rate as usize / 80);
        ensure!(pcm.dims() == [batch, 1, length], "Invalid English PCM length");
        Ok(pcm.reshape([batch, length]))
    }
}
