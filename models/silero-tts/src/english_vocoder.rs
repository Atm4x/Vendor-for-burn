use anyhow::{Result, ensure};
use burn::tensor::{Device, Tensor, activation::leaky_relu, module::{conv1d, conv_transpose1d}, ops::{ConvOptions, ConvTransposeOptions}};
use crate::store::TensorStore;
use std::time::Instant;

fn phase_end(start: Option<Instant>, stage: &str, index: Option<usize>) {
    if let Some(start) = start {
        eprintln!("silero_en_profile stage={stage} index={} ms={:.6}", index.map_or_else(|| "-".to_owned(), |i| i.to_string()), start.elapsed().as_secs_f64() * 1000.0);
    }
}

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

enum Activation { Leaky(f64), Snake { alpha: Tensor<3>, beta_inv: Tensor<3> } }
impl Activation {
    fn snake(store: &TensorStore, name: &str, channels: usize, device: &Device) -> Result<Self> {
        let a = store.float::<1>(&format!("{name}.alpha"), device)?;
        let b = store.float::<1>(&format!("{name}.beta"), device)?;
        ensure!(a.dims() == [channels] && b.dims() == [channels], "invalid SnakeBeta shape: {name}");
        Ok(Self::Snake { alpha: a.exp().reshape([1,channels,1]), beta_inv: (b.exp()+1e-9).recip().reshape([1,channels,1]) })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> { match self {
        Self::Leaky(slope) => leaky_relu(x, *slope),
        Self::Snake { alpha, beta_inv } => { let sine = (x.clone()*alpha.clone()).sin(); x + sine.clone()*sine*beta_inv.clone() }
    }}
}

struct Stage {
    activation: Activation,
    weight: Tensor<3>,
    bias: Tensor<1>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    blocks: ResidualStage,
}

impl Stage {
    fn load(store: &TensorStore, suffix: &str, index: usize, input: usize, output: usize, stride: usize, activation: Activation, device: &Device) -> Result<Self> {
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
        Ok(Self { activation, weight, bias, stride, padding: stride.div_ceil(2), output_padding: stride % 2, blocks })
    }
    fn forward(&self, x: Tensor<3>, profile: bool, index: usize) -> Tensor<3> {
        let start = profile.then(Instant::now);
        let x = conv_transpose1d(self.activation.forward(x), self.weight.clone(), Some(self.bias.clone()),
            ConvTransposeOptions::new([self.stride], [self.padding], [self.output_padding], [1], 1));
        phase_end(start, "upsample", Some(index));
        let start = profile.then(Instant::now);
        let x = self.blocks.forward(x);
        phase_end(start, "residual", Some(index));
        x
    }
}

struct RateGraph { pre: Conv, stages: Vec<Stage>, post: Conv, activation: Activation }
impl RateGraph {
    fn load(store: &TensorStore, rate: u32, v4: bool, device: &Device) -> Result<Self> {
        let (pre, suffix, channels, strides, post) = match rate {
            8000 => (0, "_8", &[256, 128, 64][..], &[5, 5][..], "conv_post"),
            24000 => (1, "", &[512, 256, 128, 64][..], &[3, 5, 5][..], "conv_post"),
            48000 => (1, "_u", &[512, 256, 128, 64, 64][..], &[3, 5, 5, 2][..], "conv_post_u"),
            _ => anyhow::bail!("Invalid English vocoder rate"),
        };
        let v4_channels: Vec<usize> = channels.iter().map(|c| c / 4).collect();
        let channels = if v4 { &v4_channels[..] } else { channels };
        let pre = Conv::load(store, &format!("vocoder.conv_pre.{pre}"), [128, channels[0]], 7, 1, device)?;
        let stages = strides.iter().enumerate().map(|(i, &stride)| {
            let activation = if v4 { Activation::snake(store, &format!("vocoder.activation_ups.{i}"), channels[i], device)? } else { Activation::Leaky(0.1) };
            Stage::load(store, suffix, i, channels[i], channels[i + 1], stride, activation, device)
        }).collect::<Result<_>>()?;
        let output = *channels.last().unwrap();
        let post = Conv::load(store, &format!("vocoder.{post}"), [output, 4], 7, 1, device)?;
        let activation = if v4 { Activation::snake(store, "vocoder.activation_post", output, device)? } else { Activation::Leaky(0.01) };
        Ok(Self { pre, stages, post, activation })
    }
    fn forward(&self, mel: Tensor<3>, profile: bool) -> Tensor<3> {
        let start = profile.then(Instant::now);
        let mut x = self.pre.forward(mel);
        phase_end(start, "pre", None);
        for (index, stage) in self.stages.iter().enumerate() { x = stage.forward(x, profile, index); }
        let start = profile.then(Instant::now);
        let x = self.post.forward(self.activation.forward(x));
        phase_end(start, "post", None);
        x
    }
}

pub struct EnglishVocoder {
    rates: [Option<RateGraph>; 3],
    downsample: Option<Tensor<3>>,
    synthesis: Tensor<3>,
    updown: Tensor<3>,
    #[cfg(feature = "reference-flex")]
    flex_pqmf: Option<crate::flex_english_pqmf::FlexEnglishPqmf>,
    profile: bool,
}
impl EnglishVocoder {
    pub fn load(store: &TensorStore, device: &Device) -> Result<Self> { Self::load_variant(store, false, device) }
    pub(crate) fn load_variant(store: &TensorStore, v4: bool, device: &Device) -> Result<Self> {
        let synthesis = store.float::<3>("vocoder.pqmf_layer.G", device)?;
        let updown = store.float::<3>("vocoder.pqmf_layer.updown_filter", device)?;
        ensure!(synthesis.dims() == [1, 4, 63] && updown.dims() == [4, 4, 4], "Invalid English PQMF shapes");
        let updown = updown * 4.0;
        #[cfg(feature = "reference-flex")]
        let flex_pqmf = if device == &Device::flex() && std::env::var_os("SILERO_EN_BASELINE_PQMF").is_none() {
            Some(crate::flex_english_pqmf::FlexEnglishPqmf::new(synthesis.clone(), updown.clone())?)
        } else { None };
        #[cfg(feature = "reference-flex")]
        let profile = device == &Device::flex() && std::env::var_os("SILERO_EN_PROFILE").is_some();
        #[cfg(not(feature = "reference-flex"))]
        let profile = false;
        let downsample = if v4 { let h = store.float::<3>("vocoder.pqmf_3.H", device)?; ensure!(h.dims()==[3,1,63], "invalid V4 PQMF analysis"); Some(h) } else { None };
        Ok(Self { rates: [if v4 { None } else { Some(RateGraph::load(store, 8000, false, device)?) }, Some(RateGraph::load(store, 24000, v4, device)?), Some(RateGraph::load(store, 48000, v4, device)?)], downsample, synthesis, updown, #[cfg(feature = "reference-flex")] flex_pqmf, profile })
    }
    pub(crate) fn profile_enabled(&self) -> bool { self.profile }

    fn burn_pqmf(&self, x: Tensor<3>) -> Tensor<3> {
        let x = conv_transpose1d(x, self.updown.clone(), None, ConvTransposeOptions::new([4], [0], [0], [1], 1));
        conv1d(x, self.synthesis.clone(), None, ConvOptions::new([1], [31], [1], 1))
    }

    pub fn forward(&self, mel: Tensor<3>, rate: u32) -> Result<Tensor<2>> {
        let [batch, channels, frames] = mel.dims();
        ensure!(channels == 128 && frames > 0, "English vocoder expects [batch, 128, frames]");
        let index = match rate { 8000 => 0, 24000 => 1, 48000 => 2, _ => anyhow::bail!("Unsupported English sample rate: {rate}") };
        if self.profile { eprintln!("silero_en_profile rate={rate} frames={frames} batch={batch}"); }
        let x = self.rates[index].as_ref().or(self.rates[1].as_ref()).expect("validated HiFi-GAN rate").forward(mel, self.profile);
        let start = self.profile.then(Instant::now);
        #[cfg(feature = "reference-flex")]
        let pcm = match &self.flex_pqmf { Some(fused) => fused.forward(x), None => self.burn_pqmf(x) };
        #[cfg(not(feature = "reference-flex"))]
        let pcm = self.burn_pqmf(x);
        phase_end(start, "pqmf", None);
        let start = self.profile.then(Instant::now);
        let limit: f32 = pcm.clone().abs().max().into_scalar();
        ensure!(limit.is_finite(), "Non-finite English PCM");
        let mut pcm = if limit > 1.0 { pcm.tanh() } else { pcm };
        if let Some(h) = self.downsample.as_ref().filter(|_| rate == 8000) {
            pcm = conv1d(pcm, h.clone(), None, ConvOptions::new([3], [31], [1], 1));
            let max: f32 = pcm.clone().abs().max().into_scalar();
            ensure!(max.is_finite(), "non-finite PQMF analysis");
            if max > 1.0 { pcm = pcm.tanh(); }
            let length = pcm.dims()[2];
            pcm = pcm.slice([0..batch, 0..1, 0..length]);
        }
        let length = frames * (rate as usize / 80);
        ensure!(pcm.dims() == [batch, 1, length], "Invalid English PCM length");
        phase_end(start, "limit", None);
        Ok(pcm.reshape([batch, length]))
    }
}
