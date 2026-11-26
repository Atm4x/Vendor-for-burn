//! Silero V5 ConvNeXt vocoder as native Burn tensor operations.

#[cfg(feature = "reference-flex")]
use crate::flex_pointwise::apply_fast_gelu;
use crate::store::TensorStore;
use anyhow::Result;
use burn::tensor::Device;
#[cfg(feature = "reference-flex")]
use burn::tensor::TensorData;
use burn::tensor::{
    Tensor,
    activation::gelu,
    module::{conv1d, layer_norm, linear},
    ops::ConvOptions,
};

fn vocoder_gelu(x: Tensor<3>) -> Tensor<3> {
    #[cfg(feature = "reference-flex")]
    if x.device() == Device::flex() && std::env::var_os("SILERO_EXACT_GELU").is_none() {
        let dims = x.dims();
        let device = x.device();
        let mut data = x
            .into_data()
            .try_into_vec::<f32>()
            .expect("Flex vocoder GELU requires f32 data");
        apply_fast_gelu(&mut data, std::env::var_os("SILERO_SERIAL_GELU").is_none());
        return Tensor::from_data(TensorData::new(data, dims), &device);
    }
    gelu(x)
}

enum ConvNeXtBlock {
    Standard {
        preprocess: ConvNeXtPreprocess,
        pointwise: PointwiseEngine,
    },
    #[cfg(feature = "reference-flex")]
    Native(crate::flex_convnext::FlexConvNeXt),
}

enum DepthwiseEngine {
    Tensor {
        weight: Tensor<3>,
        bias: Tensor<1>,
    },
    #[cfg(feature = "reference-flex")]
    Flex(crate::flex_depthwise::FlexDepthwise),
}

impl DepthwiseEngine {
    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        match self {
            Self::Tensor { weight, bias } => {
                let channels = weight.dims()[0];
                conv1d(
                    input,
                    weight.clone(),
                    Some(bias.clone()),
                    ConvOptions::new([1], [3], [1], channels),
                )
            }
            #[cfg(feature = "reference-flex")]
            Self::Flex(native) => native.forward(input),
        }
    }
}

enum ConvNeXtPreprocess {
    Separate {
        depthwise: DepthwiseEngine,
        norm_weight: Tensor<1>,
        norm_bias: Tensor<1>,
    },
    #[cfg(feature = "reference-flex")]
    Flex(crate::flex_depthwise::FlexDepthwiseNorm),
}

impl ConvNeXtPreprocess {
    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let timed = std::env::var_os("SILERO_VOCODER_TIMING").is_some();
        let start = std::time::Instant::now();
        match self {
            Self::Separate {
                depthwise,
                norm_weight,
                norm_bias,
            } => {
                let x = depthwise.forward(input);
                if timed {
                    eprintln!(
                        "[vocoder-block] depthwise_ms={:.3}",
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let start = std::time::Instant::now();
                let x = layer_norm(
                    x.swap_dims(1, 2),
                    norm_weight.clone(),
                    Some(norm_bias.clone()),
                    1e-6,
                );
                if timed {
                    eprintln!(
                        "[vocoder-block] norm_ms={:.3}",
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                x
            }
            #[cfg(feature = "reference-flex")]
            Self::Flex(native) => {
                let x = native.forward(input);
                if timed {
                    eprintln!(
                        "[vocoder-block] fused_depthwise_norm_ms={:.3}",
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                x
            }
        }
    }
}

enum PointwiseEngine {
    Tensor {
        pw1_weight: Tensor<2>,
        pw1_bias: Tensor<1>,
        pw2_weight: Tensor<2>,
        pw2_bias: Tensor<1>,
        gamma: Tensor<1>,
    },
    #[cfg(feature = "reference-flex")]
    Flex(crate::flex_pointwise::FlexPointwise),
}

impl PointwiseEngine {
    fn forward(&self, x: Tensor<3>, residual: Tensor<3>) -> Tensor<3> {
        match self {
            Self::Tensor {
                pw1_weight,
                pw1_bias,
                pw2_weight,
                pw2_bias,
                gamma,
            } => {
                let timed = std::env::var_os("SILERO_VOCODER_TIMING").is_some();
                let start = std::time::Instant::now();
                let channels = gamma.dims()[0];
                let x = linear(x, pw1_weight.clone(), Some(pw1_bias.clone()));
                if timed {
                    eprintln!(
                        "[vocoder-block] pwconv1_ms={:.3}",
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let start = std::time::Instant::now();
                let x = vocoder_gelu(x);
                if timed {
                    eprintln!(
                        "[vocoder-block] gelu_ms={:.3}",
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let start = std::time::Instant::now();
                let x = linear(x, pw2_weight.clone(), Some(pw2_bias.clone()));
                if timed {
                    eprintln!(
                        "[vocoder-block] pwconv2_ms={:.3}",
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                (x * gamma.clone().reshape([1, 1, channels])).swap_dims(1, 2) + residual
            }
            #[cfg(feature = "reference-flex")]
            Self::Flex(native) => native.forward(x, residual),
        }
    }
}

impl ConvNeXtBlock {
    fn load(weights: &TensorStore, id: usize, device: &Device) -> Result<Self> {
        let p = format!("vocoder.backbone.convnext.{id}");
        let pw1_weight = weights
            .float::<2>(&format!("{p}.pwconv1.weight"), device)?
            .transpose();
        let pw1_bias = weights.float(&format!("{p}.pwconv1.bias"), device)?;
        let pw2_weight = weights
            .float::<2>(&format!("{p}.pwconv2.weight"), device)?
            .transpose();
        let pw2_bias = weights.float(&format!("{p}.pwconv2.bias"), device)?;
        let gamma = weights.float(&format!("{p}.gamma"), device)?;
        let dwconv_weight = weights.float(&format!("{p}.dwconv.weight"), device)?;
        let dwconv_bias = weights.float(&format!("{p}.dwconv.bias"), device)?;
        let norm_weight = weights.float(&format!("{p}.norm.weight"), device)?;
        let norm_bias = weights.float(&format!("{p}.norm.bias"), device)?;
        #[cfg(feature = "reference-flex")]
        if device == &Device::flex()
            && std::env::var_os("SILERO_FLEX_WHOLE_CONVNEXT").is_some()
            && std::env::var_os("SILERO_FLEX_TENSOR_DEPTHWISE").is_none()
            && std::env::var_os("SILERO_FLEX_TENSOR_POINTWISE").is_none()
        {
            return Ok(Self::Native(crate::flex_convnext::FlexConvNeXt::new(
                crate::flex_depthwise::FlexDepthwiseNorm::new(
                    dwconv_weight,
                    dwconv_bias,
                    norm_weight,
                    norm_bias,
                )?,
                crate::flex_pointwise::FlexPointwise::new(
                    pw1_weight, pw1_bias, pw2_weight, pw2_bias, gamma,
                )?,
            )));
        }
        #[cfg(feature = "reference-flex")]
        let use_native =
            device == &Device::flex() && std::env::var_os("SILERO_FLEX_TENSOR_POINTWISE").is_none();
        #[cfg(not(feature = "reference-flex"))]
        let use_native = false;
        #[cfg(feature = "reference-flex")]
        let pointwise = if use_native {
            PointwiseEngine::Flex(crate::flex_pointwise::FlexPointwise::new(
                pw1_weight, pw1_bias, pw2_weight, pw2_bias, gamma,
            )?)
        } else {
            PointwiseEngine::Tensor {
                pw1_weight,
                pw1_bias,
                pw2_weight,
                pw2_bias,
                gamma,
            }
        };
        #[cfg(not(feature = "reference-flex"))]
        let pointwise = {
            let _ = use_native;
            PointwiseEngine::Tensor {
                pw1_weight,
                pw1_bias,
                pw2_weight,
                pw2_bias,
                gamma,
            }
        };
        #[cfg(feature = "reference-flex")]
        let fused = device == &Device::flex()
            && std::env::var_os("SILERO_FLEX_FUSED_DEPTHWISE_NORM").is_some()
            && std::env::var_os("SILERO_FLEX_TENSOR_DEPTHWISE").is_none();
        #[cfg(feature = "reference-flex")]
        let preprocess = if fused {
            ConvNeXtPreprocess::Flex(crate::flex_depthwise::FlexDepthwiseNorm::new(
                dwconv_weight,
                dwconv_bias,
                norm_weight,
                norm_bias,
            )?)
        } else {
            let depthwise = if device == &Device::flex()
                && std::env::var_os("SILERO_FLEX_TENSOR_DEPTHWISE").is_none()
            {
                DepthwiseEngine::Flex(crate::flex_depthwise::FlexDepthwise::new(
                    dwconv_weight,
                    dwconv_bias,
                )?)
            } else {
                DepthwiseEngine::Tensor {
                    weight: dwconv_weight,
                    bias: dwconv_bias,
                }
            };
            ConvNeXtPreprocess::Separate {
                depthwise,
                norm_weight,
                norm_bias,
            }
        };
        #[cfg(not(feature = "reference-flex"))]
        let preprocess = ConvNeXtPreprocess::Separate {
            depthwise: DepthwiseEngine::Tensor {
                weight: dwconv_weight,
                bias: dwconv_bias,
            },
            norm_weight,
            norm_bias,
        };
        Ok(Self::Standard {
            preprocess,
            pointwise,
        })
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        match self {
            Self::Standard {
                preprocess,
                pointwise,
            } => {
                let x = preprocess.forward(input.clone());
                pointwise.forward(x, input)
            }
            #[cfg(feature = "reference-flex")]
            Self::Native(native) => native.forward(input),
        }
    }
}

pub struct Vocoder {
    embed_weight: Tensor<3>,
    embed_bias: Tensor<1>,
    norm_weight: Tensor<1>,
    norm_bias: Tensor<1>,
    blocks: Vec<ConvNeXtBlock>,
    final_norm_weight: Tensor<1>,
    final_norm_bias: Tensor<1>,
    head_weight: Tensor<2>,
    head_bias: Tensor<1>,
}

impl Vocoder {
    pub fn load(weights: &TensorStore, device: &Device) -> Result<Self> {
        let p = "vocoder.backbone";
        let blocks = (0..8)
            .map(|id| ConvNeXtBlock::load(weights, id, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_weight: weights.float(&format!("{p}.embed.weight"), device)?,
            embed_bias: weights.float(&format!("{p}.embed.bias"), device)?,
            norm_weight: weights.float(&format!("{p}.norm.weight"), device)?,
            norm_bias: weights.float(&format!("{p}.norm.bias"), device)?,
            blocks,
            final_norm_weight: weights.float(&format!("{p}.final_layer_norm.weight"), device)?,
            final_norm_bias: weights.float(&format!("{p}.final_layer_norm.bias"), device)?,
            head_weight: weights
                .float::<2>("vocoder.head.out.weight", device)?
                .transpose(),
            head_bias: weights.float("vocoder.head.out.bias", device)?,
        })
    }

    /// Returns `(magnitude, cos(phase), sin(phase))` in `[B, 1201, T]`.
    /// This is the complete JitV4 backbone and head, before Silero's ISTFT/PQMF.
    pub fn forward(&self, mel: Tensor<3>) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let x = conv1d(
            mel,
            self.embed_weight.clone(),
            Some(self.embed_bias.clone()),
            ConvOptions::new([1], [3], [1], 1),
        );
        let mut x = layer_norm(
            x.swap_dims(1, 2),
            self.norm_weight.clone(),
            Some(self.norm_bias.clone()),
            1e-6,
        )
        .swap_dims(1, 2);
        #[cfg(feature = "reference-flex")]
        let x = if self
            .blocks
            .iter()
            .all(|block| matches!(block, ConvNeXtBlock::Native(_)))
        {
            let [batch, channels, frames] = x.dims();
            let device = x.device();
            let mut samples = x
                .into_data()
                .try_to_vec::<f32>()
                .expect("Flex ConvNeXt input must be f32");
            for block in &self.blocks {
                let ConvNeXtBlock::Native(native) = block else {
                    unreachable!()
                };
                samples = native.forward_host(&samples, batch, channels, frames);
            }
            Tensor::from_data(TensorData::new(samples, [batch, channels, frames]), &device)
        } else {
            for block in &self.blocks {
                x = block.forward(x);
            }
            x
        };
        #[cfg(not(feature = "reference-flex"))]
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = layer_norm(
            x.swap_dims(1, 2),
            self.final_norm_weight.clone(),
            Some(self.final_norm_bias.clone()),
            1e-6,
        );
        let timed = std::env::var_os("SILERO_VOCODER_TIMING").is_some();
        let start = std::time::Instant::now();
        let spectrum = linear(x, self.head_weight.clone(), Some(self.head_bias.clone()));
        if timed {
            eprintln!(
                "[vocoder-head] linear_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let start = std::time::Instant::now();
        #[cfg(feature = "reference-flex")]
        if spectrum.device() == Device::flex()
            && std::env::var_os("SILERO_FLEX_TENSOR_SPECTRAL").is_none()
        {
            let result = crate::flex_spectral::split_spectrum(spectrum);
            if timed {
                eprintln!(
                    "[vocoder-head] spectral_ms={:.3}",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
            return result;
        }
        let spectrum = spectrum.swap_dims(1, 2);
        let [batch, channels, frames] = spectrum.dims();
        let bins = channels / 2;
        let mag = spectrum
            .clone()
            .slice([0..batch, 0..bins, 0..frames])
            .exp()
            .clamp_max(100.0);
        let phase = spectrum.slice([0..batch, bins..channels, 0..frames]);
        let result = (mag, phase.clone().cos(), phase.sin());
        if timed {
            eprintln!(
                "[vocoder-head] spectral_ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        result
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn actual_silero_v5_spectral_head_parity() {
        let Ok(dir) = std::env::var("SILERO_PARITY_DIR") else {
            return;
        };
        let device = Device::flex();
        let weights = TensorStore::open(&Path::new(&dir).join("tts.safetensors")).unwrap();
        let reference = TensorStore::open(&Path::new(&dir).join("parity.safetensors")).unwrap();
        let model = Vocoder::load(&weights, &device).unwrap();
        let (mag, c, s) = model.forward(reference.float("mel", &device).unwrap());
        for (name, prediction) in [("mag", mag), ("cos", c), ("sin", s)] {
            let expected = reference
                .float::<3>(name, &device)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let got = prediction.into_data().try_to_vec::<f32>().unwrap();
            let max_abs = got
                .iter()
                .zip(expected.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            println!("{name}: max_abs={max_abs:.9}");
            assert!(max_abs < 1.0e-3, "{name} max_abs={max_abs:.9}");
        }
    }
}

#[cfg(all(test, feature = "reference-flex"))]
mod cpu_optimization_tests {
    use super::*;
    use crate::flex_pointwise::fast_gelu_scalar;

    #[test]
    fn parallel_gelu_matches_serial_bitwise() {
        let input: Vec<f32> = (0..131_072)
            .map(|i| ((i % 20_001) as f32 - 10_000.) * 0.001)
            .collect();
        let mut serial = input.clone();
        let mut parallel = input;
        apply_fast_gelu(&mut serial, false);
        apply_fast_gelu(&mut parallel, true);
        assert_eq!(serial.len(), parallel.len());
        for (a, b) in serial.into_iter().zip(parallel) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn fast_gelu_matches_exact_erf() {
        let device = Device::flex();
        let inputs = (-2000..=2000).map(|i| i as f32 * 0.005).collect::<Vec<_>>();
        let exact = gelu(Tensor::<3>::from_data(
            TensorData::new(inputs.clone(), [1, inputs.len(), 1]),
            &device,
        ))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
        for (x, expected) in inputs.into_iter().zip(exact) {
            let actual = fast_gelu_scalar(x);
            assert!(
                (actual - expected).abs() < 2e-6,
                "GELU({x}) mismatch: {actual} != {expected}"
            );
        }
    }
}
