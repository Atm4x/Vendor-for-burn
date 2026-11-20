//! Model-specific Vocos spectral decoder on TensorRT-RTX.
//! The independent Rust ISTFT receives the spectral head; CubeCL is never invoked.

use trtx::{
    ActivationType, Axes, Builder, DataType, ElementWiseOperation as Ew, Logger, MemoryPoolType,
    NetworkDefinition, ReduceOperation, Tensor, UnaryOperation,
};

use super::{
    trtx::{DeviceBuffer, TrtLinear},
    trtx_input::{TrtConv1d, conv1d, norm_affine},
    trtx_native::{binary, constant, dims, fail, linear, scalar, unary},
};
use crate::{F5Error, Result, checkpoint::Checkpoint, fft::IStft};

const SPECTRAL_BINS: usize = 513;
const HEAD_CHANNELS: usize = SPECTRAL_BINS * 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrtVocosShape {
    pub frames: usize,
    pub mel_channels: usize,
    pub dim: usize,
    pub intermediate_dim: usize,
    pub layers: usize,
}

impl TrtVocosShape {
    pub const V1: Self = Self {
        frames: 256,
        mel_channels: 100,
        dim: 512,
        intermediate_dim: 1536,
        layers: 8,
    };

    pub fn validate(self) -> Result<()> {
        if self.frames == 0
            || self.frames > 65_536
            || self.mel_channels != 100
            || self.dim == 0
            || self.intermediate_dim == 0
            || self.layers == 0
            || self.layers > 32
            || self
                .frames
                .checked_mul(HEAD_CHANNELS)
                .and_then(|x| x.checked_mul(4))
                .is_none()
            || self
                .frames
                .checked_mul(self.dim)
                .and_then(|x| x.checked_mul(4))
                .is_none()
        {
            return Err(F5Error::InvalidConfiguration(
                "Invalid TensorRT Vocos shape",
            ));
        }
        Ok(())
    }

    pub fn mel_bytes(self) -> Result<usize> {
        self.validate()?;
        self.frames
            .checked_mul(self.mel_channels)
            .and_then(|x| x.checked_mul(size_of::<f32>()))
            .ok_or(F5Error::InvalidShape("Vocos mel buffer overflow"))
    }

    pub fn head_bytes(self) -> Result<usize> {
        self.validate()?;
        self.frames
            .checked_mul(HEAD_CHANNELS)
            .and_then(|x| x.checked_mul(size_of::<f32>()))
            .ok_or(F5Error::InvalidShape("Vocos spectral buffer overflow"))
    }
}

#[derive(Debug, Clone)]
pub struct TrtVocosBlock {
    pub depthwise: TrtConv1d,
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub expand: TrtLinear,
    pub grn_gamma: Vec<f32>,
    pub grn_beta: Vec<f32>,
    pub project: TrtLinear,
}

#[derive(Debug, Clone)]
pub struct TrtVocosWeights {
    pub shape: TrtVocosShape,
    pub embed: TrtConv1d,
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub blocks: Vec<TrtVocosBlock>,
    pub final_norm_weight: Vec<f32>,
    pub final_norm_bias: Vec<f32>,
    pub head: TrtLinear,
}

impl TrtVocosWeights {
    pub fn from_checkpoint(ckpt: &Checkpoint, shape: TrtVocosShape) -> Result<Self> {
        shape.validate()?;
        let dim = shape.dim;
        let hidden = shape.intermediate_dim;
        let embed = TrtConv1d::load(ckpt, "backbone.embed", 100, dim, 1, 7)?;
        let norm_weight = ckpt.read_f32("backbone.norm.weight", &[dim])?;
        let norm_bias = ckpt.read_f32("backbone.norm.bias", &[dim])?;
        let mut blocks = Vec::with_capacity(shape.layers);
        for index in 0..shape.layers {
            let prefix = format!("backbone.convnext.{index}");
            blocks.push(TrtVocosBlock {
                depthwise: TrtConv1d::load(ckpt, &format!("{prefix}.dwconv"), dim, dim, dim, 7)?,
                norm_weight: ckpt.read_f32(&format!("{prefix}.norm.weight"), &[dim])?,
                norm_bias: ckpt.read_f32(&format!("{prefix}.norm.bias"), &[dim])?,
                expand: TrtLinear::from_checkpoint(
                    ckpt,
                    &format!("{prefix}.pwconv1"),
                    dim,
                    hidden,
                )?,
                grn_gamma: ckpt.read_f32(&format!("{prefix}.grn.gamma"), &[1, 1, hidden])?,
                grn_beta: ckpt.read_f32(&format!("{prefix}.grn.beta"), &[1, 1, hidden])?,
                project: TrtLinear::from_checkpoint(
                    ckpt,
                    &format!("{prefix}.pwconv2"),
                    hidden,
                    dim,
                )?,
            });
        }
        Ok(Self {
            shape,
            embed,
            norm_weight,
            norm_bias,
            blocks,
            final_norm_weight: ckpt.read_f32("backbone.final_layer_norm.weight", &[dim])?,
            final_norm_bias: ckpt.read_f32("backbone.final_layer_norm.bias", &[dim])?,
            head: TrtLinear::from_checkpoint(ckpt, "head.out", dim, HEAD_CHANNELS)?,
        })
    }
}

fn block_graph<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    block: &TrtVocosBlock,
    frames: usize,
) -> Result<Tensor<'a>> {
    let residual = x;
    let depthwise = conv1d(net, x, &block.depthwise, 1, frames)?;
    let normalized = norm_affine(net, depthwise, &block.norm_weight, &block.norm_bias)?;
    let expanded = linear(net, normalized, &block.expand)?;
    let expanded = net
        .add_activation(&expanded, ActivationType::kGELU_ERF)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;

    let squares = binary(net, expanded, expanded, Ew::kPROD)?;
    let sum = net
        .add_reduce(&squares, ReduceOperation::kSUM, Axes::new([1]), true)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let response = unary(net, sum, UnaryOperation::kSQRT)?;
    let average = net
        .add_reduce(&response, ReduceOperation::kAVG, Axes::new([2]), true)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let epsilon = scalar(net, 1e-6)?;
    let denominator = binary(net, average, epsilon, Ew::kSUM)?;
    let gain = binary(net, response, denominator, Ew::kDIV)?;
    let transformed = binary(net, expanded, gain, Ew::kPROD)?;
    let gamma = constant(net, &[1, 1, dims(block.grn_gamma.len())], &block.grn_gamma)?;
    let beta = constant(net, &[1, 1, dims(block.grn_beta.len())], &block.grn_beta)?;
    let transformed = binary(net, transformed, gamma, Ew::kPROD)?;
    let transformed = binary(net, transformed, beta, Ew::kSUM)?;
    let transformed = binary(net, expanded, transformed, Ew::kSUM)?;
    let projected = linear(net, transformed, &block.project)?;
    binary(net, residual, projected, Ew::kSUM)
}

pub(super) fn vocos_graph<'a>(
    net: &mut NetworkDefinition<'a>,
    mel: Tensor<'a>,
    weights: &TrtVocosWeights,
) -> Result<Tensor<'a>> {
    let frames = weights.shape.frames;
    let embed = conv1d(net, mel, &weights.embed, 1, frames)?;
    let mut x = norm_affine(net, embed, &weights.norm_weight, &weights.norm_bias)?;
    for block in &weights.blocks {
        x = block_graph(net, x, block, frames)?;
    }
    let x = norm_affine(net, x, &weights.final_norm_weight, &weights.final_norm_bias)?;
    linear(net, x, &weights.head)
}

pub fn build_serialized_vocos(
    logger: &Logger,
    weights: &TrtVocosWeights,
    workspace_bytes: usize,
) -> Result<Vec<u8>> {
    let shape = weights.shape;
    shape.validate()?;
    if weights.blocks.len() != shape.layers || workspace_bytes == 0 {
        return Err(F5Error::InvalidConfiguration(
            "Invalid TensorRT Vocos weights or workspace",
        ));
    }
    let mut builder = Builder::new(logger).map_err(fail)?;
    let mut net = builder
        .create_network(trtx::builder::network_flags::EXPLICIT_BATCH)
        .map_err(fail)?;
    let mel = net
        .add_input(
            "mel",
            DataType::kFLOAT,
            &[1, dims(shape.frames), dims(shape.mel_channels)],
        )
        .map_err(fail)?;
    let head = vocos_graph(&mut net, mel, weights)?;
    head.set_name(&mut net, "spectral_head").map_err(fail)?;
    net.mark_output(&head);
    let mut config = builder.create_config().map_err(fail)?;
    config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, workspace_bytes);
    let image = builder
        .build_serialized_network(&mut net, &mut config)
        .map_err(fail)?;
    if image.is_empty() {
        return Err(F5Error::ModelExecution(
            "TensorRT Vocos engine is empty".into(),
        ));
    }
    Ok(image.as_ref().to_vec())
}

#[derive(Debug, Clone, Copy)]
pub struct TrtVocosBindings {
    pub mel: DeviceBuffer,
    pub spectral_head: DeviceBuffer,
}

impl TrtVocosBindings {
    pub fn validate(self, shape: TrtVocosShape) -> Result<()> {
        self.mel.validate(shape.mel_bytes()?)?;
        self.spectral_head.validate(shape.head_bytes()?)?;
        let mel_end = self
            .mel
            .address
            .checked_add(self.mel.bytes)
            .ok_or(F5Error::InvalidShape("Vocos mel address overflow"))?;
        let head_end = self
            .spectral_head
            .address
            .checked_add(self.spectral_head.bytes)
            .ok_or(F5Error::InvalidShape("Vocos head address overflow"))?;
        if self.mel.address < head_end && self.spectral_head.address < mel_end {
            return Err(F5Error::InvalidShape("Vocos input/output buffers overlap"));
        }
        Ok(())
    }
}

/// # Safety
/// Both buffers must be allocated on the stream's CUDA device, and remain live until the
/// stream is synchronized. The caller owns engine/context lifecycle and allocations.
pub unsafe fn enqueue_vocos(
    context: &mut trtx::ExecutionContext<'_>,
    shape: TrtVocosShape,
    bindings: TrtVocosBindings,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    bindings.validate(shape)?;
    if stream.is_null() {
        return Err(F5Error::InvalidShape("Null CUDA stream for Vocos"));
    }
    unsafe {
        context
            .set_input_tensor_address("mel", bindings.mel.address as *const _)
            .map_err(fail)?;
        context
            .set_tensor_address("spectral_head", bindings.spectral_head.address as *mut _)
            .map_err(fail)?;
        context.enqueue_v3(stream).map_err(fail)
    }
}

/// Performs the model-specific inverse STFT on TensorRT's downloaded spectral head.
/// The returned mono samples are 24 kHz, before PCM16 quantization.
pub fn decode_vocos_spectrum(spectral_head: &[f32], restoration_gain: f32) -> Result<Vec<f32>> {
    if spectral_head.is_empty()
        || spectral_head.len() % HEAD_CHANNELS != 0
        || !restoration_gain.is_finite()
        || restoration_gain < 0.0
    {
        return Err(F5Error::InvalidShape(
            "Invalid TensorRT Vocos spectral head or gain",
        ));
    }
    let frames = spectral_head.len() / HEAD_CHANNELS;
    if frames > 65_536 {
        return Err(F5Error::InvalidShape(
            "Vocos spectral head exceeds frame limit",
        ));
    }
    let mut real = Vec::with_capacity(frames * SPECTRAL_BINS);
    let mut imag = Vec::with_capacity(frames * SPECTRAL_BINS);
    for frame in spectral_head.chunks_exact(HEAD_CHANNELS) {
        for i in 0..SPECTRAL_BINS {
            let magnitude = frame[i].exp();
            let phase = frame[i + SPECTRAL_BINS];
            if !magnitude.is_finite() || !phase.is_finite() {
                return Err(F5Error::ModelExecution(
                    "Nonfinite TensorRT Vocos spectrum".into(),
                ));
            }
            real.push(magnitude * phase.cos());
            imag.push(magnitude * phase.sin());
        }
    }
    let mut pcm = IStft::new().reconstruct(&real, &imag)?;
    for sample in &mut pcm {
        *sample *= restoration_gain;
    }
    Ok(pcm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/vocos_decoder_small.safetensors")
    }

    fn test_shape() -> TrtVocosShape {
        TrtVocosShape {
            frames: 9,
            mel_channels: 100,
            dim: 8,
            intermediate_dim: 16,
            layers: 2,
        }
    }

    #[test]
    fn loads_exact_pytorch_vocos_weight_layout() {
        let ckpt = Checkpoint::open(fixture()).unwrap();
        let weights = TrtVocosWeights::from_checkpoint(&ckpt, test_shape()).unwrap();
        assert_eq!(weights.embed.weights.len(), 8 * 100 * 7);
        assert_eq!(weights.blocks.len(), 2);
        assert_eq!(weights.blocks[0].depthwise.weights.len(), 8 * 7);
        assert_eq!(weights.blocks[1].grn_gamma.len(), 16);
        assert_eq!(weights.head.weights.len(), 1026 * 8);
        assert_eq!(weights.head.bias.len(), 1026);
    }

    #[test]
    fn spectral_head_istft_matches_independent_pytorch_waveform() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/vocos_decoder_expected.json"
        ))
        .unwrap();
        let vec = |name: &str| -> Vec<f32> {
            fixture[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };
        let wave = decode_vocos_spectrum(&vec("head"), 1.0).unwrap();
        let expected = vec("wave");
        assert_eq!(wave.len(), expected.len());
        let max_error = wave
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 2e-4, "Vocos ISTFT max error {max_error}");
    }

    #[test]
    fn buffer_shapes_overflow_and_aliasing_are_rejected() {
        assert_eq!(test_shape().mel_bytes().unwrap(), 3600);
        assert_eq!(test_shape().head_bytes().unwrap(), 9 * 1026 * 4);
        let mut bindings = TrtVocosBindings {
            mel: DeviceBuffer {
                address: 0x1000,
                bytes: 3600,
            },
            spectral_head: DeviceBuffer {
                address: 0x10000,
                bytes: 9 * 1026 * 4,
            },
        };
        bindings.validate(test_shape()).unwrap();
        bindings.spectral_head.address = 0x1000;
        assert!(bindings.validate(test_shape()).is_err());
        bindings.spectral_head.address = 0x10000;
        bindings.mel.bytes = 100;
        assert!(bindings.validate(test_shape()).is_err());
        let mut invalid = test_shape();
        invalid.frames = usize::MAX;
        assert!(invalid.validate().is_err());
        assert!(decode_vocos_spectrum(&[f32::NAN; 1026], 1.0).is_err());
        assert!(decode_vocos_spectrum(&[0.0; 1026], f32::NAN).is_err());
    }
}
