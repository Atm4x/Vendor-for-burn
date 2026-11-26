use trtx::{
    ActivationType, Axes, Builder, DataType, ElementWiseOperation as Ew, Logger, MemoryPoolType,
    NetworkDefinition, ReduceOperation, Tensor, UnaryOperation,
};

use super::{
    trtx::{
        BackboneShape, DeviceBuffer, TrtAttentionPrecision, TrtBackboneWeights, TrtLinear,
        TrtTimeWeights,
    },
    trtx_native::{
        backbone_graph, binary, constant, dims, fail, linear, permute, reshape, scalar,
        time_mlp_graph, unary,
    },
};
use crate::{F5Error, Result, checkpoint::Checkpoint};

#[derive(Debug, Clone, Copy)]
pub struct TrtInputShape {
    pub backbone: BackboneShape,
    pub text_dim: usize,
    pub vocab_size: usize,
    pub text_blocks: usize,
}

impl TrtInputShape {
    pub fn validate(self) -> Result<()> {
        self.backbone.validate()?;
        if self.text_dim == 0
            || self.text_dim % 2 != 0
            || self.text_dim > 1024
            || self.vocab_size == 0
            || self.vocab_size > 100_000
            || self.text_blocks > 8
            || self.backbone.dim % 16 != 0
        {
            return Err(F5Error::InvalidConfiguration(
                "Invalid TensorRT F5 input shape",
            ));
        }
        Ok(())
    }

    pub fn v1(vocab_size: usize) -> Self {
        Self {
            backbone: BackboneShape::V1,
            text_dim: 512,
            vocab_size,
            text_blocks: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrtConv1d {
    pub input: usize,
    pub output: usize,
    pub groups: usize,
    pub kernel: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl TrtConv1d {
    pub(super) fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        input: usize,
        output: usize,
        groups: usize,
        kernel: usize,
    ) -> Result<Self> {
        if groups == 0
            || input % groups != 0
            || output % groups != 0
            || kernel == 0
            || kernel % 2 == 0
        {
            return Err(F5Error::InvalidConfiguration(
                "Invalid F5 grouped convolution",
            ));
        }
        Ok(Self {
            input,
            output,
            groups,
            kernel,
            weights: checkpoint.read_f32(
                &format!("{prefix}.weight"),
                &[output, input / groups, kernel],
            )?,
            bias: checkpoint.read_f32(&format!("{prefix}.bias"), &[output])?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TrtTextBlock {
    pub dwconv: TrtConv1d,
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub pw1: TrtLinear,
    pub pw2: TrtLinear,
    pub grn_gamma: Vec<f32>,
    pub grn_beta: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TrtInputWeights {
    pub shape: TrtInputShape,
    pub embedding: Vec<f32>,
    pub text_blocks: Vec<TrtTextBlock>,
    pub projection: TrtLinear,
    pub pos_conv1: TrtConv1d,
    pub pos_conv2: TrtConv1d,
}

impl TrtInputWeights {
    pub fn from_checkpoint(checkpoint: &Checkpoint, shape: TrtInputShape) -> Result<Self> {
        shape.validate()?;
        let t = shape.text_dim;
        let d = shape.backbone.dim;
        let c = shape.backbone.mel_channels;
        let mut blocks = Vec::with_capacity(shape.text_blocks);
        for n in 0..shape.text_blocks {
            let prefix = format!("transformer.text_embed.text_blocks.{n}");
            blocks.push(TrtTextBlock {
                dwconv: TrtConv1d::load(checkpoint, &format!("{prefix}.dwconv"), t, t, t, 7)?,
                norm_weight: checkpoint.read_f32(&format!("{prefix}.norm.weight"), &[t])?,
                norm_bias: checkpoint.read_f32(&format!("{prefix}.norm.bias"), &[t])?,
                pw1: TrtLinear::from_checkpoint(
                    checkpoint,
                    &format!("{prefix}.pwconv1"),
                    t,
                    2 * t,
                )?,
                pw2: TrtLinear::from_checkpoint(
                    checkpoint,
                    &format!("{prefix}.pwconv2"),
                    2 * t,
                    t,
                )?,
                grn_gamma: checkpoint.read_f32(&format!("{prefix}.grn.gamma"), &[1, 1, 2 * t])?,
                grn_beta: checkpoint.read_f32(&format!("{prefix}.grn.beta"), &[1, 1, 2 * t])?,
            });
        }
        Ok(Self {
            shape,
            embedding: checkpoint.read_f32(
                "transformer.text_embed.text_embed.weight",
                &[shape.vocab_size + 1, t],
            )?,
            text_blocks: blocks,
            projection: TrtLinear::from_checkpoint(
                checkpoint,
                "transformer.input_embed.proj",
                c * 2 + t,
                d,
            )?,
            pos_conv1: TrtConv1d::load(
                checkpoint,
                "transformer.input_embed.conv_pos_embed.conv1d.0",
                d,
                d,
                16,
                31,
            )?,
            pos_conv2: TrtConv1d::load(
                checkpoint,
                "transformer.input_embed.conv_pos_embed.conv1d.2",
                d,
                d,
                16,
                31,
            )?,
        })
    }
}

pub(super) fn mish<'a>(net: &mut NetworkDefinition<'a>, x: Tensor<'a>) -> Result<Tensor<'a>> {
    let softplus = net
        .add_activation(&x, ActivationType::kSOFTPLUS)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let tanh = super::trtx_native::tanh(net, softplus)?;
    binary(net, x, tanh, Ew::kPROD)
}

pub(super) fn conv1d<'a>(
    net: &mut NetworkDefinition<'a>,
    input: Tensor<'a>,
    w: &TrtConv1d,
    batch: usize,
    frames: usize,
) -> Result<Tensor<'a>> {
    let transposed = permute(net, input, &[0, 2, 1])?;
    let x = reshape(
        net,
        transposed,
        &[dims(batch), dims(w.input), dims(frames), 1],
    )?;
    let raw = w.weights.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bias_raw = w.bias.iter().flat_map(|v| v.to_le_bytes()).collect();
    let kernel = trtx::network::OwnedWeights {
        shape: vec![dims(w.output), dims(w.input / w.groups), dims(w.kernel), 1],
        data_type: DataType::kFLOAT,
        values: raw,
        name: None,
    };
    let bias = trtx::network::OwnedWeights {
        shape: vec![dims(w.output)],
        data_type: DataType::kFLOAT,
        values: bias_raw,
        name: None,
    };
    let mut layer = net
        .add_convolution_owned_weights(
            &x,
            w.output as i32,
            &[w.kernel as i32, 1],
            trtx::network::OwnedConvWeights {
                kernel,
                bias: Some(bias),
            },
        )
        .map_err(fail)?;
    layer.set_num_groups(net, w.groups as i64);
    layer.set_padding(net, &[(w.kernel / 2) as i64, 0]);
    let output = layer.output(net, 0).map_err(fail)?;
    let output = reshape(net, output, &[dims(batch), dims(w.output), dims(frames)])?;
    permute(net, output, &[0, 2, 1])
}

fn mask_pad<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    valid: Tensor<'a>,
    batch: usize,
    frames: usize,
) -> Result<Tensor<'a>> {
    let valid = reshape(net, valid, &[dims(batch), dims(frames), 1])?;
    let valid = net
        .add_cast(&valid, DataType::kFLOAT)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    binary(net, x, valid, Ew::kPROD)
}

pub(super) fn norm_affine<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    weight: &[f32],
    bias: &[f32],
) -> Result<Tensor<'a>> {
    norm_affine_eps(net, x, weight, bias, 1e-6)
}

pub(super) fn norm_affine_eps<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Result<Tensor<'a>> {
    let mean = net
        .add_reduce(&x, ReduceOperation::kAVG, Axes::new([2]), true)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let centered = binary(net, x, mean, Ew::kSUB)?;
    let squared = binary(net, centered, centered, Ew::kPROD)?;
    let var = net
        .add_reduce(&squared, ReduceOperation::kAVG, Axes::new([2]), true)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let eps = scalar(net, epsilon)?;
    let var = binary(net, var, eps, Ew::kSUM)?;
    let root = unary(net, var, UnaryOperation::kSQRT)?;
    let x = binary(net, centered, root, Ew::kDIV)?;
    let gamma = constant(net, &[1, 1, dims(weight.len())], weight)?;
    let beta = constant(net, &[1, 1, dims(bias.len())], bias)?;
    let x = binary(net, x, gamma, Ew::kPROD)?;
    binary(net, x, beta, Ew::kSUM)
}

fn convnext<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    w: &TrtTextBlock,
    batch: usize,
    frames: usize,
) -> Result<Tensor<'a>> {
    let residual = x;
    let depth = conv1d(net, x, &w.dwconv, batch, frames)?;
    let x = norm_affine(net, depth, &w.norm_weight, &w.norm_bias)?;
    let x = linear(net, x, &w.pw1)?;
    let x = net
        .add_activation(&x, ActivationType::kGELU_ERF)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let sq = binary(net, x, x, Ew::kPROD)?;
    let sum = net
        .add_reduce(&sq, ReduceOperation::kSUM, Axes::new([1]), true)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let gx = unary(net, sum, UnaryOperation::kSQRT)?;
    let mean = net
        .add_reduce(&gx, ReduceOperation::kAVG, Axes::new([2]), true)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let eps = scalar(net, 1e-6)?;
    let denominator = binary(net, mean, eps, Ew::kSUM)?;
    let gx = binary(net, gx, denominator, Ew::kDIV)?;
    let scaled = binary(net, x, gx, Ew::kPROD)?;
    let gamma = constant(net, &[1, 1, dims(w.grn_gamma.len())], &w.grn_gamma)?;
    let beta = constant(net, &[1, 1, dims(w.grn_beta.len())], &w.grn_beta)?;
    let scaled = binary(net, scaled, gamma, Ew::kPROD)?;
    let scaled = binary(net, scaled, beta, Ew::kSUM)?;
    let x = binary(net, x, scaled, Ew::kSUM)?;
    let x = linear(net, x, &w.pw2)?;
    binary(net, residual, x, Ew::kSUM)
}

pub(super) fn input_graph<'a>(
    net: &mut NetworkDefinition<'a>,
    input: Tensor<'a>,
    condition: Tensor<'a>,
    tokens: Tensor<'a>,
    text_mask: Tensor<'a>,
    weights: &TrtInputWeights,
) -> Result<Tensor<'a>> {
    let shape = weights.shape;
    let batch = shape.backbone.batch;
    let frames = shape.backbone.frames;
    let t = shape.text_dim;
    let embed = constant(
        net,
        &[dims(shape.vocab_size + 1), dims(t)],
        &weights.embedding,
    )?;
    let index = net
        .add_gather(&embed, &tokens, 0)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let positions: Vec<f32> = (0..frames)
        .flat_map(|position| {
            (0..t).map(move |channel| {
                let half = t / 2;
                let angle = position as f32
                    * (-(10_000.0_f32.ln()) * ((channel % half) * 2) as f32 / t as f32).exp();
                if channel < half {
                    angle.cos()
                } else {
                    angle.sin()
                }
            })
        })
        .collect();
    let pos = constant(net, &[1, dims(frames), dims(t)], &positions)?;
    let text = binary(net, index, pos, Ew::kSUM)?;
    let zero_int = net
        .add_constant_owned(&[1], vec![0, 0, 0, 0], DataType::kINT32, None)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let valid = binary(net, text_mask, zero_int, Ew::kGREATER)?;
    let mut text = mask_pad(net, text, valid, batch, frames)?;
    for block in &weights.text_blocks {
        text = convnext(net, text, block, batch, frames)?;
        text = mask_pad(net, text, valid, batch, frames)?;
    }
    let mut concat = net
        .add_concatenation(&[&input, &condition, &text])
        .map_err(fail)?;
    concat.set_axis(net, 2);
    let joined = concat.output(net, 0).map_err(fail)?;
    let projected = linear(net, joined, &weights.projection)?;
    let position = conv1d(net, projected, &weights.pos_conv1, batch, frames)?;
    let position = mish(net, position)?;
    let position = conv1d(net, position, &weights.pos_conv2, batch, frames)?;
    let position = mish(net, position)?;
    binary(net, projected, position, Ew::kSUM)
}

#[derive(Debug, Clone)]
pub struct TrtCfgInputs {
    pub noisy: Vec<f32>,
    pub condition: Vec<f32>,
    pub tokens: Vec<i32>,
    pub text_mask: Vec<i32>,
}

impl TrtCfgInputs {
    pub fn from_original_tokens(
        shape: TrtInputShape,
        source_token_frames: usize,
        noisy: &[f32],
        condition: &[f32],
        original_tokens: &[i32],
    ) -> Result<Self> {
        shape.validate()?;
        let s = shape.backbone;
        if s.batch % 2 != 0 || source_token_frames == 0 {
            return Err(F5Error::InvalidConfiguration(
                "TensorRT CFG batch must contain paired conditional and unconditional rows",
            ));
        }
        let source_batch = s.batch / 2;
        let single_mel_len = source_batch
            .checked_mul(s.frames)
            .and_then(|v| v.checked_mul(s.mel_channels))
            .ok_or(F5Error::InvalidShape("Invalid CFG mel allocation length"))?;
        let single_token_len = source_batch
            .checked_mul(source_token_frames)
            .ok_or(F5Error::InvalidShape("Invalid CFG token allocation length"))?;
        if noisy.len() != single_mel_len
            || condition.len() != single_mel_len
            || original_tokens.len() != single_token_len
        {
            return Err(F5Error::InvalidShape(
                "CFG host tensor lengths do not match fixed-shape engine",
            ));
        }
        if noisy.iter().chain(condition.iter()).any(|v| !v.is_finite()) {
            return Err(F5Error::InvalidConfiguration("Nonfinite CFG mel input"));
        }
        let mut tokens = vec![0; s.batch * s.frames];
        let mut mask = vec![0; s.batch * s.frames];
        for b in 0..source_batch {
            for frame in 0..source_token_frames.min(s.frames) {
                let token = original_tokens[b * source_token_frames + frame];
                if token < -1 || token >= shape.vocab_size as i32 {
                    return Err(F5Error::InvalidShape(
                        "CFG token index is outside model vocabulary",
                    ));
                }
                if token >= 0 {
                    let offset = b * s.frames + frame;
                    tokens[offset] = token + 1;
                    mask[offset] = 1;
                    mask[offset + source_batch * s.frames] = 1;
                }
            }
        }
        let mut noisy_double = Vec::with_capacity(single_mel_len * 2);
        noisy_double.extend_from_slice(noisy);
        noisy_double.extend_from_slice(noisy);
        let mut condition_double = vec![0.0; single_mel_len * 2];
        condition_double[..single_mel_len].copy_from_slice(condition);
        Ok(Self {
            noisy: noisy_double,
            condition: condition_double,
            tokens,
            text_mask: mask,
        })
    }
}

pub fn build_serialized_full_dit(
    logger: &Logger,
    input_weights: &TrtInputWeights,
    time_weights: &TrtTimeWeights,
    backbone: &TrtBackboneWeights,
    workspace_bytes: usize,
) -> Result<Vec<u8>> {
    let shape = input_weights.shape;
    shape.validate()?;
    if shape.backbone != backbone.shape
        || workspace_bytes == 0
        || time_weights.batch != shape.backbone.batch
        || time_weights.dim != shape.backbone.dim
    {
        return Err(F5Error::InvalidConfiguration(
            "TensorRT DiT input/backbone shape mismatch",
        ));
    }
    let b = dims(shape.backbone.batch);
    let f = dims(shape.backbone.frames);
    let c = dims(shape.backbone.mel_channels);
    let mut builder = Builder::new(logger).map_err(fail)?;
    let mut net = builder
        .create_network(trtx::builder::network_flags::EXPLICIT_BATCH)
        .map_err(fail)?;
    let noisy = net
        .add_input("noisy", DataType::kFLOAT, &[b, f, c])
        .map_err(fail)?;
    let reference = net
        .add_input("condition", DataType::kFLOAT, &[b, f, c])
        .map_err(fail)?;
    let tokens = net
        .add_input("tokens", DataType::kINT32, &[b, f])
        .map_err(fail)?;
    let text_mask = net
        .add_input("text_mask", DataType::kINT32, &[b, f])
        .map_err(fail)?;
    let sincos = net
        .add_input("sincos", DataType::kFLOAT, &[b, 256])
        .map_err(fail)?;
    let time = time_mlp_graph(&mut net, sincos, time_weights)?;
    let hidden = input_graph(&mut net, noisy, reference, tokens, text_mask, input_weights)?;
    let result = backbone_graph(
        &mut net,
        hidden,
        time,
        backbone,
        TrtAttentionPrecision::Fp32,
    )?;
    result.set_name(&mut net, "output").map_err(fail)?;
    net.mark_output(&result);
    let mut config = builder.create_config().map_err(fail)?;
    config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, workspace_bytes);
    let image = builder
        .build_serialized_network(&mut net, &mut config)
        .map_err(fail)?;
    if image.is_empty() {
        return Err(F5Error::ModelExecution(
            "TensorRT F5 complete DiT engine is empty".into(),
        ));
    }
    Ok(image.as_ref().to_vec())
}

#[derive(Debug, Clone, Copy)]
pub struct FullDiTBindings {
    pub noisy: DeviceBuffer,
    pub condition: DeviceBuffer,
    pub tokens: DeviceBuffer,
    pub text_mask: DeviceBuffer,
    pub sincos: DeviceBuffer,
    pub output: DeviceBuffer,
}

impl FullDiTBindings {
    pub fn validate(self, shape: TrtInputShape) -> Result<()> {
        shape.validate()?;
        let s = shape.backbone;
        let mel_bytes = s
            .batch
            .checked_mul(s.frames)
            .and_then(|v| v.checked_mul(s.mel_channels))
            .and_then(|v| v.checked_mul(4))
            .ok_or(F5Error::InvalidShape("F5 mel buffer overflow"))?;
        let token_bytes = s
            .batch
            .checked_mul(s.frames)
            .and_then(|v| v.checked_mul(4))
            .ok_or(F5Error::InvalidShape("F5 token buffer overflow"))?;
        let t_bytes = s
            .batch
            .checked_mul(256)
            .and_then(|v| v.checked_mul(4))
            .ok_or(F5Error::InvalidShape("F5 timestep buffer overflow"))?;
        for (buffer, size) in [
            (self.noisy, mel_bytes),
            (self.condition, mel_bytes),
            (self.tokens, token_bytes),
            (self.text_mask, token_bytes),
            (self.sincos, t_bytes),
            (self.output, mel_bytes),
        ] {
            buffer.validate(size)?;
        }
        let out_end = self
            .output
            .address
            .checked_add(self.output.bytes)
            .ok_or(F5Error::InvalidShape("Output address overflow"))?;
        for input in [
            self.noisy,
            self.condition,
            self.tokens,
            self.text_mask,
            self.sincos,
        ] {
            let in_end = input
                .address
                .checked_add(input.bytes)
                .ok_or(F5Error::InvalidShape("Input address overflow"))?;
            if self.output.address < in_end && input.address < out_end {
                return Err(F5Error::InvalidShape(
                    "TensorRT full DiT output overlaps input",
                ));
            }
        }
        Ok(())
    }
}

/// # Safety
/// All buffers must reside on the stream's CUDA device, remain allocated until synchronization,
/// and must meet the engine's fixed tensor shapes. Token indices are pre-shifted: pad=0,
/// vocabulary IDs are 1..=vocab_size, and the unconditional CFG row uses all zeros.
/// `text_mask` must preserve the original conditional text-valid positions in both CFG rows.
pub unsafe fn enqueue_full_dit(
    context: &mut trtx::ExecutionContext<'_>,
    shape: TrtInputShape,
    buffers: FullDiTBindings,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    buffers.validate(shape)?;
    if stream.is_null() {
        return Err(F5Error::InvalidShape("Null CUDA stream"));
    }
    unsafe {
        context
            .set_input_tensor_address("noisy", buffers.noisy.address as *const _)
            .map_err(fail)?;
        context
            .set_input_tensor_address("condition", buffers.condition.address as *const _)
            .map_err(fail)?;
        context
            .set_input_tensor_address("tokens", buffers.tokens.address as *const _)
            .map_err(fail)?;
        context
            .set_input_tensor_address("text_mask", buffers.text_mask.address as *const _)
            .map_err(fail)?;
        context
            .set_input_tensor_address("sincos", buffers.sincos.address as *const _)
            .map_err(fail)?;
        context
            .set_tensor_address("output", buffers.output.address as *mut _)
            .map_err(fail)?;
        context.enqueue_v3(stream).map_err(fail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixture() -> (Checkpoint, TrtInputShape) {
        let file =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dit_full_small.safetensors");
        let shape = TrtInputShape {
            backbone: BackboneShape {
                batch: 2,
                frames: 5,
                dim: 16,
                heads: 4,
                depth: 2,
                ff_mult: 2,
                mel_channels: 2,
            },
            text_dim: 8,
            vocab_size: 8,
            text_blocks: 1,
        };
        (Checkpoint::open(file).unwrap(), shape)
    }

    #[test]
    fn cfg_input_host_layout_matches_independent_torch() {
        let expected: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/trtx_cfg_host_pytorch_expected.json"
        ))
        .unwrap();
        fn floats(data: &serde_json::Value, key: &str) -> Vec<f32> {
            data[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        }
        fn ints(data: &serde_json::Value, key: &str) -> Vec<i32> {
            data[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect()
        }
        let shape = TrtInputShape {
            backbone: BackboneShape {
                batch: 4,
                frames: 5,
                dim: 16,
                heads: 4,
                depth: 2,
                ff_mult: 2,
                mel_channels: 2,
            },
            text_dim: 8,
            vocab_size: 8,
            text_blocks: 1,
        };
        let packed = TrtCfgInputs::from_original_tokens(
            shape,
            expected["token_frames"].as_u64().unwrap() as usize,
            &floats(&expected, "source_noisy"),
            &floats(&expected, "source_condition"),
            &ints(&expected, "source_tokens"),
        )
        .unwrap();
        assert_eq!(packed.noisy, floats(&expected, "expected_noisy"));
        assert_eq!(packed.condition, floats(&expected, "expected_condition"));
        assert_eq!(packed.tokens, ints(&expected, "expected_tokens"));
        assert_eq!(packed.text_mask, ints(&expected, "expected_text_mask"));
    }

    #[test]
    fn frontend_fixture_checkpoint_maps_exact_parameters() {
        let (checkpoint, shape) = fixture();
        let w = TrtInputWeights::from_checkpoint(&checkpoint, shape).unwrap();
        assert_eq!(w.embedding.len(), 72);
        assert_eq!(w.text_blocks.len(), 1);
        assert_eq!(w.text_blocks[0].dwconv.weights.len(), 56);
        assert_eq!(w.text_blocks[0].grn_gamma.len(), 16);
        assert_eq!(w.projection.weights.len(), 192);
        assert_eq!(w.pos_conv1.weights.len(), 16 * 31);
    }

    #[test]
    fn cfg_packing_preserves_original_padding_mask_in_unconditional_row() {
        let (_, shape) = fixture();
        let mut noisy = vec![0.0; 5 * 2];
        for (i, val) in noisy.iter_mut().enumerate() {
            *val = i as f32 * 0.1;
        }
        let ref_mel = vec![0.5; 5 * 2];
        let packed =
            TrtCfgInputs::from_original_tokens(shape, 3, &noisy, &ref_mel, &[2, 4, -1]).unwrap();
        assert_eq!(packed.noisy[..10], packed.noisy[10..]);
        assert_eq!(&packed.condition[..10], &ref_mel);
        assert!(packed.condition[10..].iter().all(|v| *v == 0.0));
        assert_eq!(packed.tokens, [3, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(packed.text_mask, [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]);
        assert!(
            TrtCfgInputs::from_original_tokens(shape, 3, &noisy, &ref_mel, &[2, 9, -1]).is_err()
        );
        assert!(
            TrtCfgInputs::from_original_tokens(shape, 3, &noisy, &ref_mel, &[2, -2, -1]).is_err()
        );
    }

    #[test]
    fn time_mlp_matches_fused_engine_shape_and_checkpoint() {
        let (checkpoint, shape) = fixture();
        let time =
            TrtTimeWeights::from_checkpoint(&checkpoint, shape.backbone.batch, shape.backbone.dim)
                .unwrap();
        assert_eq!(time.input.input, 256);
        assert_eq!(time.output.output, 16);
        assert_eq!(time.batch, shape.backbone.batch);
    }

    #[test]
    fn full_dit_rejects_invalid_shapes_and_buffer_overlaps() {
        let (_, shape) = fixture();
        let mel = 2 * 5 * 2 * 4;
        let tokens = 2 * 5 * 4;
        let t = 2 * 256 * 4;
        let good = FullDiTBindings {
            noisy: DeviceBuffer {
                address: 0x1000,
                bytes: mel,
            },
            condition: DeviceBuffer {
                address: 0x2000,
                bytes: mel,
            },
            tokens: DeviceBuffer {
                address: 0x3000,
                bytes: tokens,
            },
            text_mask: DeviceBuffer {
                address: 0x3500,
                bytes: tokens,
            },
            sincos: DeviceBuffer {
                address: 0x4000,
                bytes: t,
            },
            output: DeviceBuffer {
                address: 0x5000,
                bytes: mel,
            },
        };
        good.validate(shape).unwrap();
        let mut bad = good;
        bad.tokens.bytes -= 4;
        assert!(bad.validate(shape).is_err());
        bad = good;
        bad.output.address = 0x1008;
        assert!(bad.validate(shape).is_err());
    }
}
