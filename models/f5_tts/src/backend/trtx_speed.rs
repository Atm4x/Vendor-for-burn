//! Native TensorRT-RTX Cross-Lingual SpeedTransformer, including padding masks and
//! masked temporal pooling. Output is the 32-class speed distribution logits.

use trtx::{
    ActivationType, Axes, Builder, DataType, ElementWiseOperation as Ew, Logger, MatrixOperation,
    MemoryPoolType, NetworkDefinition, ReduceOperation, Tensor, UnaryOperation,
};

use super::{
    trtx::{BackboneShape, DeviceBuffer, TrtLinear},
    trtx_input::{TrtConv1d, conv1d, mish, norm_affine_eps},
    trtx_native::{binary, dims, fail, gelu_tanh, linear, permute, reshape, rope, scalar, unary},
};
use crate::{
    F5Error, Result,
    checkpoint::Checkpoint,
    cross_lingual::{SPEED_CLASSES, best_speed_class, speed_from_label},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrtSpeedShape {
    pub batch: usize,
    pub frames: usize,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub ff_mult: usize,
    pub position_groups: usize,
}

impl TrtSpeedShape {
    pub const CROSS_LINGUAL: Self = Self {
        batch: 1,
        frames: 256,
        dim: 512,
        depth: 6,
        heads: 8,
        ff_mult: 4,
        position_groups: 16,
    };

    pub fn validate(self) -> Result<()> {
        if self.batch == 0
            || self.batch > 16
            || self.frames == 0
            || self.frames > 8192
            || self.dim == 0
            || self.dim > 2048
            || self.depth == 0
            || self.depth > 32
            || self.heads == 0
            || self.dim % self.heads != 0
            || (self.dim / self.heads) % 2 != 0
            || self.ff_mult == 0
            || self.dim.checked_mul(self.ff_mult).is_none()
            || self.position_groups == 0
            || self.dim % self.position_groups != 0
            || self
                .batch
                .checked_mul(self.frames)
                .and_then(|x| x.checked_mul(100))
                .and_then(|x| x.checked_mul(4))
                .is_none()
        {
            return Err(F5Error::InvalidConfiguration(
                "Invalid TensorRT SpeedTransformer shape",
            ));
        }
        Ok(())
    }

    fn buffer_bytes(self) -> Result<[usize; 3]> {
        self.validate()?;
        let count = self
            .batch
            .checked_mul(self.frames)
            .ok_or(F5Error::InvalidShape("Speed input size overflow"))?;
        Ok([count * 100 * 4, count * 4, self.batch * SPEED_CLASSES * 4])
    }

    fn attention_shape(self) -> BackboneShape {
        BackboneShape {
            batch: self.batch,
            frames: self.frames,
            dim: self.dim,
            heads: self.heads,
            depth: self.depth,
            ff_mult: self.ff_mult,
            mel_channels: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrtSpeedBlock {
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub query: TrtLinear,
    pub key: TrtLinear,
    pub value: TrtLinear,
    pub attention_out: TrtLinear,
    pub ff1: TrtLinear,
    pub ff2: TrtLinear,
}

#[derive(Debug, Clone)]
pub struct TrtSpeedWeights {
    pub shape: TrtSpeedShape,
    pub mel_proj: TrtLinear,
    pub pos1: TrtConv1d,
    pub pos2: TrtConv1d,
    pub blocks: Vec<TrtSpeedBlock>,
    pub pool1: TrtLinear,
    pub pool2: TrtLinear,
    pub classifier_norm_weight: Vec<f32>,
    pub classifier_norm_bias: Vec<f32>,
    pub classifier1: TrtLinear,
    pub classifier2: TrtLinear,
}

impl TrtSpeedWeights {
    pub fn from_checkpoint(ckpt: &Checkpoint, shape: TrtSpeedShape) -> Result<Self> {
        shape.validate()?;
        let dim = shape.dim;
        let mut blocks = Vec::with_capacity(shape.depth);
        for i in 0..shape.depth {
            let prefix = format!("speed_transformer.transformer_blocks.{i}");
            blocks.push(TrtSpeedBlock {
                norm1_weight: ckpt.read_f32(&format!("{prefix}.ln1.weight"), &[dim])?,
                norm1_bias: ckpt.read_f32(&format!("{prefix}.ln1.bias"), &[dim])?,
                norm2_weight: ckpt.read_f32(&format!("{prefix}.ln2.weight"), &[dim])?,
                norm2_bias: ckpt.read_f32(&format!("{prefix}.ln2.bias"), &[dim])?,
                query: TrtLinear::from_checkpoint(ckpt, &format!("{prefix}.attn.to_q"), dim, dim)?,
                key: TrtLinear::from_checkpoint(ckpt, &format!("{prefix}.attn.to_k"), dim, dim)?,
                value: TrtLinear::from_checkpoint(ckpt, &format!("{prefix}.attn.to_v"), dim, dim)?,
                attention_out: TrtLinear::from_checkpoint(
                    ckpt,
                    &format!("{prefix}.attn.to_out.0"),
                    dim,
                    dim,
                )?,
                ff1: TrtLinear::from_checkpoint(
                    ckpt,
                    &format!("{prefix}.ff.ff.0.0"),
                    dim,
                    dim * shape.ff_mult,
                )?,
                ff2: TrtLinear::from_checkpoint(
                    ckpt,
                    &format!("{prefix}.ff.ff.2"),
                    dim * shape.ff_mult,
                    dim,
                )?,
            });
        }
        Ok(Self {
            shape,
            mel_proj: TrtLinear::from_checkpoint(ckpt, "speed_transformer.mel_proj", 100, dim)?,
            pos1: TrtConv1d::load(
                ckpt,
                "speed_transformer.conv_layer.conv1d.0",
                dim,
                dim,
                shape.position_groups,
                31,
            )?,
            pos2: TrtConv1d::load(
                ckpt,
                "speed_transformer.conv_layer.conv1d.2",
                dim,
                dim,
                shape.position_groups,
                31,
            )?,
            blocks,
            pool1: TrtLinear::from_checkpoint(ckpt, "speed_transformer.pool.0", dim, dim)?,
            pool2: TrtLinear::from_checkpoint(ckpt, "speed_transformer.pool.2", dim, 1)?,
            classifier_norm_weight: ckpt
                .read_f32("speed_transformer.classifier.0.weight", &[dim])?,
            classifier_norm_bias: ckpt.read_f32("speed_transformer.classifier.0.bias", &[dim])?,
            classifier1: TrtLinear::from_checkpoint(
                ckpt,
                "speed_transformer.classifier.1",
                dim,
                dim,
            )?,
            classifier2: TrtLinear::from_checkpoint(
                ckpt,
                "speed_transformer.classifier.3",
                dim,
                SPEED_CLASSES,
            )?,
        })
    }
}

pub struct TrtSpeedInputs {
    pub valid_frames: Vec<i32>,
}

impl TrtSpeedInputs {
    pub fn from_lengths(shape: TrtSpeedShape, lengths: &[usize]) -> Result<Self> {
        shape.validate()?;
        if lengths.len() != shape.batch || lengths.iter().any(|&l| l == 0 || l > shape.frames) {
            return Err(F5Error::InvalidShape(
                "Invalid SpeedTransformer reference lengths",
            ));
        }
        let valid_frames = lengths
            .iter()
            .flat_map(|&length| (0..shape.frames).map(move |frame| i32::from(frame < length)))
            .collect();
        Ok(Self { valid_frames })
    }
}

fn mask_float<'a>(
    net: &mut NetworkDefinition<'a>,
    valid: Tensor<'a>,
    shape: TrtSpeedShape,
) -> Result<Tensor<'a>> {
    let zero = net
        .add_constant_owned(&[1], vec![0; 4], DataType::kINT32, None)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let active = binary(net, valid, zero, Ew::kGREATER)?;
    let active = net
        .add_cast(&active, DataType::kFLOAT)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    reshape(net, active, &[dims(shape.batch), dims(shape.frames), 1])
}

fn masked<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    mask: Tensor<'a>,
) -> Result<Tensor<'a>> {
    binary(net, x, mask, Ew::kPROD)
}

fn layer_graph<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    mask: Tensor<'a>,
    block: &TrtSpeedBlock,
    shape: TrtSpeedShape,
) -> Result<Tensor<'a>> {
    let dim = shape.dim;
    let head_dim = dim / shape.heads;
    let norm = norm_affine_eps(net, x, &block.norm1_weight, &block.norm1_bias, 1e-6)?;
    let q = linear(net, norm, &block.query)?;
    let k = linear(net, norm, &block.key)?;
    let v = linear(net, norm, &block.value)?;
    let format = |net: &mut NetworkDefinition<'a>, x| -> Result<Tensor<'a>> {
        let x = reshape(
            net,
            x,
            &[
                dims(shape.batch),
                dims(shape.frames),
                dims(shape.heads),
                dims(head_dim),
            ],
        )?;
        permute(net, x, &[0, 2, 1, 3])
    };
    let q = format(net, q)?;
    let q = rope(net, q, shape.attention_shape())?;
    let k = format(net, k)?;
    let k = rope(net, k, shape.attention_shape())?;
    let v = format(net, v)?;
    let scores = net
        .add_matrix_multiply(&q, MatrixOperation::kNONE, &k, MatrixOperation::kTRANSPOSE)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let scale = scalar(net, 1.0 / (head_dim as f32).sqrt())?;
    let scores = binary(net, scores, scale, Ew::kPROD)?;
    let key_mask = reshape(net, mask, &[dims(shape.batch), 1, 1, dims(shape.frames)])?;
    let one = scalar(net, 1.0)?;
    let invalid = binary(net, one, key_mask, Ew::kSUB)?;
    let penalty = scalar(net, -f32::MAX)?;
    let invalid = binary(net, invalid, penalty, Ew::kPROD)?;
    let scores = binary(net, scores, invalid, Ew::kSUM)?;
    let probs = net
        .add_softmax(&scores, Axes::new([3]))
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let attention = net
        .add_matrix_multiply(&probs, MatrixOperation::kNONE, &v, MatrixOperation::kNONE)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let attention = permute(net, attention, &[0, 2, 1, 3])?;
    let attention = reshape(
        net,
        attention,
        &[dims(shape.batch), dims(shape.frames), dims(dim)],
    )?;
    let attention = linear(net, attention, &block.attention_out)?;
    let attention = masked(net, attention, mask)?;
    let x = binary(net, x, attention, Ew::kSUM)?;
    let norm = norm_affine_eps(net, x, &block.norm2_weight, &block.norm2_bias, 1e-6)?;
    let ff = linear(net, norm, &block.ff1)?;
    let ff = gelu_tanh(net, ff)?;
    let ff = linear(net, ff, &block.ff2)?;
    binary(net, x, ff, Ew::kSUM)
}

pub(super) fn speed_graph<'a>(
    net: &mut NetworkDefinition<'a>,
    mel: Tensor<'a>,
    valid: Tensor<'a>,
    weights: &TrtSpeedWeights,
) -> Result<Tensor<'a>> {
    let shape = weights.shape;
    let valid_mask = mask_float(net, valid, shape)?;
    let projected = linear(net, mel, &weights.mel_proj)?;
    let projected = masked(net, projected, valid_mask)?;
    let position1 = conv1d(net, projected, &weights.pos1, shape.batch, shape.frames)?;
    let position1 = mish(net, position1)?;
    let position1 = masked(net, position1, valid_mask)?;
    let position2 = conv1d(net, position1, &weights.pos2, shape.batch, shape.frames)?;
    let position2 = mish(net, position2)?;
    let mut x = masked(net, position2, valid_mask)?;
    for block in &weights.blocks {
        x = layer_graph(net, x, valid_mask, block, shape)?;
    }
    let pooled = linear(net, x, &weights.pool1)?;
    let pooled = unary(net, pooled, UnaryOperation::kTAN)?;
    let scores = linear(net, pooled, &weights.pool2)?;
    let one = scalar(net, 1.0)?;
    let invalid = binary(net, one, valid_mask, Ew::kSUB)?;
    let penalty = scalar(net, -f32::MAX)?;
    let penalty = binary(net, invalid, penalty, Ew::kPROD)?;
    let scores = binary(net, scores, penalty, Ew::kSUM)?;
    let pool_weights = net
        .add_softmax(&scores, Axes::new([1]))
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let weighted = binary(net, x, pool_weights, Ew::kPROD)?;
    let pooled = net
        .add_reduce(&weighted, ReduceOperation::kSUM, Axes::new([1]), true)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let pooled = norm_affine_eps(
        net,
        pooled,
        &weights.classifier_norm_weight,
        &weights.classifier_norm_bias,
        1e-5,
    )?;
    let pooled = linear(net, pooled, &weights.classifier1)?;
    let pooled = net
        .add_activation(&pooled, ActivationType::kGELU_ERF)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let logits = linear(net, pooled, &weights.classifier2)?;
    reshape(net, logits, &[dims(shape.batch), SPEED_CLASSES as i64])
}

pub fn build_serialized_speed(
    logger: &Logger,
    weights: &TrtSpeedWeights,
    workspace_bytes: usize,
) -> Result<Vec<u8>> {
    weights.shape.validate()?;
    if weights.blocks.len() != weights.shape.depth || workspace_bytes == 0 {
        return Err(F5Error::InvalidConfiguration(
            "Invalid TensorRT SpeedTransformer weights or workspace",
        ));
    }
    let mut builder = Builder::new(logger).map_err(fail)?;
    let mut net = builder
        .create_network(trtx::builder::network_flags::EXPLICIT_BATCH)
        .map_err(fail)?;
    let s = weights.shape;
    let mel = net
        .add_input(
            "mel",
            DataType::kFLOAT,
            &[dims(s.batch), dims(s.frames), 100],
        )
        .map_err(fail)?;
    let valid = net
        .add_input(
            "valid_frames",
            DataType::kINT32,
            &[dims(s.batch), dims(s.frames)],
        )
        .map_err(fail)?;
    let logits = speed_graph(&mut net, mel, valid, weights)?;
    logits.set_name(&mut net, "logits").map_err(fail)?;
    net.mark_output(&logits);
    let mut config = builder.create_config().map_err(fail)?;
    config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, workspace_bytes);
    let image = builder
        .build_serialized_network(&mut net, &mut config)
        .map_err(fail)?;
    if image.is_empty() {
        return Err(F5Error::ModelExecution(
            "TensorRT speed engine is empty".into(),
        ));
    }
    Ok(image.as_ref().to_vec())
}

#[derive(Debug, Clone, Copy)]
pub struct TrtSpeedBindings {
    pub mel: DeviceBuffer,
    pub valid_frames: DeviceBuffer,
    pub logits: DeviceBuffer,
}

impl TrtSpeedBindings {
    pub fn validate(self, shape: TrtSpeedShape) -> Result<()> {
        let [mel_bytes, mask_bytes, logits_bytes] = shape.buffer_bytes()?;
        self.mel.validate(mel_bytes)?;
        self.valid_frames.validate(mask_bytes)?;
        self.logits.validate(logits_bytes)?;
        let spans = [self.mel, self.valid_frames, self.logits];
        for (i, a) in spans.iter().enumerate() {
            let a_end = a.address.checked_add(a.bytes).ok_or(F5Error::InvalidShape(
                "SpeedTransformer device address overflow",
            ))?;
            for b in spans.iter().skip(i + 1) {
                let b_end = b.address.checked_add(b.bytes).ok_or(F5Error::InvalidShape(
                    "SpeedTransformer device address overflow",
                ))?;
                if a.address < b_end && b.address < a_end {
                    return Err(F5Error::InvalidShape("SpeedTransformer buffers overlap"));
                }
            }
        }
        Ok(())
    }
}

/// # Safety
/// Caller must ensure all buffers and the stream belong to the same CUDA device and
/// remain valid until stream synchronization, including on failed inference.
pub unsafe fn enqueue_speed(
    context: &mut trtx::ExecutionContext<'_>,
    shape: TrtSpeedShape,
    bindings: TrtSpeedBindings,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    bindings.validate(shape)?;
    if stream.is_null() {
        return Err(F5Error::InvalidShape(
            "Null CUDA stream for SpeedTransformer",
        ));
    }
    unsafe {
        context
            .set_input_tensor_address("mel", bindings.mel.address as *const _)
            .map_err(fail)?;
        context
            .set_input_tensor_address("valid_frames", bindings.valid_frames.address as *const _)
            .map_err(fail)?;
        context
            .set_tensor_address("logits", bindings.logits.address as *mut _)
            .map_err(fail)?;
        context.enqueue_v3(stream).map_err(fail)
    }
}

pub fn decode_predicted_speeds(logits: &[f32], batch: usize) -> Result<Vec<f32>> {
    if batch == 0
        || logits.len()
            != batch
                .checked_mul(SPEED_CLASSES)
                .ok_or(F5Error::InvalidShape(
                    "SpeedTransformer logits size overflow",
                ))?
    {
        return Err(F5Error::InvalidShape(
            "Incorrect SpeedTransformer logits count",
        ));
    }
    logits
        .chunks_exact(SPEED_CLASSES)
        .map(|row| speed_from_label(best_speed_class(row)?))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn shape() -> TrtSpeedShape {
        TrtSpeedShape {
            batch: 2,
            frames: 5,
            dim: 8,
            depth: 1,
            heads: 2,
            ff_mult: 2,
            position_groups: 2,
        }
    }

    #[test]
    fn loads_independent_pytorch_speed_weights() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/speed_predictor_small.safetensors");
        let ckpt = Checkpoint::open(path).unwrap();
        let model = TrtSpeedWeights::from_checkpoint(&ckpt, shape()).unwrap();
        assert_eq!(model.mel_proj.weights.len(), 100 * 8);
        assert_eq!(model.pos1.weights.len(), 8 * 4 * 31);
        assert_eq!(model.blocks.len(), 1);
        assert_eq!(model.blocks[0].ff1.weights.len(), 8 * 16);
        assert_eq!(model.classifier2.weights.len(), SPEED_CLASSES * 8);
    }

    #[test]
    fn masked_inputs_and_argmax_match_pytorch_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/speed_predictor_expected.json"
        ))
        .unwrap();
        let data = TrtSpeedInputs::from_lengths(shape(), &[5, 3]).unwrap();
        assert_eq!(data.valid_frames, [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
        let logits: Vec<f32> = fixture["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let expected: Vec<f32> = fixture["classes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| speed_from_label(x.as_u64().unwrap() as usize).unwrap())
            .collect();
        assert_eq!(decode_predicted_speeds(&logits, 2).unwrap(), expected);
        assert!(TrtSpeedInputs::from_lengths(shape(), &[5, 0]).is_err());
        assert!(TrtSpeedInputs::from_lengths(shape(), &[6, 3]).is_err());
    }

    #[test]
    fn validates_device_regions_and_shapes() {
        let mut b = TrtSpeedBindings {
            mel: DeviceBuffer {
                address: 0x1000,
                bytes: 4000,
            },
            valid_frames: DeviceBuffer {
                address: 0x5000,
                bytes: 40,
            },
            logits: DeviceBuffer {
                address: 0x6000,
                bytes: 256,
            },
        };
        b.validate(shape()).unwrap();
        b.logits.address = b.mel.address;
        assert!(b.validate(shape()).is_err());
        assert!(decode_predicted_speeds(&[0.0; 32], 2).is_err());
    }
}
