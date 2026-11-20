//! Direct TensorRT-RTX network construction for the F5 DiT v1 backbone.
//! No ONNX import, shared engine graph, CubeCL memory, or Python runtime.

use trtx::{
    ActivationType, Axes, Builder, DataType, ElementWiseOperation as Ew, Logger, MatrixOperation,
    MemoryPoolType, NetworkDefinition, ReduceOperation, Tensor, UnaryOperation,
};

use super::trtx::{
    BackboneBindings, BackboneShape, DeviceBuffer, TrtBackboneWeights, TrtLinear, TrtTimeWeights,
};
use crate::{F5Error, Result};

pub(super) fn fail(error: trtx::Error) -> F5Error {
    F5Error::ModelExecution(error.to_string())
}

pub(super) fn dims(n: usize) -> i64 {
    n as i64
}

pub(super) fn scalar<'a>(net: &mut NetworkDefinition<'a>, value: f32) -> Result<Tensor<'a>> {
    net.add_small_constant_copied(&[1], &value.to_le_bytes(), DataType::kFLOAT, None)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)
}

pub(super) fn constant<'a>(
    net: &mut NetworkDefinition<'a>,
    shape: &[i64],
    values: &[f32],
) -> Result<Tensor<'a>> {
    let raw = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>();
    net.add_constant_owned(shape, raw, DataType::kFLOAT, None)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)
}

pub(super) fn binary<'a>(
    net: &mut NetworkDefinition<'a>,
    a: Tensor<'a>,
    b: Tensor<'a>,
    op: Ew,
) -> Result<Tensor<'a>> {
    net.add_elementwise(&a, &b, op)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)
}

pub(super) fn unary<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    op: UnaryOperation,
) -> Result<Tensor<'a>> {
    net.add_unary(&x, op)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)
}

pub(super) fn reshape<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    shape: &[i64],
) -> Result<Tensor<'a>> {
    let mut layer = net.add_shuffle(&x).map_err(fail)?;
    layer.set_reshape_dimensions(net, shape).map_err(fail)?;
    layer.output(net, 0).map_err(fail)
}

pub(super) fn permute<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    order: &[i32],
) -> Result<Tensor<'a>> {
    let mut layer = net.add_shuffle(&x).map_err(fail)?;
    layer.set_first_transpose(net, order).map_err(fail)?;
    layer.output(net, 0).map_err(fail)
}

fn segment<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    batch: usize,
    width: usize,
    part: usize,
) -> Result<Tensor<'a>> {
    let start = [0, dims(part * width)];
    let size = [dims(batch), dims(width)];
    net.add_slice(&x, &start, &size, &[1, 1])
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)
}

pub(super) fn linear<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    weight: &TrtLinear,
) -> Result<Tensor<'a>> {
    let shape = x.dimensions(net).map_err(fail)?;
    if shape.last() != Some(&dims(weight.input)) {
        return Err(F5Error::InvalidShape(
            "TensorRT F5 linear input width mismatch",
        ));
    }
    let matrix = constant(
        net,
        &[dims(weight.output), dims(weight.input)],
        &weight.weights,
    )?;
    let matrix = if shape.len() == 3 {
        reshape(net, matrix, &[1, dims(weight.output), dims(weight.input)])?
    } else {
        matrix
    };
    let multiplied = net
        .add_matrix_multiply(
            &x,
            MatrixOperation::kNONE,
            &matrix,
            MatrixOperation::kTRANSPOSE,
        )
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let bias = constant(net, &[dims(weight.output)], &weight.bias)?;
    binary(net, multiplied, bias, Ew::kSUM)
}

fn silu<'a>(net: &mut NetworkDefinition<'a>, x: Tensor<'a>) -> Result<Tensor<'a>> {
    let sigmoid = net
        .add_activation(&x, ActivationType::kSIGMOID)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    binary(net, x, sigmoid, Ew::kPROD)
}

pub(super) fn gelu_tanh<'a>(net: &mut NetworkDefinition<'a>, x: Tensor<'a>) -> Result<Tensor<'a>> {
    let x2 = binary(net, x, x, Ew::kPROD)?;
    let x3 = binary(net, x2, x, Ew::kPROD)?;
    let c = scalar(net, 0.044_715)?;
    let term = binary(net, x3, c, Ew::kPROD)?;
    let sum = binary(net, x, term, Ew::kSUM)?;
    let scale = scalar(net, (2.0_f32 / core::f32::consts::PI).sqrt())?;
    let inner = binary(net, sum, scale, Ew::kPROD)?;
    let tanh = unary(net, inner, UnaryOperation::kTAN)?;
    let one = scalar(net, 1.0)?;
    let plus_one = binary(net, tanh, one, Ew::kSUM)?;
    let half = scalar(net, 0.5)?;
    let half_tanh = binary(net, plus_one, half, Ew::kPROD)?;
    binary(net, x, half_tanh, Ew::kPROD)
}

fn layer_norm<'a>(net: &mut NetworkDefinition<'a>, x: Tensor<'a>) -> Result<Tensor<'a>> {
    let mean = net
        .add_reduce(&x, ReduceOperation::kAVG, Axes::new([2]), true)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let centered = binary(net, x, mean, Ew::kSUB)?;
    let squared = binary(net, centered, centered, Ew::kPROD)?;
    let variance = net
        .add_reduce(&squared, ReduceOperation::kAVG, Axes::new([2]), true)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let epsilon = scalar(net, 1.0e-6)?;
    let variance = binary(net, variance, epsilon, Ew::kSUM)?;
    let root = unary(net, variance, UnaryOperation::kSQRT)?;
    binary(net, centered, root, Ew::kDIV)
}

pub(super) fn rope<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    shape: BackboneShape,
) -> Result<Tensor<'a>> {
    let head_dim = shape.dim / shape.heads;
    let half = head_dim / 2;
    let mut cosine = Vec::with_capacity(shape.frames * head_dim);
    let mut sine = Vec::with_capacity(shape.frames * head_dim);
    for t in 0..shape.frames {
        for j in 0..head_dim {
            let freq = (-(10_000.0_f32.ln()) * ((j % half) as f32) * 2.0 / (head_dim as f32)).exp();
            let phase = (t as f32) * freq;
            cosine.push(phase.cos());
            sine.push(phase.sin());
        }
    }
    let coeff_shape = [1, 1, dims(shape.frames), dims(head_dim)];
    let c = constant(net, &coeff_shape, &cosine)?;
    let s = constant(net, &coeff_shape, &sine)?;
    let start = [0, 0, 0, 0];
    let size = [
        dims(shape.batch),
        dims(shape.heads),
        dims(shape.frames),
        dims(half),
    ];
    let x1 = net
        .add_slice(&x, &start, &size, &[1, 1, 1, 1])
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let start2 = [0, 0, 0, dims(half)];
    let x2 = net
        .add_slice(&x, &start2, &size, &[1, 1, 1, 1])
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let minus_x2 = unary(net, x2, UnaryOperation::kNEG)?;
    let mut concat = net.add_concatenation(&[&minus_x2, &x1]).map_err(fail)?;
    concat.set_axis(net, 3);
    let rotated = concat.output(net, 0).map_err(fail)?;
    let first = binary(net, x, c, Ew::kPROD)?;
    let second = binary(net, rotated, s, Ew::kPROD)?;
    binary(net, first, second, Ew::kSUM)
}

fn attention<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    shape: BackboneShape,
    q_weight: &TrtLinear,
    k_weight: &TrtLinear,
    v_weight: &TrtLinear,
    out_weight: &TrtLinear,
) -> Result<Tensor<'a>> {
    let h = shape.heads;
    let head_dim = shape.dim / h;
    let q = linear(net, x, q_weight)?;
    let k = linear(net, x, k_weight)?;
    let v = linear(net, x, v_weight)?;
    let q = reshape(
        net,
        q,
        &[
            dims(shape.batch),
            dims(shape.frames),
            dims(h),
            dims(head_dim),
        ],
    )?;
    let q = permute(net, q, &[0, 2, 1, 3])?;
    let q = rope(net, q, shape)?;
    let k = reshape(
        net,
        k,
        &[
            dims(shape.batch),
            dims(shape.frames),
            dims(h),
            dims(head_dim),
        ],
    )?;
    let k = permute(net, k, &[0, 2, 1, 3])?;
    let k = rope(net, k, shape)?;
    let v = reshape(
        net,
        v,
        &[
            dims(shape.batch),
            dims(shape.frames),
            dims(h),
            dims(head_dim),
        ],
    )?;
    let v = permute(net, v, &[0, 2, 1, 3])?;
    let scores = net
        .add_matrix_multiply(&q, MatrixOperation::kNONE, &k, MatrixOperation::kTRANSPOSE)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let denom = scalar(net, 1.0 / (head_dim as f32).sqrt())?;
    let scores = binary(net, scores, denom, Ew::kPROD)?;
    let probs = net
        .add_softmax(&scores, Axes::new([3]))
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let result = net
        .add_matrix_multiply(&probs, MatrixOperation::kNONE, &v, MatrixOperation::kNONE)
        .map_err(fail)?
        .output(net, 0)
        .map_err(fail)?;
    let result = permute(net, result, &[0, 2, 1, 3])?;
    let result = reshape(
        net,
        result,
        &[dims(shape.batch), dims(shape.frames), dims(shape.dim)],
    )?;
    linear(net, result, out_weight)
}

fn modulate<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    scale: Tensor<'a>,
    shift: Tensor<'a>,
) -> Result<Tensor<'a>> {
    let one = scalar(net, 1.0)?;
    let scale = binary(net, scale, one, Ew::kSUM)?;
    let scaled = binary(net, x, scale, Ew::kPROD)?;
    binary(net, scaled, shift, Ew::kSUM)
}

pub(super) fn backbone_graph<'a>(
    net: &mut NetworkDefinition<'a>,
    x: Tensor<'a>,
    t: Tensor<'a>,
    weights: &TrtBackboneWeights,
) -> Result<Tensor<'a>> {
    let shape = weights.shape;
    let t = silu(net, t)?;
    let mut x = x;
    for block in &weights.blocks {
        let params = linear(net, t, &block.modulation)?;
        let pieces = (0..6)
            .map(|i| segment(net, params, shape.batch, shape.dim, i))
            .collect::<Result<Vec<_>>>()?;
        let parts = pieces
            .into_iter()
            .map(|x| reshape(net, x, &[dims(shape.batch), 1, dims(shape.dim)]))
            .collect::<Result<Vec<_>>>()?;
        let normed = layer_norm(net, x)?;
        let attn_input = modulate(net, normed, parts[1], parts[0])?;
        let attn = attention(
            net,
            attn_input,
            shape,
            &block.query,
            &block.key,
            &block.value,
            &block.attention_out,
        )?;
        let gated = binary(net, attn, parts[2], Ew::kPROD)?;
        x = binary(net, x, gated, Ew::kSUM)?;
        let normalized = layer_norm(net, x)?;
        let ffn_input = modulate(net, normalized, parts[4], parts[3])?;
        let ff = linear(net, ffn_input, &block.ff_in)?;
        let ff = gelu_tanh(net, ff)?;
        let ff = linear(net, ff, &block.ff_out)?;
        let gated = binary(net, ff, parts[5], Ew::kPROD)?;
        x = binary(net, x, gated, Ew::kSUM)?;
    }
    let final_params = linear(net, t, &weights.final_modulation)?;
    let scale = segment(net, final_params, shape.batch, shape.dim, 0)?;
    let shift = segment(net, final_params, shape.batch, shape.dim, 1)?;
    let scale = reshape(net, scale, &[dims(shape.batch), 1, dims(shape.dim)])?;
    let shift = reshape(net, shift, &[dims(shape.batch), 1, dims(shape.dim)])?;
    let normalized = layer_norm(net, x)?;
    let output = modulate(net, normalized, scale, shift)?;
    linear(net, output, &weights.projection)
}

pub fn build_serialized_backbone(
    logger: &Logger,
    weights: &TrtBackboneWeights,
    workspace_bytes: usize,
) -> Result<Vec<u8>> {
    let shape = weights.shape;
    shape.validate()?;
    if weights.blocks.len() != shape.depth || workspace_bytes == 0 {
        return Err(F5Error::InvalidConfiguration(
            "Invalid TensorRT F5 backbone weights or workspace",
        ));
    }
    let mut builder = Builder::new(logger).map_err(fail)?;
    let mut net = builder
        .create_network(trtx::builder::network_flags::EXPLICIT_BATCH)
        .map_err(fail)?;
    let x = net
        .add_input(
            "hidden",
            DataType::kFLOAT,
            &[dims(shape.batch), dims(shape.frames), dims(shape.dim)],
        )
        .map_err(fail)?;
    let t = net
        .add_input(
            "time_embedding",
            DataType::kFLOAT,
            &[dims(shape.batch), dims(shape.dim)],
        )
        .map_err(fail)?;
    let output = backbone_graph(&mut net, x, t, weights)?;
    output.set_name(&mut net, "output").map_err(fail)?;
    net.mark_output(&output);
    let mut config = builder.create_config().map_err(fail)?;
    config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, workspace_bytes);
    let engine = builder
        .build_serialized_network(&mut net, &mut config)
        .map_err(fail)?;
    if engine.is_empty() {
        return Err(F5Error::ModelExecution(
            "TensorRT produced an empty F5 backbone engine".into(),
        ));
    }
    Ok(engine.as_ref().to_vec())
}

pub(super) fn time_mlp_graph<'a>(
    net: &mut NetworkDefinition<'a>,
    sincos: Tensor<'a>,
    weights: &TrtTimeWeights,
) -> Result<Tensor<'a>> {
    let hidden = linear(net, sincos, &weights.input)?;
    let hidden = silu(net, hidden)?;
    linear(net, hidden, &weights.output)
}

/// Direct TensorRT network for F5's timestep MLP. Input [B, 256], output [B, D].
pub fn build_serialized_time_mlp(
    logger: &Logger,
    weights: &TrtTimeWeights,
    workspace_bytes: usize,
) -> Result<Vec<u8>> {
    if weights.batch == 0 || weights.dim == 0 || workspace_bytes == 0 {
        return Err(F5Error::InvalidConfiguration(
            "Invalid TensorRT timestep MLP configuration",
        ));
    }
    let mut builder = Builder::new(logger).map_err(fail)?;
    let mut net = builder
        .create_network(trtx::builder::network_flags::EXPLICIT_BATCH)
        .map_err(fail)?;
    let sincos = net
        .add_input("sincos", DataType::kFLOAT, &[dims(weights.batch), 256])
        .map_err(fail)?;
    let result = time_mlp_graph(&mut net, sincos, weights)?;
    result.set_name(&mut net, "time_embedding").map_err(fail)?;
    net.mark_output(&result);
    let mut config = builder.create_config().map_err(fail)?;
    config.set_memory_pool_limit(MemoryPoolType::kWORKSPACE, workspace_bytes);
    let image = builder
        .build_serialized_network(&mut net, &mut config)
        .map_err(fail)?;
    if image.is_empty() {
        return Err(F5Error::ModelExecution(
            "TensorRT produced empty F5 time engine".into(),
        ));
    }
    Ok(image.as_ref().to_vec())
}

/// # Safety
/// The stream and all buffers must belong to the same CUDA device. Input/output allocations
/// must outlive stream synchronization and must not overlap.
pub unsafe fn enqueue_time_mlp(
    context: &mut trtx::ExecutionContext<'_>,
    batch: usize,
    dim: usize,
    input: DeviceBuffer,
    output: DeviceBuffer,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    if batch == 0 || dim == 0 {
        return Err(F5Error::InvalidConfiguration("Invalid timestep batch"));
    }
    input.validate(
        batch
            .checked_mul(256)
            .and_then(|v| v.checked_mul(4))
            .ok_or(F5Error::InvalidShape("F5 sincos overflow"))?,
    )?;
    output.validate(
        batch
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(4))
            .ok_or(F5Error::InvalidShape("F5 timestep overflow"))?,
    )?;
    if stream.is_null()
        || input.address.checked_add(input.bytes).is_none()
        || output.address.checked_add(output.bytes).is_none()
        || (input.address < output.address + output.bytes
            && output.address < input.address + input.bytes)
    {
        return Err(F5Error::InvalidShape(
            "Invalid or overlapping TensorRT timestep buffers",
        ));
    }
    unsafe {
        context
            .set_input_tensor_address("sincos", input.address as *const _)
            .map_err(fail)?;
        context
            .set_tensor_address("time_embedding", output.address as *mut _)
            .map_err(fail)?;
        context.enqueue_v3(stream).map_err(fail)
    }
}

/// Keeps engine and execution context resident while a series of CFM steps is submitted.
/// The synchronizer is invoked before either object is dropped, even on a failed submission.
pub fn with_backbone_session<T>(
    logger: &Logger,
    serialized: &[u8],
    shape: BackboneShape,
    submit: impl FnOnce(&mut trtx::ExecutionContext<'_>) -> Result<T>,
    synchronize: impl FnOnce() -> Result<()>,
) -> Result<T> {
    shape.validate()?;
    if serialized.is_empty() {
        return Err(F5Error::InvalidConfiguration("Empty TensorRT engine"));
    }
    let mut runtime = trtx::Runtime::new(logger).map_err(fail)?;
    let mut engine = runtime.deserialize_cuda_engine(serialized).map_err(fail)?;
    let mut context = engine.create_execution_context().map_err(fail)?;
    let submitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| submit(&mut context)));
    let synced = synchronize();
    match submitted {
        Err(panic) => std::panic::resume_unwind(panic),
        Ok(result) => match (result, synced) {
            (_, Err(error)) => Err(error),
            (Err(error), Ok(())) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
        },
    }
}

/// # Safety
/// Every address must refer to a live allocation on the stream's CUDA device. The caller must
/// synchronize the stream before freeing buffers or destroying the execution context.
pub unsafe fn enqueue_backbone(
    context: &mut trtx::ExecutionContext<'_>,
    shape: BackboneShape,
    buffers: BackboneBindings,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    buffers.validate(shape)?;
    if stream.is_null() {
        return Err(F5Error::InvalidShape("Null CUDA stream"));
    }
    unsafe {
        context
            .set_input_tensor_address("hidden", buffers.hidden.address as *const _)
            .map_err(fail)?;
        context
            .set_input_tensor_address("time_embedding", buffers.time_embedding.address as *const _)
            .map_err(fail)?;
        context
            .set_tensor_address("output", buffers.output.address as *mut _)
            .map_err(fail)?;
        context.enqueue_v3(stream).map_err(fail)
    }
}

#[cfg(all(test, feature = "tensorrt-mock"))]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    #[ignore = "trtx mock creates a null INetworkDefinition; requires the actual TensorRT-RTX runtime"]
    fn tiny_f5_backbone_builds_with_native_trtx_graph() {
        let checkpoint =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dit_full_small.safetensors");
        let shape = BackboneShape {
            batch: 2,
            frames: 5,
            dim: 16,
            heads: 4,
            depth: 2,
            ff_mult: 2,
            mel_channels: 2,
        };
        let weights = TrtBackboneWeights::from_path(checkpoint, shape).unwrap();
        let logger = Logger::stderr().unwrap();
        let serialized = build_serialized_backbone(&logger, &weights, 32 << 20).unwrap();
        assert!(!serialized.is_empty());
        let checkpoint = crate::checkpoint::Checkpoint::open(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dit_full_small.safetensors"),
        )
        .unwrap();
        let time_weights = TrtTimeWeights::from_checkpoint(&checkpoint, 2, 16).unwrap();
        let image = build_serialized_time_mlp(&logger, &time_weights, 32 << 20).unwrap();
        assert!(!image.is_empty());
    }

    #[test]
    #[ignore = "trtx mock creates a null INetworkDefinition; requires the actual TensorRT-RTX runtime"]
    fn twenty_two_block_backbone_builds_with_native_trtx_graph() {
        let checkpoint = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/dit_full_22_blocks.safetensors");
        let shape = BackboneShape {
            batch: 2,
            frames: 7,
            dim: 16,
            heads: 4,
            depth: 22,
            ff_mult: 2,
            mel_channels: 2,
        };
        let weights = TrtBackboneWeights::from_path(checkpoint, shape).unwrap();
        let logger = Logger::stderr().unwrap();
        let serialized = build_serialized_backbone(&logger, &weights, 32 << 20).unwrap();
        assert!(!serialized.is_empty());
    }
}

pub fn load_tensorrt_runtime(path: Option<&std::path::Path>) -> Result<()> {
    match path {
        Some(path) => trtx::dynamically_load_tensorrt(Some(path)).map_err(fail),
        None => trtx::dynamically_load_tensorrt(None::<&std::path::Path>).map_err(fail),
    }
}
