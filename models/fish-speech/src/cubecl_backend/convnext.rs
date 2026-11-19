//! Device-executed ConvNeXt feature blocks with NCT tensors and PyTorch weight layouts.

use cubecl::prelude::*;
use cubecl::server::Handle;
use fish_speech_core::{
    audio::{fish_conv_output_len, fish_conv_padding},
    error::{FishError, Result},
    firefly_cpu::{AudioTensor, Conv1dKernel, ConvNextBlock},
    firefly_ir::{FireflyGraph, FireflyNode, FireflyOp},
};
use std::collections::BTreeMap;

const MAX_TENSOR_BYTES: usize = 512 * 1024 * 1024;

fn extent(shape: &[usize]) -> Result<usize> {
    let count = shape
        .iter()
        .try_fold(1usize, |n, &s| n.checked_mul(s))
        .ok_or(FishError::InvalidShape("CubeCL ConvNeXt extent overflow"))?;
    if count == 0 || count > MAX_TENSOR_BYTES / size_of::<f32>() {
        return Err(FishError::InvalidShape(
            "CubeCL ConvNeXt tensor budget exceeded",
        ));
    }
    Ok(count)
}

fn grid(count: usize) -> Result<CubeCount> {
    Ok(CubeCount::Static(
        u32::try_from(count.div_ceil(128))
            .map_err(|_| FishError::InvalidShape("CubeCL ConvNeXt launch extent overflow"))?,
        1,
        1,
    ))
}

mod kernels;

use kernels::*;

fn validate_block(block: &ConvNextBlock, channels: usize) -> Result<usize> {
    block.depthwise.validate()?;
    let conv = &block.depthwise;
    let hidden = block.pointwise1_bias.len();
    if conv.in_channels != channels
        || conv.out_channels != channels
        || conv.groups != channels
        || conv.stride != 1
        || hidden == 0
        || block.norm_weight.len() != channels
        || block.norm_bias.len() != channels
        || block.pointwise1_weight.len() != extent(&[hidden, channels])?
        || block.pointwise2_weight.len() != extent(&[channels, hidden])?
        || block.pointwise2_bias.len() != channels
        || block
            .gamma
            .as_ref()
            .is_some_and(|gamma| gamma.len() != channels)
        || !block.epsilon.is_finite()
        || block.epsilon <= 0.0
    {
        return Err(FishError::InvalidShape("invalid CubeCL ConvNeXt block"));
    }
    if block
        .depthwise
        .weights
        .iter()
        .chain(block.depthwise.bias.as_deref().unwrap_or(&[]))
        .chain(&block.norm_weight)
        .chain(&block.norm_bias)
        .chain(&block.pointwise1_weight)
        .chain(&block.pointwise1_bias)
        .chain(&block.pointwise2_weight)
        .chain(&block.pointwise2_bias)
        .chain(block.gamma.as_deref().unwrap_or(&[]))
        .any(|x| !x.is_finite())
    {
        return Err(FishError::InvalidShape(
            "non-finite CubeCL ConvNeXt parameter",
        ));
    }
    Ok(hidden)
}

/// Executes multiple ConvNeXt blocks without CPU intermediate-tensor readback.
/// Weight buffers are currently uploaded per call; this is a correctness-first implementation.
pub struct CubeClConvNextExecutor {
    client: Client,
}

impl CubeClConvNextExecutor {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    fn launch_conv(
        &self,
        source: &Handle,
        batch: usize,
        channels: usize,
        frames: usize,
        conv: &Conv1dKernel,
        retained: &mut Vec<Handle>,
    ) -> Result<Handle> {
        conv.validate()?;
        if conv.in_channels != channels {
            return Err(FishError::InvalidShape("CubeCL Conv1d channel mismatch"));
        }
        let out_frames =
            fish_conv_output_len(frames, conv.kernel_size, conv.stride, conv.dilation)?;
        let (pad_left, _) =
            fish_conv_padding(frames, conv.kernel_size, conv.stride, conv.dilation)?;
        let output_len = extent(&[batch, conv.out_channels, out_frames])?;
        let client = &self.client;
        let weights = client.create_from_slice(bytemuck::cast_slice(&conv.weights));
        let bias = conv
            .bias
            .clone()
            .unwrap_or_else(|| vec![0.0; conv.out_channels]);
        let biases = client.create_from_slice(bytemuck::cast_slice(&bias));
        let output = client.empty(output_len * 4);
        retained.extend([weights.clone(), biases.clone(), output.clone()]);
        grouped_conv_kernel::launch(
            client,
            grid(output_len)?,
            CubeDim::new_1d(128),
            unsafe {
                BufferArg::from_raw_parts(source.clone(), extent(&[batch, channels, frames])?)
            },
            unsafe { BufferArg::from_raw_parts(weights, conv.weights.len()) },
            unsafe { BufferArg::from_raw_parts(biases, bias.len()) },
            unsafe { BufferArg::from_raw_parts(output.clone(), output_len) },
            channels,
            conv.out_channels,
            frames,
            out_frames,
            conv.kernel_size,
            conv.stride,
            conv.dilation,
            conv.groups,
            pad_left,
        );
        Ok(output)
    }

    /// Binds one ConvNeXt block from the canonical Firefly Graph IR and checkpoint keys.
    /// Graph topology is checked before any GPU submission; no CPU operator executes here.
    pub fn forward_ir_block(
        &self,
        graph: &FireflyGraph,
        checkpoint: &BTreeMap<String, Vec<f32>>,
        prefix: &str,
        input: &AudioTensor,
    ) -> Result<AudioTensor> {
        graph.validate()?;
        let find = |suffix: &str| -> Result<(usize, &FireflyNode)> {
            graph
                .nodes
                .iter()
                .enumerate()
                .find(|(_, node)| node.name == format!("{prefix}.{suffix}"))
                .ok_or(FishError::InvalidShape(
                    "missing Firefly ConvNeXt graph node",
                ))
        };
        let parameter = |node: &FireflyNode, index: usize| -> Result<Vec<f32>> {
            let name = node
                .checkpoint_keys
                .get(index)
                .ok_or(FishError::InvalidShape(
                    "missing Firefly ConvNeXt weight key",
                ))?;
            checkpoint.get(name).cloned().ok_or(FishError::InvalidShape(
                "missing Firefly ConvNeXt checkpoint tensor",
            ))
        };
        let (dw_id, depthwise) = find("dwconv")?;
        let (norm_id, norm) = find("norm")?;
        let (pw1_id, pw1) = find("pwconv1")?;
        let (gelu_id, gelu) = find("gelu")?;
        let (pw2_id, pw2) = find("pwconv2")?;
        let (scale_id, scale) = find("layer_scale")?;
        let (_, residual) = find("residual")?;
        let (kernel, stride, dilation, groups) = match depthwise.operation {
            FireflyOp::FishConv {
                kernel,
                stride,
                dilation,
                groups,
            } => (kernel, stride, dilation, groups),
            _ => {
                return Err(FishError::InvalidShape(
                    "invalid Firefly ConvNeXt depthwise operator",
                ));
            }
        };
        if !matches!(norm.operation, FireflyOp::LayerNorm)
            || !matches!(pw1.operation, FireflyOp::LinearChannelsLast)
            || !matches!(gelu.operation, FireflyOp::Gelu)
            || !matches!(pw2.operation, FireflyOp::LinearChannelsLast)
            || !matches!(scale.operation, FireflyOp::LayerScale)
            || !matches!(residual.operation, FireflyOp::Add)
            || depthwise.inputs.len() != 1
            || norm.inputs != [dw_id]
            || pw1.inputs != [norm_id]
            || gelu.inputs != [pw1_id]
            || pw2.inputs != [gelu_id]
            || scale.inputs != [pw2_id]
            || residual.inputs != [depthwise.inputs[0], scale_id]
        {
            return Err(FishError::InvalidShape(
                "invalid Firefly ConvNeXt graph chain",
            ));
        }
        let channels = input.channels;
        if groups != channels
            || depthwise.output.channels != channels
            || norm.output.channels != channels
            || scale.output.channels != channels
            || pw2.output.channels != channels
            || pw1.output.channels != channels * 4
            || depthwise.output.frames != input.frames
            || depthwise.output.batch != input.batch
        {
            return Err(FishError::InvalidShape(
                "Firefly ConvNeXt profile does not match input",
            ));
        }
        let block = ConvNextBlock {
            depthwise: Conv1dKernel {
                in_channels: channels,
                out_channels: channels,
                kernel_size: kernel,
                stride,
                dilation,
                groups,
                weights: parameter(depthwise, 0)?,
                bias: Some(parameter(depthwise, 1)?),
            },
            norm_weight: parameter(norm, 0)?,
            norm_bias: parameter(norm, 1)?,
            pointwise1_weight: parameter(pw1, 0)?,
            pointwise1_bias: parameter(pw1, 1)?,
            pointwise2_weight: parameter(pw2, 0)?,
            pointwise2_bias: parameter(pw2, 1)?,
            gamma: Some(parameter(scale, 0)?),
            epsilon: 1e-6,
        };
        self.forward(&block, input)
    }

    pub fn conv1d(&self, conv: &Conv1dKernel, input: &AudioTensor) -> Result<AudioTensor> {
        if input.data.iter().any(|x| !x.is_finite()) {
            return Err(FishError::InvalidShape("non-finite CubeCL Conv1d input"));
        }
        let frames =
            fish_conv_output_len(input.frames, conv.kernel_size, conv.stride, conv.dilation)?;
        let size = extent(&[input.batch, conv.out_channels, frames])?;
        let data = self
            .client
            .create_from_slice(bytemuck::cast_slice(&input.data));
        let mut retained = vec![data.clone()];
        let output = self.launch_conv(
            &data,
            input.batch,
            input.channels,
            input.frames,
            conv,
            &mut retained,
        )?;
        let bytes = self
            .client
            .read_one(output)
            .map_err(|_| FishError::InvalidTransition("CubeCL Conv1d readback failed"))?;
        if bytes.len() != size * 4 {
            return Err(FishError::InvalidShape(
                "CubeCL Conv1d output size mismatch",
            ));
        }
        AudioTensor::new(
            input.batch,
            conv.out_channels,
            frames,
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )
    }

    pub fn forward(&self, block: &ConvNextBlock, input: &AudioTensor) -> Result<AudioTensor> {
        self.forward_chain(std::slice::from_ref(block), input)
    }

    pub fn forward_chain(
        &self,
        blocks: &[ConvNextBlock],
        input: &AudioTensor,
    ) -> Result<AudioTensor> {
        if blocks.is_empty() || input.data.iter().any(|x| !x.is_finite()) {
            return Err(FishError::InvalidShape("invalid CubeCL ConvNeXt input"));
        }
        let channels = input.channels;
        let frames = input.frames;
        let size = extent(&[input.batch, channels, frames])?;
        let hidden = blocks
            .iter()
            .map(|block| validate_block(block, channels))
            .collect::<Result<Vec<_>>>()?;
        let client = &self.client;
        let mut current = client.create_from_slice(bytemuck::cast_slice(&input.data));
        let mut retained = vec![current.clone()];
        for (block, &wide) in blocks.iter().zip(hidden.iter()) {
            let dw = self.launch_conv(
                &current,
                input.batch,
                channels,
                frames,
                &block.depthwise,
                &mut retained,
            )?;
            let norm_w = client.create_from_slice(bytemuck::cast_slice(&block.norm_weight));
            let norm_b = client.create_from_slice(bytemuck::cast_slice(&block.norm_bias));
            let normalized = client.empty(size * 4);
            let epsilon = client.create_from_slice(bytemuck::cast_slice(&[block.epsilon]));
            retained.extend([
                norm_w.clone(),
                norm_b.clone(),
                epsilon.clone(),
                normalized.clone(),
            ]);
            layer_norm_kernel::launch(
                client,
                grid(input.batch * frames)?,
                CubeDim::new_1d(128),
                unsafe { BufferArg::from_raw_parts(dw, size) },
                unsafe { BufferArg::from_raw_parts(norm_w, channels) },
                unsafe { BufferArg::from_raw_parts(norm_b, channels) },
                unsafe { BufferArg::from_raw_parts(epsilon, 1) },
                unsafe { BufferArg::from_raw_parts(normalized.clone(), size) },
                channels,
                frames,
            );
            let wide_len = extent(&[input.batch, wide, frames])?;
            let pw1 = client.create_from_slice(bytemuck::cast_slice(&block.pointwise1_weight));
            let pw1_b = client.create_from_slice(bytemuck::cast_slice(&block.pointwise1_bias));
            let expanded = client.empty(wide_len * 4);
            retained.extend([pw1.clone(), pw1_b.clone(), expanded.clone()]);
            linear_channels_kernel::launch(
                client,
                grid(wide_len)?,
                CubeDim::new_1d(128),
                unsafe { BufferArg::from_raw_parts(normalized, size) },
                unsafe { BufferArg::from_raw_parts(pw1, block.pointwise1_weight.len()) },
                unsafe { BufferArg::from_raw_parts(pw1_b, wide) },
                unsafe { BufferArg::from_raw_parts(expanded.clone(), wide_len) },
                channels,
                wide,
                frames,
            );
            let activated = client.empty(wide_len * 4);
            retained.push(activated.clone());
            gelu_kernel::launch(
                client,
                grid(wide_len)?,
                CubeDim::new_1d(128),
                unsafe { BufferArg::from_raw_parts(expanded, wide_len) },
                unsafe { BufferArg::from_raw_parts(activated.clone(), wide_len) },
            );
            let pw2 = client.create_from_slice(bytemuck::cast_slice(&block.pointwise2_weight));
            let pw2_b = client.create_from_slice(bytemuck::cast_slice(&block.pointwise2_bias));
            let projected = client.empty(size * 4);
            retained.extend([pw2.clone(), pw2_b.clone(), projected.clone()]);
            linear_channels_kernel::launch(
                client,
                grid(size)?,
                CubeDim::new_1d(128),
                unsafe { BufferArg::from_raw_parts(activated, wide_len) },
                unsafe { BufferArg::from_raw_parts(pw2, block.pointwise2_weight.len()) },
                unsafe { BufferArg::from_raw_parts(pw2_b, channels) },
                unsafe { BufferArg::from_raw_parts(projected.clone(), size) },
                wide,
                channels,
                frames,
            );
            let scales = block.gamma.clone().unwrap_or_else(|| vec![1.0; channels]);
            let gamma = client.create_from_slice(bytemuck::cast_slice(&scales));
            let residual = client.empty(size * 4);
            retained.extend([gamma.clone(), residual.clone()]);
            layer_scale_residual_kernel::launch(
                client,
                grid(size)?,
                CubeDim::new_1d(128),
                unsafe { BufferArg::from_raw_parts(current, size) },
                unsafe { BufferArg::from_raw_parts(projected, size) },
                unsafe { BufferArg::from_raw_parts(gamma, channels) },
                unsafe { BufferArg::from_raw_parts(residual.clone(), size) },
                channels,
                frames,
            );
            current = residual;
        }
        let bytes = client
            .read_one(current)
            .map_err(|_| FishError::InvalidTransition("CubeCL ConvNeXt readback failed"))?;
        if bytes.len() != size * 4 {
            return Err(FishError::InvalidShape(
                "CubeCL ConvNeXt readback length mismatch",
            ));
        }
        AudioTensor::new(
            input.batch,
            channels,
            frames,
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )
    }
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests;
