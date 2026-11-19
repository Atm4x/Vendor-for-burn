//! Native CubeCL Firefly acoustic decoder. The CPU weighted executor is used only for parity.

use super::super::fsq::CubeClFsqExecutor;
use super::*;
use fish_speech_core::{
    audio::{FIREFLY_LEVELS, fish_transconv_output_len},
    firefly_cpu::TransConv1dKernel,
    firefly_trtx_mask::FireflyMaskPlan,
    fsq_cpu::{FsqCodebook, FsqGroupProjection, GroupedResidualFsqDecoder},
};
use std::collections::BTreeMap;

mod codec;
mod kernels;
mod validate;

pub use codec::{CubeClFireflyAcousticDecoder, CubeClFireflyCodec};
use kernels::*;
use validate::validate_decoder;

pub struct CubeClFireflyDecoder {
    client: Client,
}

impl CubeClFireflyDecoder {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// Executes the entire waveform Graph IR with a device-resident FSQ -> decoder boundary.
    /// Readback is performed once for the final mono PCM waveform.
    pub fn decode(
        &self,
        graph: &FireflyGraph,
        checkpoint: &BTreeMap<String, Vec<f32>>,
        indices: &[i32],
    ) -> Result<Vec<f32>> {
        validate_decoder(graph, checkpoint, indices)?;
        let client = &self.client;
        let count = |node_id: usize| -> Result<usize> {
            let x = graph.nodes[node_id].output;
            extent(&[x.batch, x.channels, x.frames])
        };
        let mut values: BTreeMap<usize, Handle> = BTreeMap::new();
        let mut retained: Vec<Handle> = Vec::new();
        for (id, node) in graph.nodes.iter().enumerate() {
            if id == graph.input {
                continue;
            }
            let ty = node.output;
            let len = count(id)?;
            let get = |index: usize| -> Result<Handle> {
                let predecessor = *node
                    .inputs
                    .get(index)
                    .ok_or(FishError::InvalidShape("missing decoder predecessor"))?;
                values
                    .get(&predecessor)
                    .cloned()
                    .ok_or(FishError::InvalidShape("unbound decoder predecessor"))
            };
            let param = |index: usize| -> Result<&[f32]> {
                let key = node
                    .checkpoint_keys
                    .get(index)
                    .ok_or(FishError::InvalidShape("missing decoder parameter key"))?;
                checkpoint
                    .get(key)
                    .map(Vec::as_slice)
                    .ok_or(FishError::InvalidShape("missing decoder checkpoint tensor"))
            };
            let handle = match node.operation {
                FireflyOp::FsqDecode { groups, quantizers } => {
                    let projections = (0..groups)
                        .map(|group| -> Result<FsqGroupProjection> {
                            Ok(FsqGroupProjection {
                                weights: param(2 * group)?.to_vec(),
                                bias: param(2 * group + 1)?.to_vec(),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let fsq = GroupedResidualFsqDecoder {
                        codebook: FsqCodebook::new(&FIREFLY_LEVELS)?,
                        num_quantizers: quantizers,
                        projections,
                    };
                    let (out, keep, channels) = CubeClFsqExecutor::new(client.clone())
                        .decode_device(&fsq, indices, ty.batch, ty.frames)?;
                    if channels != ty.channels {
                        return Err(FishError::InvalidShape("decoder FSQ channel mismatch"));
                    }
                    retained.extend(keep);
                    out
                }
                FireflyOp::FishConv { .. } | FireflyOp::Conv1x1 => {
                    let before = graph.nodes[node.inputs[0]].output;
                    let (kernel, stride, dilation, groups) = match node.operation {
                        FireflyOp::Conv1x1 => (1, 1, 1, 1),
                        FireflyOp::FishConv {
                            kernel,
                            stride,
                            dilation,
                            groups,
                        } => (kernel, stride, dilation, groups),
                        _ => unreachable!(),
                    };
                    let conv = Conv1dKernel {
                        in_channels: before.channels,
                        out_channels: ty.channels,
                        kernel_size: kernel,
                        stride,
                        dilation,
                        groups,
                        weights: param(0)?.to_vec(),
                        bias: Some(param(1)?.to_vec()),
                    };
                    CubeClConvNextExecutor::new(client.clone()).launch_conv(
                        &get(0)?,
                        ty.batch,
                        before.channels,
                        before.frames,
                        &conv,
                        &mut retained,
                    )?
                }
                FireflyOp::FishTransConv { kernel, stride } => {
                    let before = graph.nodes[node.inputs[0]].output;
                    let conv = TransConv1dKernel {
                        in_channels: before.channels,
                        out_channels: ty.channels,
                        kernel_size: kernel,
                        stride,
                        dilation: 1,
                        weights: param(0)?.to_vec(),
                        bias: Some(param(1)?.to_vec()),
                    };
                    self.launch_transconv(
                        &get(0)?,
                        before.batch,
                        before.frames,
                        &conv,
                        &mut retained,
                    )?
                }
                FireflyOp::LayerNorm => {
                    let w = client.create_from_slice(bytemuck::cast_slice(param(0)?));
                    let b = client.create_from_slice(bytemuck::cast_slice(param(1)?));
                    let eps = client.create_from_slice(bytemuck::cast_slice(&[1e-6f32]));
                    let out = client.empty(len * 4);
                    layer_norm_kernel::launch(
                        client,
                        grid(ty.batch * ty.frames)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(w.clone(), ty.channels) },
                        unsafe { BufferArg::from_raw_parts(b.clone(), ty.channels) },
                        unsafe { BufferArg::from_raw_parts(eps.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(out.clone(), len) },
                        ty.channels,
                        ty.frames,
                    );
                    retained.extend([w, b, eps]);
                    out
                }
                FireflyOp::LinearChannelsLast => {
                    let before = graph.nodes[node.inputs[0]].output;
                    let w = client.create_from_slice(bytemuck::cast_slice(param(0)?));
                    let b = client.create_from_slice(bytemuck::cast_slice(param(1)?));
                    let out = client.empty(len * 4);
                    linear_channels_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe {
                            BufferArg::from_raw_parts(
                                get(0)?,
                                extent(&[before.batch, before.channels, before.frames])?,
                            )
                        },
                        unsafe { BufferArg::from_raw_parts(w.clone(), param(0)?.len()) },
                        unsafe { BufferArg::from_raw_parts(b.clone(), param(1)?.len()) },
                        unsafe { BufferArg::from_raw_parts(out.clone(), len) },
                        before.channels,
                        ty.channels,
                        ty.frames,
                    );
                    retained.extend([w, b]);
                    out
                }
                FireflyOp::Gelu => {
                    let out = client.empty(len * 4);
                    gelu_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(out.clone(), len) },
                    );
                    out
                }
                FireflyOp::LayerScale => {
                    let gamma = client.create_from_slice(bytemuck::cast_slice(param(0)?));
                    let out = client.empty(len * 4);
                    channel_scale_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(gamma.clone(), ty.channels) },
                        unsafe { BufferArg::from_raw_parts(out.clone(), len) },
                        ty.channels,
                        ty.frames,
                    );
                    retained.push(gamma);
                    out
                }
                FireflyOp::SiLu | FireflyOp::Tanh => {
                    let out = client.empty(len * 4);
                    decoder_activation_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(out.clone(), len) },
                        matches!(node.operation, FireflyOp::Tanh),
                    );
                    out
                }
                FireflyOp::Add => {
                    let out = client.empty(len * 4);
                    add_tensor_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(get(1)?, len) },
                        unsafe { BufferArg::from_raw_parts(out.clone(), len) },
                    );
                    out
                }
                FireflyOp::ParallelMean => {
                    let branches = node.inputs.len();
                    let mut accumulator = client.empty(len * 4);
                    decoder_mean_first_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(accumulator.clone(), len) },
                        branches,
                    );
                    retained.push(accumulator.clone());
                    for branch in 1..branches {
                        let out = client.empty(len * 4);
                        decoder_mean_add_kernel::launch(
                            client,
                            grid(len)?,
                            CubeDim::new_1d(128),
                            unsafe { BufferArg::from_raw_parts(accumulator, len) },
                            unsafe { BufferArg::from_raw_parts(get(branch)?, len) },
                            unsafe { BufferArg::from_raw_parts(out.clone(), len) },
                            branches,
                        );
                        retained.push(out.clone());
                        accumulator = out;
                    }
                    accumulator
                }
                FireflyOp::Mask => {
                    let valid = FireflyMaskPlan::from_node(graph, id)?
                        .ok_or(FishError::InvalidShape("invalid decoder mask"))?
                        .valid_frames;
                    let out = client.empty(len * 4);
                    temporal_mask_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(out.clone(), len) },
                        ty.frames,
                        valid,
                    );
                    out
                }
                _ => {
                    return Err(FishError::InvalidShape(
                        "unsupported waveform decoder operator",
                    ));
                }
            };
            retained.push(handle.clone());
            values.insert(id, handle);
        }
        let output = values
            .remove(&graph.output)
            .ok_or(FishError::InvalidShape("missing decoder output"))?;
        let bytes = client
            .read_one(output)
            .map_err(|_| FishError::InvalidTransition("CubeCL waveform readback failed"))?;
        let ty = graph.nodes[graph.output].output;
        if bytes.len() != count(graph.output)? * 4 {
            return Err(FishError::InvalidShape(
                "CubeCL audio output length mismatch",
            ));
        }
        let wave = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        if wave.iter().any(|x| !x.is_finite()) || ty.batch != 1 || ty.channels != 1 {
            return Err(FishError::InvalidShape("invalid CubeCL mono waveform"));
        }
        Ok(wave[..graph.valid_frames].to_vec())
    }

    pub fn decode_codes(
        &self,
        checkpoint: &BTreeMap<String, Vec<f32>>,
        codes: &[i32],
        frames: usize,
    ) -> Result<Vec<f32>> {
        let graph = FireflyGraph::waveform_decode(1, frames, frames)?;
        self.decode(&graph, checkpoint, codes)
    }

    pub fn transconv1d(
        &self,
        conv: &TransConv1dKernel,
        input: &AudioTensor,
    ) -> Result<AudioTensor> {
        conv.validate()?;
        if input.channels != conv.in_channels {
            return Err(FishError::InvalidShape(
                "CubeCL ConvTranspose input channels mismatch",
            ));
        }
        let source = self
            .client
            .create_from_slice(bytemuck::cast_slice(&input.data));
        let mut retained = vec![source.clone()];
        let handle =
            self.launch_transconv(&source, input.batch, input.frames, conv, &mut retained)?;
        let frames =
            fish_transconv_output_len(input.frames, conv.kernel_size, conv.stride, conv.dilation)?;
        let size = extent(&[input.batch, conv.out_channels, frames])?;
        let bytes = self
            .client
            .read_one(handle)
            .map_err(|_| FishError::InvalidTransition("CubeCL ConvTranspose readback failed"))?;
        if bytes.len() != size * 4 {
            return Err(FishError::InvalidShape(
                "CubeCL ConvTranspose output extent mismatch",
            ));
        }
        AudioTensor::new(
            input.batch,
            conv.out_channels,
            frames,
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        )
    }

    fn launch_transconv(
        &self,
        input: &Handle,
        batch: usize,
        frames: usize,
        conv: &TransConv1dKernel,
        retained: &mut Vec<Handle>,
    ) -> Result<Handle> {
        conv.validate()?;
        let out_frames =
            fish_transconv_output_len(frames, conv.kernel_size, conv.stride, conv.dilation)?;
        let len = extent(&[batch, conv.out_channels, out_frames])?;
        let weight = self
            .client
            .create_from_slice(bytemuck::cast_slice(&conv.weights));
        let bias = self.client.create_from_slice(bytemuck::cast_slice(
            &conv
                .bias
                .clone()
                .unwrap_or_else(|| vec![0.0; conv.out_channels]),
        ));
        let output = self.client.empty(len * 4);
        transconv_kernel::launch(
            &self.client,
            grid(len)?,
            CubeDim::new_1d(128),
            unsafe {
                BufferArg::from_raw_parts(
                    input.clone(),
                    extent(&[batch, conv.in_channels, frames])?,
                )
            },
            unsafe { BufferArg::from_raw_parts(weight.clone(), conv.weights.len()) },
            unsafe { BufferArg::from_raw_parts(bias.clone(), conv.out_channels) },
            unsafe { BufferArg::from_raw_parts(output.clone(), len) },
            conv.in_channels,
            conv.out_channels,
            frames,
            out_frames,
            conv.kernel_size,
            conv.stride,
            conv.dilation,
        );
        retained.extend([weight, bias, output.clone()]);
        Ok(output)
    }
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests;

pub(crate) use codec::CubeClFireflyReferenceEncoder;
