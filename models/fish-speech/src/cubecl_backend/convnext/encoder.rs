//! Graph-IR-ordered reference feature encoder with explicit CPU or native CubeCL LogMel inputs.
//! Prepared models reuse immutable device weights across request-scoped feature executions.

use super::super::fsq::{CubeClFsqExecutor, CubeClPreparedFsqEncode};
use super::super::logmel::CubeClLogMel;
use super::*;
use fish_speech_core::{
    audio::FIREFLY_LEVELS,
    fsq_cpu::{FsqCodebook, FsqGroupProjection, GroupedResidualFsqEncoder},
};
use fish_speech_core::{
    firefly_ir::{FireflyOp, FireflyStage, FireflyTensor},
    firefly_trtx_mask::FireflyMaskPlan,
    graph_ir::ScalarType,
};
use std::collections::BTreeSet;

fn weights<'a>(
    graph: &FireflyGraph,
    checkpoint: &'a BTreeMap<String, Vec<f32>>,
    id: usize,
    index: usize,
) -> Result<&'a [f32]> {
    let key = graph.nodes[id]
        .checkpoint_keys
        .get(index)
        .ok_or(FishError::InvalidShape("missing Firefly parameter binding"))?;
    checkpoint
        .get(key)
        .map(Vec::as_slice)
        .ok_or(FishError::InvalidShape("missing Firefly checkpoint key"))
}

fn tensor_len(shape: FireflyTensor) -> Result<usize> {
    extent(&[shape.batch, shape.channels, shape.frames])
}

fn prepared_steps(
    graph: &FireflyGraph,
    checkpoint: &BTreeMap<String, Vec<f32>>,
    start: usize,
    end: usize,
) -> Result<Vec<usize>> {
    if start >= end || end >= graph.nodes.len() {
        return Err(FishError::InvalidShape("invalid CubeCL encoder boundary"));
    }
    let mut required = BTreeSet::new();
    let mut stack = vec![end];
    while let Some(id) = stack.pop() {
        if !required.insert(id) || id == start {
            continue;
        }
        if id < start {
            return Err(FishError::InvalidShape(
                "encoder segment crosses LogMel boundary",
            ));
        }
        stack.extend(graph.nodes[id].inputs.iter().copied());
    }
    if !required.contains(&start) {
        return Err(FishError::InvalidShape(
            "encoder output does not depend on LogMel",
        ));
    }
    for &id in &required {
        let node = &graph.nodes[id];
        let ty = node.output;
        if ty.dtype != ScalarType::F32 {
            return Err(FishError::InvalidShape("non-F32 encoder feature node"));
        }
        tensor_len(ty)?;
        if id == start {
            continue;
        }
        let expected = match node.operation {
            FireflyOp::FishConv { .. }
            | FireflyOp::Conv1x1
            | FireflyOp::LayerNorm
            | FireflyOp::LinearChannelsLast => 2,
            FireflyOp::LayerScale => 1,
            FireflyOp::Gelu | FireflyOp::Add | FireflyOp::Mask => 0,
            _ => {
                return Err(FishError::InvalidShape(
                    "unsupported CubeCL feature operator",
                ));
            }
        };
        if node.checkpoint_keys.len() != expected {
            return Err(FishError::InvalidShape(
                "incorrect encoder checkpoint key count",
            ));
        }
        if node.operation == FireflyOp::Mask {
            FireflyMaskPlan::from_node(graph, id)?;
        }
        for key in &node.checkpoint_keys {
            let shape = &graph
                .weights
                .get(key)
                .ok_or(FishError::InvalidShape(
                    "missing declared encoder checkpoint tensor",
                ))?
                .shape;
            let count = shape
                .iter()
                .try_fold(1usize, |n, &s| n.checked_mul(s))
                .ok_or(FishError::InvalidShape("encoder parameter extent overflow"))?;
            let values = checkpoint
                .get(key)
                .ok_or(FishError::InvalidShape("missing encoder checkpoint tensor"))?;
            if values.len() != count || values.iter().any(|v| !v.is_finite()) {
                return Err(FishError::InvalidShape("invalid encoder checkpoint tensor"));
            }
        }
    }
    Ok(required.into_iter().filter(|&id| id != start).collect())
}

struct ResidentFeatures {
    handle: Handle,
    retained: Vec<Handle>,
    shape: FireflyTensor,
}

impl CubeClConvNextExecutor {
    /// Executes the reference encoder from external LogMel to an arbitrary F32 feature node.
    /// All intermediate activations remain device-resident; one readback occurs at `stop_name`.
    /// The entire required graph and every weight are validated before any GPU dispatch.
    pub fn reference_features_until(
        &self,
        graph: &FireflyGraph,
        checkpoint: &BTreeMap<String, Vec<f32>>,
        mel: &AudioTensor,
        stop_name: &str,
    ) -> Result<AudioTensor> {
        let resident = self.reference_device_until(graph, checkpoint, mel, stop_name)?;
        let len = tensor_len(resident.shape)?;
        let bytes = self
            .client
            .read_one(resident.handle)
            .map_err(|_| FishError::InvalidTransition("CubeCL encoder feature readback failed"))?;
        if bytes.len() != len * 4 {
            return Err(FishError::InvalidShape(
                "CubeCL encoder readback extent mismatch",
            ));
        }
        AudioTensor::new(
            resident.shape.batch,
            resident.shape.channels,
            resident.shape.frames,
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        )
    }

    fn reference_device_until(
        &self,
        graph: &FireflyGraph,
        checkpoint: &BTreeMap<String, Vec<f32>>,
        mel: &AudioTensor,
        stop_name: &str,
    ) -> Result<ResidentFeatures> {
        if mel.data.iter().any(|v| !v.is_finite()) {
            return Err(FishError::InvalidShape("invalid reference encoder inputs"));
        }
        let input = self.client.create_from_slice(bytemuck::cast_slice(&mel.data));
        self.reference_device_from_handle(
            graph, checkpoint, mel.batch, mel.channels, mel.frames,
            input.clone(), vec![input], stop_name, None,
        )
    }

    fn reference_device_from_handle(
        &self,
        graph: &FireflyGraph,
        checkpoint: &BTreeMap<String, Vec<f32>>,
        batch: usize,
        channels: usize,
        frames: usize,
        input: Handle,
        mut retained: Vec<Handle>,
        stop_name: &str,
        device_weights: Option<&BTreeMap<String, Handle>>,
    ) -> Result<ResidentFeatures> {
        graph.validate()?;
        if graph.stage != FireflyStage::ReferenceEncode {
            return Err(FishError::InvalidShape("invalid reference encoder graph"));
        }
        let start = graph
            .nodes
            .iter()
            .position(|v| v.operation == FireflyOp::LogMel)
            .ok_or(FishError::InvalidShape("missing reference LogMel boundary"))?;
        let stop = graph
            .nodes
            .iter()
            .position(|v| v.name == stop_name)
            .ok_or(FishError::InvalidShape("unknown reference feature boundary"))?;
        let logmel = graph.nodes[start].output;
        if (batch, channels, frames) != (logmel.batch, logmel.channels, logmel.frames) {
            return Err(FishError::InvalidShape("LogMel device tensor profile mismatch"));
        }
        let steps = prepared_steps(graph, checkpoint, start, stop)?;
        let client = &self.client;
        let mut values = BTreeMap::new();
        values.insert(start, input);
        for id in steps {
            let node = &graph.nodes[id];
            let output = node.output;
            let len = tensor_len(output)?;
            let get = |index: usize| -> Result<Handle> {
                let source = *node
                    .inputs
                    .get(index)
                    .ok_or(FishError::InvalidShape("missing encoder input edge"))?;
                values
                    .get(&source)
                    .cloned()
                    .ok_or(FishError::InvalidShape("unbound encoder input edge"))
            };
            let parameter_handle = |index: usize| -> Result<Handle> {
                let key = node.checkpoint_keys.get(index).ok_or(FishError::InvalidShape("missing encoder weight key"))?;
                if let Some(bank) = device_weights {
                    bank.get(key).cloned().ok_or(FishError::InvalidShape("missing prepared CubeCL weight"))
                } else {
                    Ok(client.create_from_slice(bytemuck::cast_slice(weights(graph, checkpoint, id, index)?)))
                }
            };
            let handle = match node.operation {
                FireflyOp::FishConv {
                    kernel,
                    stride,
                    dilation,
                    groups,
                } => {
                    let before = graph.nodes[node.inputs[0]].output;
                    let conv = Conv1dKernel {
                        in_channels: before.channels,
                        out_channels: output.channels,
                        kernel_size: kernel,
                        stride,
                        dilation,
                        groups,
                        weights: weights(graph, checkpoint, id, 0)?.to_vec(),
                        bias: Some(weights(graph, checkpoint, id, 1)?.to_vec()),
                    };
                    self.launch_conv_bound(
                        &get(0)?, before.batch, before.channels, before.frames,
                        &conv, &mut retained,
                        if device_weights.is_some() { Some((parameter_handle(0)?, parameter_handle(1)?)) } else { None },
                    )?
                }
                FireflyOp::Conv1x1 => {
                    let before = graph.nodes[node.inputs[0]].output;
                    let conv = Conv1dKernel {
                        in_channels: before.channels,
                        out_channels: output.channels,
                        kernel_size: 1,
                        stride: 1,
                        dilation: 1,
                        groups: 1,
                        weights: weights(graph, checkpoint, id, 0)?.to_vec(),
                        bias: Some(weights(graph, checkpoint, id, 1)?.to_vec()),
                    };
                    self.launch_conv_bound(
                        &get(0)?, before.batch, before.channels, before.frames,
                        &conv, &mut retained,
                        if device_weights.is_some() { Some((parameter_handle(0)?, parameter_handle(1)?)) } else { None },
                    )?
                }
                FireflyOp::LayerNorm => {
                    let w = parameter_handle(0)?;
                    let b = parameter_handle(1)?;
                    let epsilon = client.create_from_slice(bytemuck::cast_slice(&[1e-6f32]));
                    let result = client.empty(len * 4);
                    retained.extend([w.clone(), b.clone(), epsilon.clone()]);
                    layer_norm_kernel::launch(
                        client,
                        grid(output.batch * output.frames)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(w, output.channels) },
                        unsafe { BufferArg::from_raw_parts(b, output.channels) },
                        unsafe { BufferArg::from_raw_parts(epsilon, 1) },
                        unsafe { BufferArg::from_raw_parts(result.clone(), len) },
                        output.channels,
                        output.frames,
                    );
                    result
                }
                FireflyOp::LinearChannelsLast => {
                    let before = graph.nodes[node.inputs[0]].output;
                    let w = weights(graph, checkpoint, id, 0)?;
                    let b = weights(graph, checkpoint, id, 1)?;
                    let dw = parameter_handle(0)?;
                    let db = parameter_handle(1)?;
                    let result = client.empty(len * 4);
                    retained.extend([dw.clone(), db.clone()]);
                    linear_channels_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, tensor_len(before)?) },
                        unsafe { BufferArg::from_raw_parts(dw, w.len()) },
                        unsafe { BufferArg::from_raw_parts(db, b.len()) },
                        unsafe { BufferArg::from_raw_parts(result.clone(), len) },
                        before.channels,
                        output.channels,
                        output.frames,
                    );
                    result
                }
                FireflyOp::Gelu => {
                    let result = client.empty(len * 4);
                    gelu_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(result.clone(), len) },
                    );
                    result
                }
                FireflyOp::LayerScale => {
                    let gamma = parameter_handle(0)?;
                    retained.push(gamma.clone());
                    let result = client.empty(len * 4);
                    channel_scale_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(gamma, output.channels) },
                        unsafe { BufferArg::from_raw_parts(result.clone(), len) },
                        output.channels,
                        output.frames,
                    );
                    result
                }
                FireflyOp::Add => {
                    let result = client.empty(len * 4);
                    add_tensor_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(get(1)?, len) },
                        unsafe { BufferArg::from_raw_parts(result.clone(), len) },
                    );
                    result
                }
                FireflyOp::Mask => {
                    let mask = FireflyMaskPlan::from_node(graph, id)?
                        .ok_or(FishError::InvalidShape("expected reference mask operator"))?;
                    let result = client.empty(len * 4);
                    temporal_mask_kernel::launch(
                        client,
                        grid(len)?,
                        CubeDim::new_1d(128),
                        unsafe { BufferArg::from_raw_parts(get(0)?, len) },
                        unsafe { BufferArg::from_raw_parts(result.clone(), len) },
                        output.frames,
                        mask.valid_frames,
                    );
                    result
                }
                _ => {
                    return Err(FishError::InvalidShape(
                        "unsupported encoder feature operator",
                    ));
                }
            };
            retained.push(handle.clone());
            values.insert(id, handle);
        }
        let shape = graph.nodes[stop].output;
        Ok(ResidentFeatures {
            handle: values
                .remove(&stop)
                .ok_or(FishError::InvalidShape("missing encoder segment output"))?,
            retained,
            shape,
        })
    }

    /// Executes the complete reference feature graph and FSQ quantizer with no feature readback.
    /// The external LogMel boundary is explicit; CPU spectrogram preprocessing is not hidden.
    /// Explicit CPU LogMel preprocessing followed by native CubeCL reference encoding.
    /// Spectrogram evaluation is not claimed to run on CubeCL.
    pub fn reference_encode_with_cpu_logmel(
        &self,
        checkpoint: &BTreeMap<String, Vec<f32>>,
        pcm: &[&[f32]],
        sample_rate: usize,
    ) -> Result<Vec<i32>> {
        let prepared = fish_speech_core::reference_audio::ReferenceMelPreprocessor::new()
            .prepare_mono(pcm, sample_rate)?;
        let graph = prepared.reference_graph()?;
        self.reference_encode_from_mel(&graph, checkpoint, &prepared.mels)
    }

    pub fn reference_encode_from_mel(
        &self,
        graph: &FireflyGraph,
        checkpoint: &BTreeMap<String, Vec<f32>>,
        mel: &AudioTensor,
    ) -> Result<Vec<i32>> {
        let (pre_fsq, encoder) = Self::prepare_fsq_terminal(graph, checkpoint)?;
        let device = self.reference_device_until(graph, checkpoint, mel, &graph.nodes[pre_fsq].name)?;
        let _keep_alive = &device.retained;
        CubeClFsqExecutor::new(self.client.clone()).encode_device(
            &encoder, device.handle, device.shape.batch, device.shape.channels, device.shape.frames,
        )
    }

    /// Executes a Graph IR prefix directly from PCM, retaining the LogMel device buffer.
    pub fn reference_features_until_from_cubecl_pcm(
        &self,
        frontend: &CubeClLogMel,
        graph: &FireflyGraph,
        checkpoint: &BTreeMap<String, Vec<f32>>,
        pcm: &[f32],
        stop_name: &str,
    ) -> Result<AudioTensor> {
        let mel = frontend.transform_device(pcm)?;
        let resident = self.reference_device_from_handle(
            graph, checkpoint, 1, fish_speech_core::audio::N_MELS,
            mel.frames, mel.handle, mel.retained, stop_name, None,
        )?;
        let len = tensor_len(resident.shape)?;
        let bytes = self.client.read_one(resident.handle)
            .map_err(|_| FishError::InvalidTransition("CubeCL PCM encoder readback failed"))?;
        if bytes.len() != len * 4 {
            return Err(FishError::InvalidShape("CubeCL PCM encoder readback extent mismatch"));
        }
        AudioTensor::new(
            resident.shape.batch, resident.shape.channels, resident.shape.frames,
            bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect(),
        )
    }

    /// Native CubeCL PCM -> FFT -> mel -> complete Firefly encoder -> FSQ.
    /// Keeps the LogMel output on the device; `frontend` owns reusable FFT/mel constants.
    pub fn reference_encode_from_cubecl_pcm(
        &self,
        frontend: &CubeClLogMel,
        checkpoint: &BTreeMap<String, Vec<f32>>,
        pcm: &[f32],
    ) -> Result<Vec<i32>> {
        let frames = pcm.len() / fish_speech_core::audio::HOP_LENGTH;
        let graph = FireflyGraph::reference_encode(1, pcm.len())?;
        let (pre_fsq, encoder) = Self::prepare_fsq_terminal(&graph, checkpoint)?;
        let mel = frontend.transform_device(pcm)?;
        if mel.frames != frames {
            return Err(FishError::InvalidShape("CubeCL LogMel frame mismatch"));
        }
        let device = self.reference_device_from_handle(
            &graph, checkpoint, 1, fish_speech_core::audio::N_MELS,
            mel.frames, mel.handle, mel.retained, &graph.nodes[pre_fsq].name, None,
        )?;
        let _keep_alive = &device.retained;
        CubeClFsqExecutor::new(self.client.clone()).encode_device(
            &encoder, device.handle, device.shape.batch, device.shape.channels, device.shape.frames,
        )
    }

    fn prepare_fsq_terminal(
        graph: &FireflyGraph,
        checkpoint: &BTreeMap<String, Vec<f32>>,
    ) -> Result<(usize, GroupedResidualFsqEncoder)> {
        graph.validate()?;
        if graph.stage != FireflyStage::ReferenceEncode {
            return Err(FishError::InvalidShape(
                "not a Firefly reference encoder graph",
            ));
        }
        let fsq_id = graph.output;
        let fsq_node = &graph.nodes[fsq_id];
        let (groups, quantizers) = match fsq_node.operation {
            FireflyOp::FsqEncode { groups, quantizers } => (groups, quantizers),
            _ => return Err(FishError::InvalidShape("missing terminal reference FSQ")),
        };
        if fsq_node.inputs.len() != 1
            || fsq_node.checkpoint_keys.len() != groups * 2
            || groups == 0
            || quantizers == 0
        {
            return Err(FishError::InvalidShape(
                "invalid terminal FSQ projection bindings",
            ));
        }
        let pre_fsq = fsq_node.inputs[0];
        let before = graph.nodes[pre_fsq].output;
        if before.channels % groups != 0
            || fsq_node.output.dtype != ScalarType::I32
            || fsq_node.output.batch != before.batch
            || fsq_node.output.channels != groups * quantizers
            || fsq_node.output.frames != before.frames
        {
            return Err(FishError::InvalidShape(
                "invalid terminal FSQ tensor profile",
            ));
        }
        let projections = (0..groups)
            .map(|group| -> Result<FsqGroupProjection> {
                let w = weights(graph, checkpoint, fsq_id, group * 2)?;
                let b = weights(graph, checkpoint, fsq_id, group * 2 + 1)?;
                let w_shape = &graph.weights[&fsq_node.checkpoint_keys[group * 2]].shape;
                let b_shape = &graph.weights[&fsq_node.checkpoint_keys[group * 2 + 1]].shape;
                if w_shape != &[FIREFLY_LEVELS.len(), before.channels / groups]
                    || b_shape != &[FIREFLY_LEVELS.len()]
                    || w.len() != FIREFLY_LEVELS.len() * (before.channels / groups)
                    || b.len() != FIREFLY_LEVELS.len()
                    || w.iter().chain(b).any(|v| !v.is_finite())
                {
                    return Err(FishError::InvalidShape("invalid FSQ checkpoint projection"));
                }
                Ok(FsqGroupProjection {
                    weights: w.to_vec(),
                    bias: b.to_vec(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let encoder = GroupedResidualFsqEncoder {
            codebook: FsqCodebook::new(&FIREFLY_LEVELS)?,
            num_quantizers: quantizers,
            projections,
        };
        encoder.validate()?;
        Ok((pre_fsq, encoder))
    }

}

/// Model-scoped immutable Firefly encoder weights. All referenced Graph IR weights are
/// uploaded once. Request activations and LogMel FFT buffers remain request-local.
pub struct CubeClPreparedFireflyEncoder {
    executor: CubeClConvNextExecutor,
    frontend: CubeClLogMel,
    graph: FireflyGraph,
    checkpoint: BTreeMap<String, Vec<f32>>,
    device_weights: BTreeMap<String, Handle>,
    fsq: CubeClPreparedFsqEncode,
}

impl CubeClPreparedFireflyEncoder {
    pub fn new(
        client: Client,
        graph: FireflyGraph,
        checkpoint: BTreeMap<String, Vec<f32>>,
    ) -> Result<Self> {
        graph.validate()?;
        if graph.stage != FireflyStage::ReferenceEncode {
            return Err(FishError::InvalidShape("expected reference Firefly encoder"));
        }
        let (pre_fsq, fsq_encoder) = CubeClConvNextExecutor::prepare_fsq_terminal(&graph, &checkpoint)?;
        let start = graph.nodes.iter().position(|n| n.operation == FireflyOp::LogMel)
            .ok_or(FishError::InvalidShape("missing LogMel Graph IR input"))?;
        prepared_steps(&graph, &checkpoint, start, pre_fsq)?;
        let fsq = CubeClFsqExecutor::new(client.clone()).prepare_encode_device(&fsq_encoder)?;
        let mut device_weights = BTreeMap::new();
        for (node_id, node) in graph.nodes.iter().enumerate() {
            if node_id == graph.output { continue; }
            for key in &node.checkpoint_keys {
                if device_weights.contains_key(key) { continue; }
                let values = checkpoint.get(key)
                    .ok_or(FishError::InvalidShape("missing prepared Firefly checkpoint tensor"))?;
                let extent = graph.weights.get(key).ok_or(FishError::InvalidShape("undeclared Firefly checkpoint tensor"))?
                    .shape.iter().try_fold(1usize, |acc, &n| acc.checked_mul(n))
                    .ok_or(FishError::InvalidShape("Firefly weight extent overflow"))?;
                if values.len() != extent || values.iter().any(|v| !v.is_finite()) {
                    return Err(FishError::InvalidShape("invalid prepared Firefly checkpoint weight"));
                }
                device_weights.insert(key.clone(), client.create_from_slice(bytemuck::cast_slice(values)));
            }
        }
        Ok(Self {
            executor: CubeClConvNextExecutor::new(client.clone()),
            frontend: CubeClLogMel::new(client),
            graph,
            checkpoint,
            device_weights,
            fsq,
        })
    }

    pub fn prepared_weight_count(&self) -> usize { self.device_weights.len() }

    pub fn encode_mono_pcm(&self, pcm: &[f32]) -> Result<Vec<i32>> {
        let pre_fsq = self.graph.nodes[self.graph.output].inputs[0];
        let expected_frames = self.graph.nodes.iter().find(|n| n.operation == FireflyOp::LogMel)
            .ok_or(FishError::InvalidShape("missing prepared LogMel boundary"))?.output.frames;
        if pcm.len() / fish_speech_core::audio::HOP_LENGTH != expected_frames {
            return Err(FishError::InvalidShape("PCM length mismatches prepared Firefly graph"));
        }
        let mel = self.frontend.transform_device(pcm)?;
        let device = self.executor.reference_device_from_handle(
            &self.graph, &self.checkpoint, 1, fish_speech_core::audio::N_MELS,
            mel.frames, mel.handle, mel.retained, &self.graph.nodes[pre_fsq].name,
            Some(&self.device_weights),
        )?;
        let _keep_alive = &device.retained;
        CubeClFsqExecutor::new(self.executor.client.clone()).encode_device_prepared(
            &self.fsq, device.handle, device.shape.batch, device.shape.channels, device.shape.frames,
        )
    }
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests;
