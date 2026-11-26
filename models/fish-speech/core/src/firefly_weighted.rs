use std::collections::BTreeMap;
use std::sync::Arc;

use crate::audio::{FIREFLY_GROUPS, FIREFLY_LEVELS, HOP_LENGTH};
use crate::error::{FishError, Result};
use crate::firefly_cpu::{
    gelu, layer_norm_channels_first, silu, AudioTensor, Conv1dView, TransConv1dView,
};
use crate::firefly_ir::{FireflyGraph, FireflyNode, FireflyOp, FireflyStage};
use crate::fsq_cpu::{
    FsqCodebook, FsqGroupProjection, GroupedResidualFsqDecoder, GroupedResidualFsqEncoder,
};
use crate::graph_ir::ScalarType;
use crate::spectrogram_cpu::FishLogMel;

#[derive(Debug, Clone, PartialEq)]
pub enum FireflyValue {
    Floats(AudioTensor),
    Codes {
        batch: usize,
        channels: usize,
        frames: usize,
        data: Vec<i32>,
    },
}

impl FireflyValue {
    fn shape(&self) -> (usize, usize, usize, ScalarType) {
        match self {
            Self::Floats(a) => (a.batch, a.channels, a.frames, ScalarType::F32),
            Self::Codes {
                batch,
                channels,
                frames,
                ..
            } => (*batch, *channels, *frames, ScalarType::I32),
        }
    }

    fn as_audio(&self) -> Result<&AudioTensor> {
        match self {
            Self::Floats(a) => Ok(a),
            _ => Err(FishError::InvalidShape(
                "expected Firefly FP32 audio tensor",
            )),
        }
    }

    fn as_codes(&self) -> Result<(&[i32], usize, usize)> {
        match self {
            Self::Codes {
                data,
                batch,
                frames,
                ..
            } => Ok((data, *batch, *frames)),
            _ => Err(FishError::InvalidShape("expected Firefly FSQ indices")),
        }
    }
}

pub struct WeightedFireflyExecutor {
    graph: FireflyGraph,
    weights: Arc<BTreeMap<String, Vec<f32>>>,
    logmel: FishLogMel,
}

impl WeightedFireflyExecutor {
    pub fn new(graph: FireflyGraph, weights: BTreeMap<String, Vec<f32>>) -> Result<Self> {
        Self::with_shared_weights(graph, Arc::new(weights))
    }

    pub fn with_shared_weights(
        graph: FireflyGraph,
        weights: Arc<BTreeMap<String, Vec<f32>>>,
    ) -> Result<Self> {
        graph.validate()?;
        for (name, ty) in &graph.weights {
            let data = weights
                .get(name)
                .ok_or(FishError::InvalidShape("missing Firefly checkpoint tensor"))?;
            let size = ty
                .shape
                .iter()
                .try_fold(1usize, |v, &dim| v.checked_mul(dim))
                .ok_or(FishError::InvalidShape("Firefly checkpoint shape overflow"))?;
            if ty.dtype != ScalarType::F32
                || data.len() != size
                || data.iter().any(|v| !v.is_finite())
            {
                return Err(FishError::InvalidShape(
                    "incorrect Firefly checkpoint tensor shape or values",
                ));
            }
        }
        Ok(Self {
            graph,
            weights,
            logmel: FishLogMel::new(),
        })
    }

    pub fn graph(&self) -> &FireflyGraph {
        &self.graph
    }

    pub fn encode(&self, audio: AudioTensor) -> Result<(Vec<i32>, usize)> {
        if self.graph.stage != FireflyStage::ReferenceEncode {
            return Err(FishError::InvalidTransition(
                "Firefly decoder cannot encode reference",
            ));
        }
        let result = self.execute(FireflyValue::Floats(audio))?;
        let (codes, batch, frames) = result.as_codes()?;
        if batch != 1 {
            return Err(FishError::InvalidShape("reference API requires batch=1"));
        }
        let valid = self.graph.valid_frames;
        if valid > frames {
            return Err(FishError::InvalidShape("invalid Firefly feature length"));
        }
        let mut trimmed = Vec::with_capacity(FIREFLY_GROUPS * valid);
        for ch in 0..FIREFLY_GROUPS {
            trimmed.extend_from_slice(&codes[ch * frames..ch * frames + valid]);
        }
        Ok((trimmed, valid))
    }

    pub fn decode(&self, indices: &[i32]) -> Result<Vec<f32>> {
        if self.graph.stage != FireflyStage::WaveformDecode {
            return Err(FishError::InvalidTransition(
                "Firefly encoder cannot decode codes",
            ));
        }
        let input = self.graph.nodes[self.graph.input].output;
        if input.batch != 1 || indices.len() != input.channels * input.frames {
            return Err(FishError::InvalidShape(
                "acoustic code input shape mismatch",
            ));
        }
        let result = self.execute(FireflyValue::Codes {
            batch: 1,
            channels: input.channels,
            frames: input.frames,
            data: indices.to_vec(),
        })?;
        let audio = result.as_audio()?;
        if audio.batch != 1 || audio.channels != 1 || self.graph.valid_frames > audio.frames {
            return Err(FishError::InvalidShape("invalid Firefly waveform shape"));
        }
        Ok(audio.data[..self.graph.valid_frames].to_vec())
    }

    fn weight(&self, node: &FireflyNode, index: usize) -> Result<&[f32]> {
        let key = node
            .checkpoint_keys
            .get(index)
            .ok_or(FishError::InvalidShape(
                "missing Firefly operator parameter name",
            ))?;
        self.weights
            .get(key)
            .map(Vec::as_slice)
            .ok_or(FishError::InvalidShape(
                "missing Firefly operator parameter",
            ))
    }

    fn fsq_projections(
        &self,
        node: &FireflyNode,
        groups: usize,
    ) -> Result<Vec<FsqGroupProjection>> {
        if node.checkpoint_keys.len() != groups * 2 {
            return Err(FishError::InvalidShape("FSQ projection key count mismatch"));
        }
        (0..groups)
            .map(|g| {
                Ok(FsqGroupProjection {
                    weights: self.weight(node, 2 * g)?.to_vec(),
                    bias: self.weight(node, 2 * g + 1)?.to_vec(),
                })
            })
            .collect()
    }

    fn conv(
        &self,
        node: &FireflyNode,
        x: &AudioTensor,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<FireflyValue> {
        let op = Conv1dView {
            in_channels: x.channels,
            out_channels: node.output.channels,
            kernel_size: kernel,
            stride,
            dilation,
            groups,
            weights: self.weight(node, 0)?,
            bias: Some(self.weight(node, 1)?),
        };
        Ok(FireflyValue::Floats(op.forward(x)?))
    }

    fn evaluate(&self, node: &FireflyNode, args: &[&FireflyValue]) -> Result<FireflyValue> {
        let x = args
            .first()
            .ok_or(FishError::InvalidShape("Firefly node has no input"))?;
        match node.operation {
            FireflyOp::AudioInput | FireflyOp::CodeInput => {
                Err(FishError::InvalidTransition("input cannot be evaluated"))
            }
            FireflyOp::LogMel => {
                let audio = x.as_audio()?;
                if audio.channels != 1 {
                    return Err(FishError::InvalidShape("Firefly LogMel expects mono"));
                }
                let mut result =
                    vec![0f32; node.output.batch * node.output.channels * node.output.frames];
                for b in 0..audio.batch {
                    let source = &audio.data[b * audio.frames..(b + 1) * audio.frames];
                    let mel = self.logmel.transform_mono(source)?;
                    if mel.frames != node.output.frames || mel.channels != node.output.channels {
                        return Err(FishError::InvalidShape("Firefly LogMel profile mismatch"));
                    }
                    let chunk = mel.data.len();
                    result[b * chunk..(b + 1) * chunk].copy_from_slice(&mel.data);
                }
                Ok(FireflyValue::Floats(AudioTensor::new(
                    audio.batch,
                    node.output.channels,
                    node.output.frames,
                    result,
                )?))
            }
            FireflyOp::FishConv {
                kernel,
                stride,
                dilation,
                groups,
            } => self.conv(node, x.as_audio()?, kernel, stride, dilation, groups),
            FireflyOp::Conv1x1 => self.conv(node, x.as_audio()?, 1, 1, 1, 1),
            FireflyOp::FishTransConv { kernel, stride } => {
                let audio = x.as_audio()?;
                let op = TransConv1dView {
                    in_channels: audio.channels,
                    out_channels: node.output.channels,
                    kernel_size: kernel,
                    stride,
                    dilation: 1,
                    weights: self.weight(node, 0)?,
                    bias: Some(self.weight(node, 1)?),
                };
                Ok(FireflyValue::Floats(op.forward(audio)?))
            }
            FireflyOp::LayerNorm => Ok(FireflyValue::Floats(layer_norm_channels_first(
                x.as_audio()?,
                self.weight(node, 0)?,
                self.weight(node, 1)?,
                1e-6,
            )?)),
            FireflyOp::LinearChannelsLast => {
                let src = x.as_audio()?;
                let w = self.weight(node, 0)?;
                let bias = self.weight(node, 1)?;
                let mut out = vec![0f32; src.batch * node.output.channels * src.frames];
                for b in 0..src.batch {
                    for c in 0..node.output.channels {
                        for t in 0..src.frames {
                            let mut v = bias[c];
                            for i in 0..src.channels {
                                v += w[c * src.channels + i]
                                    * src.data[(b * src.channels + i) * src.frames + t];
                            }
                            out[(b * node.output.channels + c) * src.frames + t] = v;
                        }
                    }
                }
                Ok(FireflyValue::Floats(AudioTensor::new(
                    src.batch,
                    node.output.channels,
                    src.frames,
                    out,
                )?))
            }
            FireflyOp::SiLu | FireflyOp::Gelu | FireflyOp::Tanh => {
                let mut out = x.as_audio()?.clone();
                for v in &mut out.data {
                    *v = match node.operation {
                        FireflyOp::SiLu => silu(*v),
                        FireflyOp::Gelu => gelu(*v),
                        _ => v.tanh(),
                    };
                }
                Ok(FireflyValue::Floats(out))
            }
            FireflyOp::LayerScale => {
                let mut out = x.as_audio()?.clone();
                let scale = self.weight(node, 0)?;
                for b in 0..out.batch {
                    for (c, &s) in scale.iter().enumerate() {
                        for t in 0..out.frames {
                            out.data[(b * out.channels + c) * out.frames + t] *= s;
                        }
                    }
                }
                Ok(FireflyValue::Floats(out))
            }
            FireflyOp::Add | FireflyOp::ParallelMean => {
                if args.is_empty() || (matches!(node.operation, FireflyOp::Add) && args.len() != 2)
                {
                    return Err(FishError::InvalidShape("invalid Firefly residual inputs"));
                }
                let mut out = args[0].as_audio()?.clone();
                if matches!(node.operation, FireflyOp::ParallelMean) {
                    out.data.fill(0.0);
                }
                for arg in if matches!(node.operation, FireflyOp::ParallelMean) {
                    args
                } else {
                    &args[1..]
                } {
                    let a = arg.as_audio()?;
                    if (a.batch, a.channels, a.frames) != (out.batch, out.channels, out.frames) {
                        return Err(FishError::InvalidShape("Firefly residual shape mismatch"));
                    }
                    let div = if matches!(node.operation, FireflyOp::ParallelMean) {
                        args.len() as f32
                    } else {
                        1.0
                    };
                    for (d, s) in out.data.iter_mut().zip(&a.data) {
                        *d += *s / div;
                    }
                }
                Ok(FireflyValue::Floats(out))
            }
            FireflyOp::Mask => {
                let mut out = x.as_audio()?.clone();
                let valid = match (self.graph.stage, node.name.as_str()) {
                    (
                        FireflyStage::ReferenceEncode,
                        "reference.mel_mask" | "reference.feature_mask",
                    ) => out.frames,
                    (FireflyStage::WaveformDecode, "waveform.quantized_mel_mask") => {
                        self.graph.valid_frames / HOP_LENGTH
                    }
                    (FireflyStage::WaveformDecode, "waveform.audio_mask") => {
                        self.graph.valid_frames
                    }
                    _ => return Err(FishError::InvalidShape("unknown Firefly mask")),
                };
                if valid > out.frames {
                    return Err(FishError::InvalidShape("Firefly mask exceeds tensor"));
                }
                for channel in out.data.chunks_exact_mut(out.frames) {
                    channel[valid..].fill(0.0);
                }
                Ok(FireflyValue::Floats(out))
            }
            FireflyOp::FsqEncode { groups, quantizers } => {
                let fsq = GroupedResidualFsqEncoder {
                    codebook: FsqCodebook::new(&FIREFLY_LEVELS)?,
                    num_quantizers: quantizers,
                    projections: self.fsq_projections(node, groups)?,
                };
                let src = x.as_audio()?;
                Ok(FireflyValue::Codes {
                    batch: src.batch,
                    channels: node.output.channels,
                    frames: src.frames,
                    data: fsq.encode(src)?,
                })
            }
            FireflyOp::FsqDecode { groups, quantizers } => {
                let fsq = GroupedResidualFsqDecoder {
                    codebook: FsqCodebook::new(&FIREFLY_LEVELS)?,
                    num_quantizers: quantizers,
                    projections: self.fsq_projections(node, groups)?,
                };
                let (codes, batch, frames) = x.as_codes()?;
                Ok(FireflyValue::Floats(fsq.decode(codes, batch, frames)?))
            }
        }
    }

    pub fn execute(&self, input: FireflyValue) -> Result<FireflyValue> {
        self.graph.validate()?;
        let expected = self.graph.nodes[self.graph.input].output;
        if input.shape()
            != (
                expected.batch,
                expected.channels,
                expected.frames,
                expected.dtype,
            )
        {
            return Err(FishError::InvalidShape(
                "Firefly input does not match static graph",
            ));
        }
        let length = expected
            .batch
            .checked_mul(expected.channels)
            .and_then(|n| n.checked_mul(expected.frames))
            .ok_or(FishError::InvalidShape("Firefly input extent overflow"))?;
        match &input {
            FireflyValue::Floats(audio)
                if audio.data.len() != length || audio.data.iter().any(|s| !s.is_finite()) =>
            {
                return Err(FishError::InvalidShape(
                    "invalid Firefly audio input payload",
                ));
            }
            FireflyValue::Codes { data, .. } if data.len() != length => {
                return Err(FishError::InvalidShape(
                    "invalid Firefly code input payload",
                ));
            }
            _ => {}
        }
        let mut remaining = vec![0usize; self.graph.nodes.len()];
        for node in &self.graph.nodes {
            for &dependency in &node.inputs {
                remaining[dependency] += 1;
            }
        }
        let mut values: Vec<Option<FireflyValue>> = vec![None; self.graph.nodes.len()];
        values[self.graph.input] = Some(input);
        for (id, node) in self.graph.nodes.iter().enumerate() {
            if id == self.graph.input {
                continue;
            }
            let arguments = node
                .inputs
                .iter()
                .map(|&i| {
                    values[i].as_ref().ok_or(FishError::InvalidTransition(
                        "missing Firefly DAG dependency",
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let output = self.evaluate(node, &arguments)?;
            let shape = node.output;
            if output.shape() != (shape.batch, shape.channels, shape.frames, shape.dtype) {
                return Err(FishError::InvalidShape(
                    "Firefly operator output shape mismatch",
                ));
            }
            if let FireflyValue::Floats(ref tensor) = output {
                if tensor.data.iter().any(|s| !s.is_finite()) {
                    return Err(FishError::InvalidShape(
                        "non-finite weighted Firefly output",
                    ));
                }
            }
            values[id] = Some(output);
            for &dep in &node.inputs {
                remaining[dep] -= 1;
                if remaining[dep] == 0 && dep != self.graph.output {
                    values[dep] = None;
                }
            }
        }
        values[self.graph.output]
            .take()
            .ok_or(FishError::InvalidTransition("Firefly output was released"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::firefly_ir::FireflyTensor;
    use crate::graph_ir::{Layout, TensorType};

    fn node(
        name: &str,
        operation: FireflyOp,
        inputs: &[usize],
        c: usize,
        f: usize,
        dtype: ScalarType,
        keys: &[&str],
    ) -> FireflyNode {
        FireflyNode {
            name: name.to_string(),
            operation,
            inputs: inputs.to_vec(),
            output: FireflyTensor {
                batch: 1,
                channels: c,
                frames: f,
                dtype,
            },
            checkpoint_keys: keys.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn insert_weight(
        graph: &mut FireflyGraph,
        weights: &mut BTreeMap<String, Vec<f32>>,
        key: &str,
        shape: &[usize],
        values: Vec<f32>,
    ) {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        graph.weights.insert(
            key.to_string(),
            TensorType::new(shape.to_vec(), ScalarType::F32, Layout::Plain),
        );
        weights.insert(key.to_string(), values);
    }

    #[test]
    fn weighted_decoder_fsq_convolution_tanh_masks_tail() {
        let mut graph = FireflyGraph {
            stage: FireflyStage::WaveformDecode,
            nodes: vec![
                node(
                    "codes",
                    FireflyOp::CodeInput,
                    &[],
                    8,
                    2,
                    ScalarType::I32,
                    &[],
                ),
                node(
                    "quantizer.fsq_decode",
                    FireflyOp::FsqDecode {
                        groups: 8,
                        quantizers: 1,
                    },
                    &[0],
                    16,
                    2,
                    ScalarType::F32,
                    &[],
                ),
                node(
                    "linear",
                    FireflyOp::Conv1x1,
                    &[1],
                    1,
                    2,
                    ScalarType::F32,
                    &["linear.weight", "linear.bias"],
                ),
                node("act", FireflyOp::Tanh, &[2], 1, 2, ScalarType::F32, &[]),
                node(
                    "waveform.audio_mask",
                    FireflyOp::Mask,
                    &[3],
                    1,
                    2,
                    ScalarType::F32,
                    &[],
                ),
            ],
            weights: BTreeMap::new(),
            input: 0,
            output: 4,
            valid_frames: 1,
        };
        let mut weights = BTreeMap::new();
        for group in 0..8 {
            let w = format!("quantizer.residual_fsq.rvqs.{group}.project_out.weight");
            let b = format!("quantizer.residual_fsq.rvqs.{group}.project_out.bias");
            insert_weight(&mut graph, &mut weights, &w, &[2, 4], vec![0.0; 8]);
            insert_weight(&mut graph, &mut weights, &b, &[2], vec![0.25, 0.25]);
            graph.nodes[1].checkpoint_keys.extend([w, b]);
        }
        insert_weight(
            &mut graph,
            &mut weights,
            "linear.weight",
            &[1, 16, 1],
            vec![0.125; 16],
        );
        insert_weight(&mut graph, &mut weights, "linear.bias", &[1], vec![0.0]);
        let executor = WeightedFireflyExecutor::new(graph, weights).unwrap();
        let pcm = executor.decode(&vec![0i32; 16]).unwrap();
        assert_eq!(pcm.len(), 1);
        assert!((pcm[0] - 0.5f32.tanh()).abs() < 1e-6);
        assert!(executor.decode(&vec![1000; 16]).is_err());
    }

    #[test]
    fn weighted_encoder_uses_real_logmel_and_fsq_cpu() {
        let mut graph = FireflyGraph {
            stage: FireflyStage::ReferenceEncode,
            nodes: vec![
                node(
                    "audio",
                    FireflyOp::AudioInput,
                    &[],
                    1,
                    2048,
                    ScalarType::F32,
                    &[],
                ),
                node(
                    "spec_transform.logmel",
                    FireflyOp::LogMel,
                    &[0],
                    160,
                    4,
                    ScalarType::F32,
                    &[],
                ),
                node(
                    "quantizer.fsq_encode",
                    FireflyOp::FsqEncode {
                        groups: 8,
                        quantizers: 1,
                    },
                    &[1],
                    8,
                    4,
                    ScalarType::I32,
                    &[],
                ),
            ],
            weights: BTreeMap::new(),
            input: 0,
            output: 2,
            valid_frames: 4,
        };
        let mut weights = BTreeMap::new();
        for group in 0..8 {
            let w = format!("quantizer.residual_fsq.rvqs.{group}.project_in.weight");
            let b = format!("quantizer.residual_fsq.rvqs.{group}.project_in.bias");
            insert_weight(&mut graph, &mut weights, &w, &[4, 20], vec![0.0; 80]);
            insert_weight(&mut graph, &mut weights, &b, &[4], vec![0.0; 4]);
            graph.nodes[2].checkpoint_keys.extend([w, b]);
        }
        let executor = WeightedFireflyExecutor::new(graph, weights).unwrap();
        let input = AudioTensor::new(
            1,
            1,
            2048,
            (0..2048).map(|i| (i as f32 * 0.02).sin() * 0.1).collect(),
        )
        .unwrap();
        let (codes, valid) = executor.encode(input).unwrap();
        assert_eq!(valid, 4);
        assert_eq!(codes.len(), 32);
        assert!(codes.iter().all(|&c| (0..1000).contains(&c)));
    }

    #[test]
    fn rejects_missing_nonfinite_and_wrong_shape_weights() {
        let graph = FireflyGraph {
            stage: FireflyStage::WaveformDecode,
            nodes: vec![
                node(
                    "codes",
                    FireflyOp::CodeInput,
                    &[],
                    8,
                    1,
                    ScalarType::I32,
                    &[],
                ),
                node(
                    "post",
                    FireflyOp::Conv1x1,
                    &[0],
                    1,
                    1,
                    ScalarType::F32,
                    &["post.weight", "post.bias"],
                ),
            ],
            weights: BTreeMap::from([
                (
                    "post.weight".to_string(),
                    TensorType::new(vec![1, 8, 1], ScalarType::F32, Layout::Plain),
                ),
                (
                    "post.bias".to_string(),
                    TensorType::new(vec![1], ScalarType::F32, Layout::Plain),
                ),
            ]),
            input: 0,
            output: 1,
            valid_frames: 1,
        };
        let mut weights = BTreeMap::from([
            ("post.weight".to_string(), vec![0.0; 8]),
            ("post.bias".to_string(), vec![0.0]),
        ]);
        assert!(WeightedFireflyExecutor::new(graph.clone(), BTreeMap::new()).is_err());
        weights.get_mut("post.weight").unwrap().push(0.0);
        assert!(WeightedFireflyExecutor::new(graph.clone(), weights.clone()).is_err());
        weights.get_mut("post.weight").unwrap().pop();
        weights.get_mut("post.weight").unwrap()[0] = f32::NAN;
        assert!(WeightedFireflyExecutor::new(graph, weights).is_err());
    }

    #[test]
    fn branched_add_preserves_original_before_last_consumer() {
        let graph = FireflyGraph {
            stage: FireflyStage::WaveformDecode,
            nodes: vec![
                node(
                    "codes",
                    FireflyOp::CodeInput,
                    &[],
                    8,
                    1,
                    ScalarType::I32,
                    &[],
                ),
                node(
                    "fsq",
                    FireflyOp::FsqDecode {
                        groups: 8,
                        quantizers: 1,
                    },
                    &[0],
                    8,
                    1,
                    ScalarType::F32,
                    &[],
                ),
                node("silu", FireflyOp::SiLu, &[1], 8, 1, ScalarType::F32, &[]),
                node(
                    "residual",
                    FireflyOp::Add,
                    &[1, 2],
                    8,
                    1,
                    ScalarType::F32,
                    &[],
                ),
            ],
            weights: BTreeMap::new(),
            input: 0,
            output: 3,
            valid_frames: 1,
        };
        let mut graph = graph;
        let mut weights = BTreeMap::new();
        for group in 0..8 {
            let w = format!("quantizer.residual_fsq.rvqs.{group}.project_out.weight");
            let b = format!("quantizer.residual_fsq.rvqs.{group}.project_out.bias");
            insert_weight(&mut graph, &mut weights, &w, &[1, 4], vec![0.0; 4]);
            insert_weight(&mut graph, &mut weights, &b, &[1], vec![2.0]);
            graph.nodes[1].checkpoint_keys.extend([w, b]);
        }
        let output = WeightedFireflyExecutor::new(graph, weights)
            .unwrap()
            .execute(FireflyValue::Codes {
                batch: 1,
                channels: 8,
                frames: 1,
                data: vec![0; 8],
            })
            .unwrap();
        assert!(output
            .as_audio()
            .unwrap()
            .data
            .iter()
            .all(|&v| (v - 2.0 - silu(2.0)).abs() < 1e-6));
    }
}

#[cfg(test)]
mod full_synthetic_graph_tests {
    use super::*;

    #[test]
    #[ignore = "large CPU reference graph; run explicitly when checking the full Firefly pipeline"]
    fn all_reference_and_decoder_nodes_execute_with_zero_weights() {
        let decoder_graph = FireflyGraph::waveform_decode(1, 1, 1).unwrap();
        let decoder_weights = decoder_graph
            .weights
            .iter()
            .map(|(key, ty)| (key.clone(), vec![0.0; ty.shape.iter().product()]))
            .collect();
        let decoder = WeightedFireflyExecutor::new(decoder_graph, decoder_weights).unwrap();
        assert_eq!(
            decoder.decode(&[0; FIREFLY_GROUPS]).unwrap(),
            vec![0.0; 2048]
        );
        drop(decoder);

        let encoder_graph = FireflyGraph::reference_encode(1, 4096).unwrap();
        let encoder_weights = encoder_graph
            .weights
            .iter()
            .map(|(key, ty)| (key.clone(), vec![0.0; ty.shape.iter().product()]))
            .collect();
        let encoder = WeightedFireflyExecutor::new(encoder_graph, encoder_weights).unwrap();
        let (codes, frames) = encoder
            .encode(AudioTensor::new(1, 1, 4096, vec![0.0; 4096]).unwrap())
            .unwrap();
        assert_eq!(frames, 2);
        assert_eq!(codes.len(), 16);
        assert!(codes.iter().all(|&code| (0..1000).contains(&code)));
    }
}

#[cfg(test)]
mod full_nonzero_graph_test {
    use super::*;

    #[test]
    #[ignore = "full-sized synthetic Firefly CPU execution; requires optimized test profile"]
    fn weighted_full_decoder_returns_finite_nonzero_waveform() {
        let graph = FireflyGraph::waveform_decode(1, 2, 2).unwrap();
        let weights = graph
            .weights
            .iter()
            .map(|(key, ty)| {
                let count = ty.shape.iter().product();
                let value = if key.ends_with(".bias") {
                    0.0002
                } else {
                    0.00001
                };
                (key.clone(), vec![value; count])
            })
            .collect();
        let decoder = WeightedFireflyExecutor::new(graph, weights).unwrap();
        let pcm = decoder
            .decode(&[
                0, 2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
            ])
            .unwrap();
        assert_eq!(pcm.len(), 4096);
        assert!(pcm.iter().all(|&value| value.is_finite()));
        assert!(pcm.iter().any(|&value| value != 0.0));
    }
}
