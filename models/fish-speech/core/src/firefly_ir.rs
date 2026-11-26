use std::collections::{BTreeMap, BTreeSet};

use crate::audio::{
    firefly_decode_lengths, firefly_reference_lengths, fish_conv_output_len,
    fish_transconv_output_len, FIREFLY_GROUPS, HIFIGAN_KERNELS, HIFIGAN_UPSAMPLE, N_MELS,
};
use crate::error::{FishError, Result};
use crate::graph_ir::{Layout, ScalarType, TensorType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FireflyStage {
    ReferenceEncode,
    WaveformDecode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FireflyOp {
    AudioInput,
    CodeInput,
    LogMel,
    FishConv {
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    },
    Conv1x1,
    FishTransConv {
        kernel: usize,
        stride: usize,
    },
    LayerNorm,
    LinearChannelsLast,
    SiLu,
    Gelu,
    LayerScale,
    Add,
    ParallelMean,
    Mask,
    FsqEncode {
        groups: usize,
        quantizers: usize,
    },
    FsqDecode {
        groups: usize,
        quantizers: usize,
    },
    Tanh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireflyTensor {
    pub batch: usize,
    pub channels: usize,
    pub frames: usize,
    pub dtype: ScalarType,
}

impl FireflyTensor {
    fn new(batch: usize, channels: usize, frames: usize, dtype: ScalarType) -> Result<Self> {
        if batch == 0
            || channels == 0
            || frames == 0
            || batch
                .checked_mul(channels)
                .and_then(|v| v.checked_mul(frames))
                .is_none()
        {
            return Err(FishError::InvalidShape("invalid Firefly tensor shape"));
        }
        Ok(Self {
            batch,
            channels,
            frames,
            dtype,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FireflyNode {
    pub name: String,
    pub operation: FireflyOp,
    pub inputs: Vec<usize>,
    pub output: FireflyTensor,
    pub checkpoint_keys: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FireflyGraph {
    pub stage: FireflyStage,
    pub nodes: Vec<FireflyNode>,
    pub weights: BTreeMap<String, TensorType>,
    pub input: usize,
    pub output: usize,
    pub valid_frames: usize,
}

impl FireflyGraph {
    fn new(stage: FireflyStage) -> Self {
        Self {
            stage,
            nodes: Vec::new(),
            weights: BTreeMap::new(),
            input: 0,
            output: 0,
            valid_frames: 0,
        }
    }

    fn tensor(&self, id: usize) -> Result<FireflyTensor> {
        self.nodes
            .get(id)
            .map(|node| node.output)
            .ok_or(FishError::InvalidShape("unbound Firefly graph value"))
    }

    fn add(
        &mut self,
        name: impl Into<String>,
        operation: FireflyOp,
        inputs: &[usize],
        output: FireflyTensor,
        checkpoint_keys: Vec<String>,
    ) -> Result<usize> {
        let name = name.into();
        if name.is_empty()
            || self.nodes.iter().any(|node| node.name == name)
            || inputs.iter().any(|&id| id >= self.nodes.len())
        {
            return Err(FishError::InvalidShape(
                "invalid Firefly DAG name or predecessor",
            ));
        }
        let id = self.nodes.len();
        self.nodes.push(FireflyNode {
            name,
            operation,
            inputs: inputs.to_vec(),
            output,
            checkpoint_keys,
        });
        Ok(id)
    }

    fn parameter(&mut self, key: String, shape: &[usize]) -> Result<String> {
        let ty = TensorType::new(shape.to_vec(), ScalarType::F32, Layout::Plain);
        if shape.is_empty()
            || shape.contains(&0)
            || shape
                .iter()
                .try_fold(1usize, |p, &v| p.checked_mul(v))
                .is_none()
        {
            return Err(FishError::InvalidShape("invalid Firefly weight shape"));
        }
        match self.weights.insert(key.clone(), ty.clone()) {
            None => Ok(key),
            Some(old) if old == ty => Ok(key),
            Some(_) => Err(FishError::InvalidShape(
                "conflicting Firefly weight dimensions",
            )),
        }
    }

    fn conv(
        &mut self,
        prefix: &str,
        input: usize,
        channels: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<usize> {
        let before = self.tensor(input)?;
        if groups == 0 || before.channels % groups != 0 || channels % groups != 0 {
            return Err(FishError::InvalidShape("invalid Firefly Conv1d groups"));
        }
        let frames = fish_conv_output_len(before.frames, kernel, stride, dilation)?;
        let keys = vec![
            self.parameter(
                format!("{prefix}.conv.weight"),
                &[channels, before.channels / groups, kernel],
            )?,
            self.parameter(format!("{prefix}.conv.bias"), &[channels])?,
        ];
        let output = FireflyTensor::new(before.batch, channels, frames, ScalarType::F32)?;
        self.add(
            prefix,
            FireflyOp::FishConv {
                kernel,
                stride,
                dilation,
                groups,
            },
            &[input],
            output,
            keys,
        )
    }

    fn pointwise_conv(&mut self, prefix: &str, input: usize, channels: usize) -> Result<usize> {
        let before = self.tensor(input)?;
        let keys = vec![
            self.parameter(format!("{prefix}.weight"), &[channels, before.channels, 1])?,
            self.parameter(format!("{prefix}.bias"), &[channels])?,
        ];
        self.add(
            prefix,
            FireflyOp::Conv1x1,
            &[input],
            FireflyTensor::new(before.batch, channels, before.frames, ScalarType::F32)?,
            keys,
        )
    }

    fn transconv(
        &mut self,
        prefix: &str,
        input: usize,
        channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<usize> {
        let before = self.tensor(input)?;
        let frames = fish_transconv_output_len(before.frames, kernel, stride, 1)?;
        let keys = vec![
            self.parameter(
                format!("{prefix}.conv.weight"),
                &[before.channels, channels, kernel],
            )?,
            self.parameter(format!("{prefix}.conv.bias"), &[channels])?,
        ];
        self.add(
            prefix,
            FireflyOp::FishTransConv { kernel, stride },
            &[input],
            FireflyTensor::new(before.batch, channels, frames, ScalarType::F32)?,
            keys,
        )
    }

    fn norm(&mut self, prefix: &str, input: usize) -> Result<usize> {
        let before = self.tensor(input)?;
        let keys = vec![
            self.parameter(format!("{prefix}.weight"), &[before.channels])?,
            self.parameter(format!("{prefix}.bias"), &[before.channels])?,
        ];
        self.add(prefix, FireflyOp::LayerNorm, &[input], before, keys)
    }

    fn linear(&mut self, prefix: &str, input: usize, channels: usize) -> Result<usize> {
        let before = self.tensor(input)?;
        let keys = vec![
            self.parameter(format!("{prefix}.weight"), &[channels, before.channels])?,
            self.parameter(format!("{prefix}.bias"), &[channels])?,
        ];
        self.add(
            prefix,
            FireflyOp::LinearChannelsLast,
            &[input],
            FireflyTensor::new(before.batch, channels, before.frames, ScalarType::F32)?,
            keys,
        )
    }

    fn unary(&mut self, name: impl Into<String>, op: FireflyOp, input: usize) -> Result<usize> {
        self.add(name, op, &[input], self.tensor(input)?, Vec::new())
    }

    fn add_two(&mut self, name: impl Into<String>, left: usize, right: usize) -> Result<usize> {
        let ty = self.tensor(left)?;
        if self.tensor(right)? != ty {
            return Err(FishError::InvalidShape("Firefly residual shape mismatch"));
        }
        self.add(name, FireflyOp::Add, &[left, right], ty, Vec::new())
    }

    fn convnext(&mut self, prefix: &str, input: usize, kernel: usize) -> Result<usize> {
        let ty = self.tensor(input)?;
        let channels = ty.channels;
        let dw = self.conv(
            &format!("{prefix}.dwconv"),
            input,
            channels,
            kernel,
            1,
            1,
            channels,
        )?;
        let normalized = self.norm(&format!("{prefix}.norm"), dw)?;
        let expanded_channels = channels.checked_mul(4).ok_or(FishError::InvalidShape(
            "ConvNeXt projection extent overflow",
        ))?;
        let expanded = self.linear(&format!("{prefix}.pwconv1"), normalized, expanded_channels)?;
        let activated = self.unary(format!("{prefix}.gelu"), FireflyOp::Gelu, expanded)?;
        let shrunk = self.linear(&format!("{prefix}.pwconv2"), activated, channels)?;
        let gamma = self.parameter(format!("{prefix}.gamma"), &[channels])?;
        let scaled = self.add(
            format!("{prefix}.layer_scale"),
            FireflyOp::LayerScale,
            &[shrunk],
            ty,
            vec![gamma],
        )?;
        self.add_two(format!("{prefix}.residual"), input, scaled)
    }

    fn fsq(&mut self, input: usize, encode: bool) -> Result<usize> {
        let before = self.tensor(input)?;
        let quantizers = 1;
        if !encode && before.channels != FIREFLY_GROUPS * quantizers {
            return Err(FishError::InvalidShape(
                "incorrect FSQ codebook channel count",
            ));
        }
        if encode && before.channels % FIREFLY_GROUPS != 0 {
            return Err(FishError::InvalidShape(
                "FSQ encoded feature width not divisible by groups",
            ));
        }
        let features = if encode { before.channels } else { 512 };
        let width = features / FIREFLY_GROUPS;
        let mut keys = Vec::new();
        for group in 0..FIREFLY_GROUPS {
            let prefix = format!(
                "quantizer.residual_fsq.rvqs.{group}.project_{}",
                if encode { "in" } else { "out" }
            );
            let (out, input_channels) = if encode { (4, width) } else { (width, 4) };
            keys.push(self.parameter(format!("{prefix}.weight"), &[out, input_channels])?);
            keys.push(self.parameter(format!("{prefix}.bias"), &[out])?);
        }
        self.add(
            if encode {
                "quantizer.fsq_encode"
            } else {
                "quantizer.fsq_decode"
            },
            if encode {
                FireflyOp::FsqEncode {
                    groups: FIREFLY_GROUPS,
                    quantizers,
                }
            } else {
                FireflyOp::FsqDecode {
                    groups: FIREFLY_GROUPS,
                    quantizers,
                }
            },
            &[input],
            FireflyTensor::new(
                before.batch,
                if encode {
                    FIREFLY_GROUPS * quantizers
                } else {
                    features
                },
                before.frames,
                if encode {
                    ScalarType::I32
                } else {
                    ScalarType::F32
                },
            )?,
            keys,
        )
    }

    pub fn reference_encode(batch: usize, samples: usize) -> Result<Self> {
        let lengths = firefly_reference_lengths(samples)?;
        let mut graph = Self::new(FireflyStage::ReferenceEncode);
        let audio = graph.add(
            "audio",
            FireflyOp::AudioInput,
            &[],
            FireflyTensor::new(batch, 1, samples, ScalarType::F32)?,
            Vec::new(),
        )?;
        graph.input = audio;
        let mut x = graph.add(
            "spec_transform.logmel",
            FireflyOp::LogMel,
            &[audio],
            FireflyTensor::new(batch, N_MELS, lengths.mel_frames, ScalarType::F32)?,
            Vec::new(),
        )?;
        x = graph.unary("reference.mel_mask", FireflyOp::Mask, x)?;
        let backbone_dims = [128, 256, 384, 512];
        let backbone_depths = [3, 3, 9, 3];
        x = graph.conv(
            "backbone.downsample_layers.0.0",
            x,
            backbone_dims[0],
            7,
            1,
            1,
            1,
        )?;
        x = graph.norm("backbone.downsample_layers.0.1", x)?;
        for stage in 0..backbone_dims.len() {
            if stage > 0 {
                x = graph.norm(&format!("backbone.downsample_layers.{stage}.0"), x)?;
                x = graph.pointwise_conv(
                    &format!("backbone.downsample_layers.{stage}.1"),
                    x,
                    backbone_dims[stage],
                )?;
            }
            for block in 0..backbone_depths[stage] {
                x = graph.convnext(&format!("backbone.stages.{stage}.{block}"), x, 7)?;
            }
        }
        x = graph.norm("backbone.norm", x)?;
        x = graph.unary("reference.feature_mask", FireflyOp::Mask, x)?;
        for stage in 0..2 {
            x = graph.conv(
                &format!("quantizer.downsample.{stage}.0"),
                x,
                512,
                2,
                2,
                1,
                1,
            )?;
            x = graph.convnext(&format!("quantizer.downsample.{stage}.1"), x, 7)?;
        }
        x = graph.fsq(x, true)?;
        if graph.tensor(x)?.frames != lengths.allocated_code_frames {
            return Err(FishError::InvalidShape(
                "Firefly encode temporal profile mismatch",
            ));
        }
        graph.output = x;
        graph.valid_frames = lengths.valid_code_frames;
        graph.validate()?;
        Ok(graph)
    }

    fn resblock(&mut self, prefix: &str, input: usize, kernel: usize) -> Result<usize> {
        let mut current = input;
        for (step, dilation) in [1, 3, 5].into_iter().enumerate() {
            let a = self.unary(format!("{prefix}.silu1.{step}"), FireflyOp::SiLu, current)?;
            let b = self.conv(
                &format!("{prefix}.convs1.{step}"),
                a,
                self.tensor(a)?.channels,
                kernel,
                1,
                dilation,
                1,
            )?;
            let c = self.unary(format!("{prefix}.silu2.{step}"), FireflyOp::SiLu, b)?;
            let d = self.conv(
                &format!("{prefix}.convs2.{step}"),
                c,
                self.tensor(c)?.channels,
                kernel,
                1,
                dilation,
                1,
            )?;
            current = self.add_two(format!("{prefix}.residual.{step}"), current, d)?;
        }
        Ok(current)
    }

    fn parallel_resblocks(&mut self, prefix: &str, input: usize) -> Result<usize> {
        let outputs = [3, 7, 11]
            .into_iter()
            .enumerate()
            .map(|(index, kernel)| {
                self.resblock(&format!("{prefix}.blocks.{index}"), input, kernel)
            })
            .collect::<Result<Vec<_>>>()?;
        let ty = self.tensor(outputs[0])?;
        if outputs.iter().any(|&id| self.tensor(id).ok() != Some(ty)) {
            return Err(FishError::InvalidShape(
                "parallel HiFiGAN block shape mismatch",
            ));
        }
        self.add(
            format!("{prefix}.mean"),
            FireflyOp::ParallelMean,
            &outputs,
            ty,
            Vec::new(),
        )
    }

    pub fn waveform_decode(batch: usize, code_frames: usize, valid_frames: usize) -> Result<Self> {
        let lengths = firefly_decode_lengths(code_frames, valid_frames)?;
        let mut graph = Self::new(FireflyStage::WaveformDecode);
        let codes = graph.add(
            "codes",
            FireflyOp::CodeInput,
            &[],
            FireflyTensor::new(batch, FIREFLY_GROUPS, code_frames, ScalarType::I32)?,
            Vec::new(),
        )?;
        graph.input = codes;
        let mut x = graph.fsq(codes, false)?;
        for stage in 0..2 {
            x = graph.transconv(&format!("quantizer.upsample.{stage}.0"), x, 512, 2, 2)?;
            x = graph.convnext(&format!("quantizer.upsample.{stage}.1"), x, 7)?;
        }
        x = graph.unary("waveform.quantized_mel_mask", FireflyOp::Mask, x)?;
        x = graph.conv("head.conv_pre", x, 512, 13, 1, 1, 1)?;
        for (stage, (&stride, &kernel)) in HIFIGAN_UPSAMPLE
            .iter()
            .zip(HIFIGAN_KERNELS.iter())
            .enumerate()
        {
            x = graph.unary(format!("head.ups.{stage}.silu"), FireflyOp::SiLu, x)?;
            let output_channels = 512 >> (stage + 1);
            x = graph.transconv(
                &format!("head.ups.{stage}"),
                x,
                output_channels,
                kernel,
                stride,
            )?;
            x = graph.parallel_resblocks(&format!("head.resblocks.{stage}"), x)?;
        }
        x = graph.unary("head.activation_post", FireflyOp::SiLu, x)?;
        x = graph.conv("head.conv_post", x, 1, 13, 1, 1, 1)?;
        x = graph.unary("head.tanh", FireflyOp::Tanh, x)?;
        x = graph.unary("waveform.audio_mask", FireflyOp::Mask, x)?;
        if graph.tensor(x)?.frames != lengths.allocated_audio_samples {
            return Err(FishError::InvalidShape(
                "Firefly decode temporal profile mismatch",
            ));
        }
        graph.output = x;
        graph.valid_frames = lengths.valid_audio_samples;
        graph.validate()?;
        Ok(graph)
    }

    pub fn validate(&self) -> Result<()> {
        if self.nodes.is_empty()
            || self.output >= self.nodes.len()
            || self.input >= self.nodes.len()
        {
            return Err(FishError::InvalidShape("empty or unbound Firefly DAG"));
        }
        let mut names = BTreeSet::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if !names.insert(&node.name)
                || node.inputs.iter().any(|&input| input >= index)
                || node.output.batch == 0
                || node.output.channels == 0
                || node.output.frames == 0
                || node
                    .checkpoint_keys
                    .iter()
                    .any(|key| !self.weights.contains_key(key))
            {
                return Err(FishError::InvalidShape(
                    "invalid Firefly node or dangling checkpoint binding",
                ));
            }
        }
        let output = self.tensor(self.output)?;
        if self.valid_frames > output.frames || self.valid_frames == 0 {
            return Err(FishError::InvalidShape("invalid Firefly valid-frame mask"));
        }
        if !matches!(
            self.nodes[self.input].operation,
            FireflyOp::AudioInput | FireflyOp::CodeInput
        ) {
            return Err(FishError::InvalidShape(
                "Firefly input is not a graph input",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_reference_encode_wires_backbone_and_fsq_weights() {
        let graph = FireflyGraph::reference_encode(1, 12_345).unwrap();
        let output = graph.tensor(graph.output).unwrap();
        assert_eq!(
            (output.channels, output.frames, output.dtype),
            (8, 6, ScalarType::I32)
        );
        assert_eq!(graph.valid_frames, (12_345 / 512) / 4);
        assert!(graph.nodes.len() > 140, "nodes={}", graph.nodes.len());
        assert!(graph.weights.len() > 200);
        assert_eq!(
            graph.weights["backbone.stages.2.8.pwconv1.weight"].shape,
            vec![1536, 384]
        );
        assert_eq!(
            graph.weights["quantizer.residual_fsq.rvqs.7.project_in.weight"].shape,
            vec![4, 64]
        );
        assert_eq!(
            graph.weights["quantizer.downsample.1.0.conv.weight"].shape,
            vec![512, 512, 2]
        );
        graph.validate().unwrap();
    }

    #[test]
    fn full_waveform_decode_wires_hifigan_and_fsqs() {
        let graph = FireflyGraph::waveform_decode(1, 7, 6).unwrap();
        let output = graph.tensor(graph.output).unwrap();
        assert_eq!(
            (output.channels, output.frames, output.dtype),
            (1, 7 * 2048, ScalarType::F32)
        );
        assert_eq!(graph.valid_frames, 6 * 2048);
        assert!(graph.nodes.len() > 240);
        assert!(graph.weights.len() > 200);
        assert_eq!(
            graph.weights["head.ups.0.conv.weight"].shape,
            vec![512, 256, 16]
        );
        assert_eq!(
            graph.weights["head.resblocks.4.blocks.2.convs1.2.conv.weight"].shape,
            vec![16, 16, 11]
        );
        assert_eq!(
            graph.weights["quantizer.residual_fsq.rvqs.7.project_out.weight"].shape,
            vec![64, 4]
        );
        assert!(FireflyGraph::waveform_decode(1, 7, 8).is_err());
        graph.validate().unwrap();
    }

    #[test]
    fn decoder_shape_profiles_at_different_frame_counts() {
        for frames in [1, 2, 3, 11, 31, 128] {
            let graph = FireflyGraph::waveform_decode(2, frames, frames).unwrap();
            assert_eq!(graph.tensor(graph.output).unwrap().frames, frames * 2048);
            assert_eq!(graph.tensor(graph.output).unwrap().batch, 2);
        }
    }
}
