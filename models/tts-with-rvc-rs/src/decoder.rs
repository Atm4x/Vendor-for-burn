use burn::tensor::{activation, module, ops::{ConvOptions, ConvTransposeOptions}, Device, Tensor, TensorData};
use serde::Deserialize;
use rayon::prelude::*;
use std::time::Instant;

use crate::{checkpoint::RvcCheckpoint, conv1d_layout::{Conv1dWeightLayout, default_conv1d_layout, load_conv1d_weight}};
use crate::nsf_source::SourceModuleHnNSF;
use crate::nsf_tiling::{ConvGeometry, NsfTileGeometry, StageGeometry};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualExecution {
    Sequential,
    Parallel,
}

impl ResidualExecution {
    pub const PRODUCTION: Self = Self::Sequential;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualAggregation {
    ZeroSeeded,
    FirstBranch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NsfScheduling {
    Default,
    FlushPerBranch,
    FlushPerStage,
    SyncPerBranch,
    FlushPerPair,
    FlushPerConv,
    SyncPerPair,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct NsfProfileStep {
    pub section: String,
    pub milliseconds: f64,
}

struct Conv1d {
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    stride: usize,
    dilation: usize,
    padding: usize,
}

impl Conv1d {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, stride: usize, dilation: usize, padding: usize, device: &Device) -> Result<Self, String> {
        if stride == 0 || dilation == 0 {
            return Err(format!("{prefix}: stride/dilation must be positive"));
        }
        let normalized = checkpoint.contains(&format!("{prefix}.weight_v"));
        let kernel_mode = std::env::var("RVC_NSF_CONV_KERNEL").unwrap_or_default();
        let layout = if matches!(kernel_mode.as_str(), "packed" | "quad" | "quad_interior") {
            Conv1dWeightLayout::Temporal
        } else {
            default_conv1d_layout()
        };
        let weight = load_conv1d_weight(checkpoint, prefix, normalized, layout, device)?;
        let bias = checkpoint.contains(&format!("{prefix}.bias"))
            .then(|| checkpoint.tensor::<1>(&format!("{prefix}.bias"), device))
            .transpose()?;
        if let Some(bias) = &bias {
            if bias.dims()[0] != weight.dims()[0] {
                return Err(format!("{prefix}: bias and weight output channels differ"));
            }
        }
        Ok(Self {weight, bias, stride, dilation, padding})
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let options = ConvOptions::new_with_padding([self.stride], [(self.padding, self.padding)], [self.dilation], 1);
        if std::env::var("RVC_NSF_IMPLICIT_GEMM").as_deref() == Ok("1")
            || std::env::var("RVC_NSF_CONV1D_GEMM").as_deref() == Ok("1")
        {
            crate::cubecl_conv::conv1d(input, self.weight.clone(), self.bias.clone(), options)
        } else {
            module::conv1d(input, self.weight.clone(), self.bias.clone(), options)
        }
    }
}

struct TransposedConv1d {
    weight: Tensor<3>,
    bias: Tensor<1>,
    stride: usize,
    padding: usize,
}

impl TransposedConv1d {
    fn load(checkpoint: &RvcCheckpoint, prefix: &str, stride: usize, device: &Device) -> Result<Self, String> {
        if stride == 0 { return Err(format!("{prefix}: invalid stride")); }
        let weight = checkpoint.weight_norm_conv1d(prefix, device)?;
        let bias = checkpoint.tensor::<1>(&format!("{prefix}.bias"), device)?;
        let [_, output_channels, kernel] = weight.dims();
        if kernel < stride || (kernel - stride) % 2 != 0 || bias.dims()[0] != output_channels {
            return Err(format!("{prefix}: unsupported ConvTranspose1d geometry"));
        }
        Ok(Self {weight, bias, stride, padding: (kernel - stride) / 2})
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        if std::env::var("RVC_NSF_IMPLICIT_GEMM").as_deref() == Ok("1")
            || std::env::var("RVC_NSF_TRANSPOSE_GEMM").as_deref() == Ok("1")
        {
            crate::cubecl_conv::conv_transpose1d(input, self.weight.clone(), self.bias.clone(), self.stride, self.padding)
        } else {
            module::conv_transpose1d(input, self.weight.clone(), Some(self.bias.clone()),
                ConvTransposeOptions::new([self.stride], [self.padding], [0], [1], 1))
        }
    }
}

pub struct ResBlock1 {
    first: Vec<Conv1d>,
    second: Vec<Conv1d>,
    channels: usize,
}

impl ResBlock1 {
    pub fn load(checkpoint: &RvcCheckpoint, index: usize, dilations: [usize; 3], device: &Device) -> Result<Self, String> {
        let prefix = format!("dec.resblocks.{index}");
        let mut first = Vec::with_capacity(dilations.len());
        let mut second = Vec::with_capacity(dilations.len());
        let mut channels = None;
        for (layer, dilation) in dilations.into_iter().enumerate() {
            for (family, dilation, target) in [("convs1", dilation, &mut first), ("convs2", 1, &mut second)] {
                let name = format!("{prefix}.{family}.{layer}");
                let shape = checkpoint.tensor_f32(&format!("{name}.weight_v"))?.0;
                if shape.len() != 3 || shape[0] != shape[1] || shape[2] % 2 == 0 {
                    return Err(format!("{name}: expected same-channel odd-kernel Conv1d"));
                }
                if channels.get_or_insert(shape[0]) != &shape[0] {
                    return Err(format!("{name}: inconsistent channel count"));
                }
                target.push(Conv1d::load(checkpoint, &name, 1, dilation, (shape[2] - 1) * dilation / 2, device)?);
            }
        }
        Ok(Self {first, second, channels: channels.unwrap()})
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_with_scheduling(x, NsfScheduling::Default)
    }

    fn forward_with_scheduling(&self, mut x: Tensor<3>, scheduling: NsfScheduling) -> Tensor<3> {
        assert_eq!(x.dims()[1], self.channels);
        let fused_limit = std::env::var("RVC_NSF_FUSED_PAIR_MAX_CHANNELS").ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|limit| *limit > 0);
        let shared_tile = std::env::var("RVC_NSF_PAIR_SHARED_TILE").as_deref() == Ok("1")
            && scheduling == NsfScheduling::Default;
        for (first, second) in self.first.iter().zip(&self.second) {
            if shared_tile && crate::nsf_pair_tile::supports(self.channels, second.weight.dims()[2], second.dilation) {
                x = crate::nsf_pair_tile::forward(
                    x, first.weight.clone(), first.bias.clone(), first.dilation, first.padding,
                    second.weight.clone(), second.bias.clone(), second.dilation, second.padding,
                ).expect("NSF shared-tile two-convolution kernel failed");
                continue;
            }
            if scheduling == NsfScheduling::Default && fused_limit.is_some_and(|limit| self.channels <= limit) {
                let h = crate::nsf_fused_pair::preact_conv(
                    x.clone(), first.weight.clone(), first.bias.clone(), None,
                    first.dilation, first.padding,
                ).expect("NSF experimental first fused convolution failed");
                x = crate::nsf_fused_pair::preact_conv(
                    h, second.weight.clone(), second.bias.clone(), Some(x),
                    second.dilation, second.padding,
                ).expect("NSF experimental second fused convolution failed");
                continue;
            }
            let h = activation::leaky_relu(x.clone(), 0.1);
            let h = first.forward(h);
            if scheduling == NsfScheduling::FlushPerConv {
                h.device().flush().expect("NSF first Conv flush failed");
            }
            let h = activation::leaky_relu(h, 0.1);
            let h = second.forward(h);
            if scheduling == NsfScheduling::FlushPerConv {
                h.device().flush().expect("NSF second Conv flush failed");
            }
            x = x + h;
            match scheduling {
                NsfScheduling::FlushPerPair => x.device().flush().expect("NSF pair flush failed"),
                NsfScheduling::SyncPerPair => x.device().sync().expect("NSF pair sync failed"),
                _ => {},
            }
        }
        x
    }

    fn profile_cpu(
        &self,
        mut x: Tensor<3>,
        steps: &mut Vec<NsfProfileStep>,
        stage_index: usize,
        block_index: usize,
    ) -> Tensor<3> {
        let prefix = format!("stage{stage_index}.residual{block_index}");
        for (index, (first, second)) in self.first.iter().zip(&self.second).enumerate() {
            let mut record = |section: &str, tensor: &Tensor<3>, start: Instant| {
                let _ = tensor.clone().into_data();
                steps.push(NsfProfileStep {
                    section: format!("{prefix}.pair{index}.{section}"),
                    milliseconds: start.elapsed().as_secs_f64() * 1000.0,
                });
            };
            let start = Instant::now();
            let h = activation::leaky_relu(x.clone(), 0.1);
            record("first_activation", &h, start);
            let start = Instant::now();
            let h = first.forward(h);
            record("first_conv", &h, start);
            let start = Instant::now();
            let h = activation::leaky_relu(h, 0.1);
            record("second_activation", &h, start);
            let start = Instant::now();
            let h = second.forward(h);
            record("second_conv", &h, start);
            let start = Instant::now();
            x = x + h;
            record("residual_add", &x, start);
        }
        x
    }
}

pub struct DecoderStage {
    upsample: TransposedConv1d,
    noise: Conv1d,
    residual: Vec<ResBlock1>,
}

impl DecoderStage {
    pub fn load(checkpoint: &RvcCheckpoint, stage: usize, stride: usize, future_stride: usize, resblock_kernels: usize, dilations: [usize; 3], device: &Device) -> Result<Self, String> {
        if future_stride == 0 || resblock_kernels == 0 {
            return Err("decoder stage requires a positive source stride and residual count".into());
        }
        let upsample = TransposedConv1d::load(checkpoint, &format!("dec.ups.{stage}"), stride, device)?;
        let name = format!("dec.noise_convs.{stage}");
        let noise_shape = checkpoint.tensor_f32(&format!("{name}.weight"))?.0;
        if noise_shape.len() != 3 || noise_shape[0] != upsample.bias.dims()[0] || noise_shape[1] != 1 {
            return Err(format!("{name}: source channel dimensions do not match upsampling"));
        }
        let noise = Conv1d::load(checkpoint, &name, future_stride, 1, future_stride / 2, device)?;
        let mut residual = Vec::with_capacity(resblock_kernels);
        for i in 0..resblock_kernels {
            let rb = ResBlock1::load(checkpoint, stage * resblock_kernels + i, dilations, device)?;
            if rb.channels != upsample.bias.dims()[0] {
                return Err(format!("stage {stage}: residual channels don't match upsampling"));
            }
            residual.push(rb);
        }
        Ok(Self { upsample, noise, residual })
    }

    pub fn upsample_only(&self, input: Tensor<3>) -> Tensor<3> {
        self.upsample.forward(activation::leaky_relu(input, 0.1))
    }

    pub fn forward(&self, input: Tensor<3>, harmonic_source: Tensor<3>) -> Tensor<3> {
        self.forward_with_execution(input, harmonic_source, ResidualExecution::PRODUCTION)
    }

    fn forward_with_execution(
        &self,
        input: Tensor<3>,
        harmonic_source: Tensor<3>,
        execution: ResidualExecution,
    ) -> Tensor<3> {
        self.forward_with_aggregation(input, harmonic_source, execution, ResidualAggregation::ZeroSeeded)
    }

    fn forward_with_aggregation(
        &self,
        input: Tensor<3>,
        harmonic_source: Tensor<3>,
        execution: ResidualExecution,
        aggregation: ResidualAggregation,
    ) -> Tensor<3> {
        self.forward_with_scheduling(input, harmonic_source, execution, aggregation, NsfScheduling::Default, None)
    }

    fn forward_with_scheduling(
        &self,
        input: Tensor<3>,
        harmonic_source: Tensor<3>,
        execution: ResidualExecution,
        aggregation: ResidualAggregation,
        scheduling: NsfScheduling,
        diagnostic_stage: Option<usize>,
    ) -> Tensor<3> {
        let trace = diagnostic_stage.filter(|_| crate::diagnostic::enabled());
        let device = trace.map(|_| input.device());
        let started = trace.map(|_| Instant::now());
        let upsampled = self.upsample_only(input);
        if let (Some(stage), Some(device), Some(started)) = (trace, &device, started) {
            crate::diagnostic::measure(device, "nsf.stage_upsample", started,
                serde_json::json!({"stage": stage, "shape": upsampled.dims()}));
        }
        let started = trace.map(|_| Instant::now());
        let source = self.noise.forward(harmonic_source);
        if let (Some(stage), Some(device), Some(started)) = (trace, &device, started) {
            crate::diagnostic::measure(device, "nsf.stage_noise_conv", started,
                serde_json::json!({"stage": stage, "shape": source.dims()}));
        }
        assert_eq!(upsampled.dims(), source.dims(), "harmonic source must align with upsampled features");
        let prepared = upsampled + source;
        let residual_started = trace.map(|_| Instant::now());
        let mut result = match aggregation {
            ResidualAggregation::ZeroSeeded => Some(Tensor::<3>::zeros(prepared.dims(), &prepared.device())),
            ResidualAggregation::FirstBranch => None,
        };
        let mut accumulate = |branch: Tensor<3>| {
            result = Some(match result.take() {
                Some(current) => current + branch,
                None => branch,
            });
        };
        match execution {
            ResidualExecution::Sequential => {
                for block in &self.residual {
                    let branch = block.forward_with_scheduling(prepared.clone(), scheduling);
                    match scheduling {
                        NsfScheduling::FlushPerBranch => branch.device().flush().expect("NSF branch flush failed"),
                        NsfScheduling::SyncPerBranch => branch.device().sync().expect("NSF branch sync failed"),
                        _ => {},
                    }
                    accumulate(branch);
                }
            }
            ResidualExecution::Parallel => {
                let branches: Vec<Tensor<3>> = self.residual.par_iter()
                    .map(|block| block.forward_with_scheduling(prepared.clone(), scheduling)).collect();
                for branch in branches {
                    match scheduling {
                        NsfScheduling::FlushPerBranch => branch.device().flush().expect("NSF branch flush failed"),
                        NsfScheduling::SyncPerBranch => branch.device().sync().expect("NSF branch sync failed"),
                        _ => {},
                    }
                    accumulate(branch);
                }
            }
        }
        let result = result.expect("decoder stage must contain residual blocks") / self.residual.len() as f32;
        if scheduling == NsfScheduling::FlushPerStage {
            result.device().flush().expect("NSF stage flush failed");
        }
        if let (Some(stage), Some(device), Some(started)) = (trace, &device, residual_started) {
            crate::diagnostic::measure(device, "nsf.stage_residuals_aggregate", started,
                serde_json::json!({"stage": stage, "residual_count": self.residual.len(), "shape": result.dims()}));
        }
        result
    }

    fn profile_cpu(
        &self,
        input: Tensor<3>,
        harmonic_source: Tensor<3>,
        steps: &mut Vec<NsfProfileStep>,
        stage_index: usize,
        aggregation: ResidualAggregation,
    ) -> Tensor<3> {
        fn record(steps: &mut Vec<NsfProfileStep>, stage_index: usize, label: &str, tensor: &Tensor<3>, start: Instant) {
            let _ = tensor.clone().into_data();
            steps.push(NsfProfileStep {
                section: format!("stage{stage_index}.{label}"),
                milliseconds: start.elapsed().as_secs_f64() * 1000.0,
            });
        }
        let start = Instant::now();
        let upsampled = self.upsample_only(input);
        record(steps, stage_index, "upsample", &upsampled, start);
        let start = Instant::now();
        let source = self.noise.forward(harmonic_source);
        record(steps, stage_index, "noise_conv", &source, start);
        let start = Instant::now();
        let prepared = upsampled + source;
        record(steps, stage_index, "prepare_add", &prepared, start);
        let mut branches = Vec::with_capacity(self.residual.len());
        for (index, block) in self.residual.iter().enumerate() {
            let start = Instant::now();
            let branch = block.profile_cpu(prepared.clone(), steps, stage_index, index);
            steps.push(NsfProfileStep {
                section: format!("stage{stage_index}.residual{index}.total"),
                milliseconds: start.elapsed().as_secs_f64() * 1000.0,
            });
            branches.push(branch);
        }
        let start = Instant::now();
        let mut acc = match aggregation {
            ResidualAggregation::ZeroSeeded => Some(Tensor::<3>::zeros(prepared.dims(), &prepared.device())),
            ResidualAggregation::FirstBranch => None,
        };
        for branch in branches {
            acc = Some(match acc.take() {
                Some(prev) => prev + branch,
                None => branch,
            });
        }
        let result = acc.expect("decoder stage must contain residual blocks") / self.residual.len() as f32;
        record(steps, stage_index, "aggregate", &result, start);
        result
    }

    pub fn residual_count(&self) -> usize { self.residual.len() }
}

#[derive(Debug, Deserialize)]
pub struct DecoderStageFixture {
    pub stage: usize,
    pub stride: usize,
    pub future_stride: usize,
    pub num_kernels: usize,
    pub dilations: [usize; 3],
    pub input_shape: [usize; 3],
    pub source_shape: [usize; 3],
    pub input: Vec<f32>,
    pub harmonic_source: Vec<f32>,
    pub expected_upsampled: Vec<f32>,
    pub expected_output: Vec<f32>,
}

impl DecoderStageFixture {
    pub fn tensors(&self, device: &Device) -> (Tensor<3>, Tensor<3>) {
        let input = Tensor::from_data(TensorData::new(self.input.clone(), self.input_shape), device);
        let source = Tensor::from_data(TensorData::new(self.harmonic_source.clone(), self.source_shape), device);
        (input, source)
    }

    pub fn load(&self, checkpoint: &RvcCheckpoint, device: &Device) -> Result<DecoderStage, String> {
        DecoderStage::load(checkpoint, self.stage, self.stride, self.future_stride, self.num_kernels, self.dilations, device)
    }
}


pub struct NsfWaveformDecoder {
    pre: Conv1d,
    condition: Option<Conv1d>,
    stages: Vec<DecoderStage>,
    post: Conv1d,
    total_upsample: usize,
}

impl NsfWaveformDecoder {
    pub fn load(
        checkpoint: &RvcCheckpoint,
        upsample_rates: &[usize],
        kernels_per_stage: usize,
        dilations: [usize; 3],
        device: &Device,
    ) -> Result<Self, String> {
        if upsample_rates.is_empty() || upsample_rates.iter().any(|rate| *rate == 0) {
            return Err("NSF decoder requires nonzero upsampling rates".into());
        }
        let pre = Conv1d::load(checkpoint, "dec.conv_pre", 1, 1, 3, device)?;
        let condition = checkpoint.contains("dec.cond.weight")
            .then(|| Conv1d::load(checkpoint, "dec.cond", 1, 1, 0, device))
            .transpose()?;
        let post = Conv1d::load(checkpoint, "dec.conv_post", 1, 1, 3, device)?;
        let total_upsample = upsample_rates.iter().try_fold(1_usize, |product, rate| product.checked_mul(*rate))
            .ok_or("NSF upsampling product overflows usize")?;
        let mut stages = Vec::with_capacity(upsample_rates.len());
        for (index, stride) in upsample_rates.iter().copied().enumerate() {
            let future_stride = upsample_rates[index + 1..].iter().product();
            stages.push(DecoderStage::load(checkpoint, index, stride, future_stride, kernels_per_stage, dilations, device)?);
        }
        let channel_count = pre.weight.dims()[0];
        if stages[0].upsample.weight.dims()[0] != channel_count {
            return Err("NSF conv_pre/upsample channels mismatch".into());
        }
        if post.weight.dims()[1] != stages.last().unwrap().upsample.bias.dims()[0]
            || post.weight.dims()[0] != 1
        {
            return Err("NSF conv_post channels mismatch".into());
        }
        if let Some(cond) = &condition {
            if cond.weight.dims()[0] != channel_count || cond.weight.dims()[2] != 1 {
                return Err("NSF speaker conditioning channels mismatch".into());
            }
        }
        Ok(Self { pre, condition, stages, post, total_upsample })
    }

    pub fn forward(&self, input: Tensor<3>, harmonic_source: Tensor<3>, g: Option<Tensor<3>>) -> Tensor<3> {
        self.forward_with_execution(input, harmonic_source, g, ResidualExecution::PRODUCTION)
    }

    pub fn forward_with_execution(
        &self,
        input: Tensor<3>,
        harmonic_source: Tensor<3>,
        g: Option<Tensor<3>>,
        execution: ResidualExecution,
    ) -> Tensor<3> {
        self.forward_with_aggregation(input, harmonic_source, g, execution, ResidualAggregation::ZeroSeeded)
    }

    pub fn forward_with_aggregation(
        &self,
        input: Tensor<3>,
        harmonic_source: Tensor<3>,
        g: Option<Tensor<3>>,
        execution: ResidualExecution,
        aggregation: ResidualAggregation,
    ) -> Tensor<3> {
        self.forward_with_scheduling(input, harmonic_source, g, execution, aggregation, NsfScheduling::Default)
    }

    pub fn forward_with_scheduling(
        &self,
        input: Tensor<3>,
        harmonic_source: Tensor<3>,
        g: Option<Tensor<3>>,
        execution: ResidualExecution,
        aggregation: ResidualAggregation,
        scheduling: NsfScheduling,
    ) -> Tensor<3> {
        let [batch, _, time] = input.dims();
        assert_eq!(harmonic_source.dims(), [batch, 1, time * self.total_upsample]);
        let device = input.device();
        let pre_started = std::time::Instant::now();
        let mut x = self.pre.forward(input);
        match (&self.condition, g) {
            (Some(conv), Some(condition)) => x = x + conv.forward(condition),
            (None, None) => {},
            _ => panic!("NSF decoder conditioning does not match checkpoint"),
        }
        crate::diagnostic::measure(&device, "nsf.conv_pre_condition", pre_started, serde_json::json!({"shape":x.dims()}));
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            let started = std::time::Instant::now();
            x = stage.forward_with_scheduling(x, harmonic_source.clone(), execution, aggregation, scheduling, Some(stage_idx));
            crate::diagnostic::measure(&device, "nsf.upsample_resblocks", started,
                serde_json::json!({"stage":stage_idx,"shape":x.dims()}));
        }
        let started = std::time::Instant::now();
        let output = activation::tanh(self.post.forward(activation::leaky_relu(x, 0.01)));
        crate::diagnostic::measure(&device, "nsf.conv_post", started, serde_json::json!({"shape":output.dims()}));
        output
    }

    pub fn profile_cpu(
        &self,
        input: Tensor<3>,
        harmonic_source: Tensor<3>,
        g: Option<Tensor<3>>,
        aggregation: ResidualAggregation,
    ) -> (Tensor<3>, Vec<NsfProfileStep>) {
        let [batch, _, time] = input.dims();
        assert_eq!(harmonic_source.dims(), [batch, 1, time * self.total_upsample]);
        let mut steps = Vec::new();
        let start = Instant::now();
        let mut x = self.pre.forward(input);
        let _ = x.clone().into_data();
        steps.push(NsfProfileStep { section: "pre_conv".into(), milliseconds: start.elapsed().as_secs_f64() * 1000.0 });
        match (&self.condition, g) {
            (Some(conv), Some(condition)) => {
                let start = Instant::now();
                x = x + conv.forward(condition);
                let _ = x.clone().into_data();
                steps.push(NsfProfileStep { section: "conditioning".into(), milliseconds: start.elapsed().as_secs_f64() * 1000.0 });
            }
            (None, None) => {},
            _ => panic!("NSF decoder conditioning does not match checkpoint"),
        }
        for (index, stage) in self.stages.iter().enumerate() {
            x = stage.profile_cpu(x, harmonic_source.clone(), &mut steps, index, aggregation);
        }
        let start = Instant::now();
        let out = activation::tanh(self.post.forward(activation::leaky_relu(x, 0.01)));
        let _ = out.clone().into_data();
        steps.push(NsfProfileStep { section: "post_conv_activation".into(), milliseconds: start.elapsed().as_secs_f64() * 1000.0 });
        (out, steps)
    }

    pub fn forward_from_f0(
        &self,
        input: Tensor<3>,
        f0: Tensor<2>,
        condition: Option<Tensor<3>>,
        source: &SourceModuleHnNSF,
        initial_phases: Tensor<3>,
        gaussian_noise: Tensor<3>,
    ) -> Result<Tensor<3>, String> {
        self.forward_from_f0_with_execution(
            input, f0, condition, source, initial_phases, gaussian_noise, ResidualExecution::PRODUCTION,
        )
    }

    pub fn forward_from_f0_with_execution(
        &self,
        input: Tensor<3>,
        f0: Tensor<2>,
        condition: Option<Tensor<3>>,
        source: &SourceModuleHnNSF,
        initial_phases: Tensor<3>,
        gaussian_noise: Tensor<3>,
        execution: ResidualExecution,
    ) -> Result<Tensor<3>, String> {
        let [batch, _, frames] = input.dims();
        if f0.dims() != [batch, frames] {
            return Err("NSF decoder F0 shape must match input frames".into());
        }
        let device = f0.device();
        let started = std::time::Instant::now();
        let (harmonic, _) = source.forward_with_random(f0, self.total_upsample, initial_phases, gaussian_noise)?;
        crate::diagnostic::measure(&device, "nsf.source_generation", started, serde_json::json!({"harmonic_shape":harmonic.dims()}));
        let started = std::time::Instant::now();
        let harmonic = harmonic.permute([0, 2, 1]);
        let output = match std::env::var("RVC_NSF_TILE_CORE_FRAMES") {
            Ok(frames) if !frames.is_empty() => {
                let core = frames.parse::<usize>()
                    .map_err(|_| "RVC_NSF_TILE_CORE_FRAMES must be a positive integer".to_string())?;
                if execution != ResidualExecution::PRODUCTION {
                    return Err("NSF tiling requires sequential production residual execution".into());
                }
                self.forward_tiled(input, harmonic, condition, core)?
            }
            _ => self.forward_with_execution(input, harmonic, condition, execution),
        };
        crate::diagnostic::measure(&device, "nsf.decoder_all", started, serde_json::json!({"output_shape":output.dims()}));
        Ok(output)
    }

    pub fn tiling_geometry(&self) -> NsfTileGeometry {
        let conv_geometry = |conv: &Conv1d| ConvGeometry {
            kernel: conv.weight.dims()[2],
            stride: conv.stride,
            dilation: conv.dilation,
            padding: conv.padding,
        };
        NsfTileGeometry {
            pre: conv_geometry(&self.pre),
            post: conv_geometry(&self.post),
            stages: self.stages.iter().map(|stage| {
                let max_residual_radius = stage.residual.iter().map(|block| {
                    block.first.iter().zip(&block.second).map(|(first, second)| {
                        (first.weight.dims()[2] - 1) * first.dilation / 2
                            + (second.weight.dims()[2] - 1) * second.dilation / 2
                    }).sum::<usize>()
                }).max().unwrap_or(0);
                StageGeometry {
                    upsample: ConvGeometry {
                        kernel: stage.upsample.weight.dims()[2],
                        stride: stage.upsample.stride,
                        dilation: 1,
                        padding: stage.upsample.padding,
                    },
                    noise: conv_geometry(&stage.noise),
                    max_residual_radius,
                }
            }).collect(),
            total_upsample: self.total_upsample,
        }
    }

    pub fn forward_tiled(
        &self,
        input: Tensor<3>,
        harmonic_source: Tensor<3>,
        condition: Option<Tensor<3>>,
        core_frames: usize,
    ) -> Result<Tensor<3>, String> {
        let [batch, channels, frames] = input.dims();
        let total_samples = frames.checked_mul(self.total_upsample)
            .ok_or("NSF tiled sample count overflow")?;
        if harmonic_source.dims() != [batch, 1, total_samples] {
            return Err("NSF tiled harmonic input has unexpected shape".into());
        }
        if core_frames == 0 {
            return Err("NSF tiled core frames must be positive".into());
        }
        let plan = self.tiling_geometry().plan(frames, core_frames)?;
        let mut parts = Vec::with_capacity(plan.len());
        for part in plan {
            let latent = input.clone().slice([0..batch, 0..channels, part.input_start..part.input_end]);
            let source_start = part.input_start * self.total_upsample;
            let source_end = part.input_end * self.total_upsample;
            let harmonic = harmonic_source.clone().slice([0..batch, 0..1, source_start..source_end]);
            let tile_condition = match &condition {
                Some(g) if g.dims()[0] != batch => {
                    return Err("NSF tiled conditioning batch dimensions differ".into());
                }
                Some(g) if g.dims()[2] == 1 => Some(g.clone()),
                Some(g) if g.dims()[2] == frames => Some(g.clone().slice([
                    0..batch, 0..g.dims()[1], part.input_start..part.input_end,
                ])),
                Some(_) => return Err("NSF tiled conditioning must be broadcast or time-aligned".into()),
                None => None,
            };
            let full_tile = self.forward_with_execution(
                latent, harmonic, tile_condition, ResidualExecution::PRODUCTION,
            );
            let samples = (part.input_end - part.input_start) * self.total_upsample;
            if full_tile.dims() != [batch, 1, samples] {
                return Err("NSF tiled decoder output has unexpected shape".into());
            }
            parts.push(full_tile.slice([0..batch, 0..1, part.output_start..part.output_end]));
        }
        Ok(Tensor::cat(parts, 2))
    }

    pub fn stage_count(&self) -> usize { self.stages.len() }
}

#[derive(Debug, Deserialize)]
pub struct NsfDecoderFixture {
    pub upsample_rates: Vec<usize>,
    pub kernels_per_stage: usize,
    pub dilations: [usize; 3],
    pub input_shape: [usize; 3],
    pub source_shape: [usize; 3],
    pub condition_shape: Option<[usize; 3]>,
    pub input: Vec<f32>,
    pub harmonic_source: Vec<f32>,
    pub condition: Option<Vec<f32>>,
    pub expected_waveform: Vec<f32>,
}

impl NsfDecoderFixture {
    pub fn tensors(&self, device: &Device) -> (Tensor<3>, Tensor<3>, Option<Tensor<3>>) {
        let input = Tensor::from_data(TensorData::new(self.input.clone(), self.input_shape), device);
        let harmonic = Tensor::from_data(TensorData::new(self.harmonic_source.clone(), self.source_shape), device);
        let condition = self.condition.as_ref().map(|values| {
            Tensor::from_data(TensorData::new(values.clone(), self.condition_shape.expect("conditioning tensor missing shape")), device)
        });
        (input, harmonic, condition)
    }

    pub fn load(&self, checkpoint: &RvcCheckpoint, device: &Device) -> Result<NsfWaveformDecoder, String> {
        NsfWaveformDecoder::load(checkpoint, &self.upsample_rates, self.kernels_per_stage, self.dilations, device)
    }
}

#[cfg(test)]
mod production_execution_tests {
    use super::ResidualExecution;

    #[test]
    fn production_residuals_are_sequential() {
        assert_eq!(ResidualExecution::PRODUCTION, ResidualExecution::Sequential);
    }
}
