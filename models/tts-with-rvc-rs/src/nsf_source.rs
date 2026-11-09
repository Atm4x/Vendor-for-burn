use burn::tensor::{activation, Device, Tensor, TensorData};
use serde::Deserialize;

use crate::checkpoint::RvcCheckpoint;

/// The original RVC SineGen with explicit random inputs for numerical parity.
pub struct SineGen {
    sample_rate: f32,
    harmonics: usize,
    sine_amp: f32,
    noise_std: f32,
    voiced_threshold: f32,
}

pub struct SineOutput {
    pub waves: Tensor<3>,
    pub voiced: Tensor<3>,
    pub noise: Tensor<3>,
}

impl SineGen {
    pub fn new(sample_rate: usize, harmonics: usize) -> Result<Self, String> {
        if sample_rate == 0 || harmonics != 0 {
            return Err("this RVC generator supports only harmonic_num=0 (the original multi-harmonic branch is not executable)".into());
        }
        Ok(Self {
            sample_rate: sample_rate as f32,
            harmonics,
            sine_amp: 0.1,
            noise_std: 0.003,
            voiced_threshold: 0.0,
        })
    }

    pub fn forward_with_random(
        &self,
        f0: Tensor<2>,
        upsample: usize,
        initial_phases: Tensor<3>,
        gaussian_noise: Tensor<3>,
    ) -> Result<SineOutput, String> {
        let [batch, frames] = f0.dims();
        let dimension = self.harmonics + 1;
        if upsample == 0 || frames == 0 || batch == 0 {
            return Err("SineGen requires nonempty F0 and a positive upsample".into());
        }
        let samples = frames.checked_mul(upsample).ok_or("NSF length overflow")?;
        if initial_phases.dims() != [1, 1, dimension]
            || gaussian_noise.dims() != [batch, samples, dimension]
        {
            return Err("SineGen random phase or noise dimensions do not match F0".into());
        }
        let device = f0.device();
        let upsample_factors: Vec<f32> = (1..=upsample).map(|i| i as f32).collect();
        let factor = Tensor::<3>::from_data(
            TensorData::new(upsample_factors, [1, 1, upsample]), &device,
        );
        let rad = (f0.clone().unsqueeze_dim::<3>(2) / self.sample_rate) * factor;
        let accumulated = if frames > 1 {
            let final_step = rad.clone().slice([0..batch, 0..frames - 1, upsample - 1..upsample]);
            let wrapped = (final_step + 0.5).fmod_scalar(1.0) - 0.5;
            let prefix = wrapped.cumsum(1).fmod_scalar(1.0);
            Tensor::cat(vec![Tensor::zeros([batch, 1, 1], &device), prefix], 1)
        } else {
            Tensor::zeros([batch, 1, 1], &device)
        };
        let rad = (rad + accumulated).reshape([batch, samples, 1]);
        let harmonics = Tensor::<3>::from_data(
            TensorData::new((1..=dimension).map(|i| i as f32).collect::<Vec<_>>(), [1, 1, dimension]),
            &device,
        );
        let phase_mask = Tensor::<3>::from_data(
            TensorData::new((0..dimension).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect::<Vec<f32>>(), [1, 1, dimension]),
            &device,
        );
        let sine = ((rad * harmonics + initial_phases * phase_mask) * (2.0 * std::f32::consts::PI)).sin() * self.sine_amp;
        let voiced_frames = f0.greater_elem(self.voiced_threshold).float().unsqueeze_dim::<4>(2);
        let voiced = (voiced_frames * Tensor::<4>::ones([1, 1, upsample, 1], &device))
            .reshape([batch, samples, 1]);
        let noise_amplitude = voiced.clone() * self.noise_std
            + (1.0 - voiced.clone()) * (self.sine_amp / 3.0);
        let noise: Tensor<3> = gaussian_noise * noise_amplitude;
        let waves = sine * voiced.clone() + noise.clone();
        Ok(SineOutput { waves, voiced, noise })
    }
}

pub struct SourceModuleHnNSF {
    generator: SineGen,
    weight: Tensor<2>,
    bias: Tensor<1>,
}

impl SourceModuleHnNSF {
    pub fn load(
        checkpoint: &RvcCheckpoint,
        sample_rate: usize,
        harmonics: usize,
        device: &Device,
    ) -> Result<Self, String> {
        let generator = SineGen::new(sample_rate, harmonics)?;
        let weight = checkpoint.tensor::<2>("dec.m_source.l_linear.weight", device)?;
        let bias = checkpoint.tensor::<1>("dec.m_source.l_linear.bias", device)?;
        if weight.dims() != [1, harmonics + 1] || bias.dims() != [1] {
            return Err("NSF source linear layer does not match harmonic count".into());
        }
        Ok(Self { generator, weight, bias })
    }

    pub fn forward_with_random(
        &self,
        f0: Tensor<2>,
        upsample: usize,
        initial_phases: Tensor<3>,
        gaussian_noise: Tensor<3>,
    ) -> Result<(Tensor<3>, SineOutput), String> {
        let output = self.generator.forward_with_random(f0, upsample, initial_phases, gaussian_noise)?;
        let [batch, time, dimensions] = output.waves.dims();
        let merged = activation::tanh(
            output.waves.clone().reshape([batch * time, dimensions])
                .matmul(self.weight.clone().transpose())
                .reshape([batch, time, 1])
                + self.bias.clone().reshape([1, 1, 1]),
        );
        Ok((merged, output))
    }
}

#[derive(Debug, Deserialize)]
pub struct NsfSourceFixture {
    pub sample_rate: usize,
    pub harmonic_num: usize,
    pub upsample: usize,
    pub f0_shape: [usize; 2],
    pub f0: Vec<f32>,
    pub initial_phases: Vec<f32>,
    pub gaussian_noise: Vec<f32>,
    pub expected_sines: Vec<f32>,
    pub expected_voiced: Vec<f32>,
    pub expected_noise: Vec<f32>,
    pub expected_merged: Vec<f32>,
}

impl NsfSourceFixture {
    pub fn run(&self, checkpoint: &RvcCheckpoint, device: &Device) -> Result<(f32, f32, f32, f32), String> {
        let [batch, frames] = self.f0_shape;
        let dims = self.harmonic_num + 1;
        if self.f0.len() != batch * frames || self.initial_phases.len() != dims
            || self.gaussian_noise.len() != batch * frames * self.upsample * dims
        {
            return Err("NSF fixture has inconsistent dimensions".into());
        }
        let model = SourceModuleHnNSF::load(checkpoint, self.sample_rate, self.harmonic_num, device)?;
        let f0 = Tensor::from_data(TensorData::new(self.f0.clone(), self.f0_shape), device);
        let phases = Tensor::from_data(TensorData::new(self.initial_phases.clone(), [1, 1, dims]), device);
        let noise = Tensor::from_data(TensorData::new(self.gaussian_noise.clone(), [batch, frames * self.upsample, dims]), device);
        let (merged, output) = model.forward_with_random(f0, self.upsample, phases, noise)?;
        let err = |a: Tensor<3>, b: &[f32]| -> Result<f32, String> {
            let x = a.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
            if x.len() != b.len() || x.iter().any(|v| !v.is_finite()) {
                return Err("NSF fixture output dimension or values invalid".into());
            }
            Ok(crate::max_abs_error(&x, b))
        };
        Ok((
            err(output.waves, &self.expected_sines)?,
            err(output.voiced, &self.expected_voiced)?,
            err(output.noise, &self.expected_noise)?,
            err(merged, &self.expected_merged)?,
        ))
    }
}

#[derive(Debug, Deserialize)]
pub struct NsfFromF0Fixture {
    pub sample_rate: usize,
    pub upsample_rates: Vec<usize>,
    pub kernels_per_stage: usize,
    pub dilations: [usize; 3],
    pub input_shape: [usize; 3],
    pub condition_shape: Option<[usize; 3]>,
    pub input: Vec<f32>,
    pub condition: Option<Vec<f32>>,
    pub f0: Vec<f32>,
    pub initial_phases: Vec<f32>,
    pub gaussian_noise: Vec<f32>,
    pub expected_waveform: Vec<f32>,
}

impl NsfFromF0Fixture {
    pub fn run(&self, checkpoint: &RvcCheckpoint, device: &Device) -> Result<f32, String> {
        let [batch, _, frames] = self.input_shape;
        let upsample: usize = self.upsample_rates.iter().product();
        if self.f0.len() != batch * frames || self.initial_phases.len() != 1
            || self.gaussian_noise.len() != batch * frames * upsample
        {
            return Err("NSF waveform fixture F0/noise dimensions are inconsistent".into());
        }
        let decoder = crate::decoder::NsfWaveformDecoder::load(
            checkpoint, &self.upsample_rates, self.kernels_per_stage, self.dilations, device,
        )?;
        let source = SourceModuleHnNSF::load(checkpoint, self.sample_rate, 0, device)?;
        let x = Tensor::from_data(TensorData::new(self.input.clone(), self.input_shape), device);
        let f0 = Tensor::from_data(TensorData::new(self.f0.clone(), [batch, frames]), device);
        let phases = Tensor::from_data(TensorData::new(self.initial_phases.clone(), [1, 1, 1]), device);
        let noise = Tensor::from_data(TensorData::new(self.gaussian_noise.clone(), [batch, frames * upsample, 1]), device);
        let condition = self.condition.as_ref().map(|values| {
            Tensor::from_data(TensorData::new(values.clone(), self.condition_shape.expect("missing condition shape")), device)
        });
        let output = decoder.forward_from_f0(x, f0, condition, &source, phases, noise)?;
        if output.dims() != [batch, 1, frames * upsample] {
            return Err("NSF source-integrated decoder produced incorrect dimensions".into());
        }
        let actual = output.into_data().try_into_vec::<f32>().map_err(|error| error.to_string())?;
        if actual.len() != self.expected_waveform.len() || actual.iter().any(|value| !value.is_finite()) {
            return Err("NSF decoder generated unexpected waveform values".into());
        }
        Ok(crate::max_abs_error(&actual, &self.expected_waveform))
    }
}
