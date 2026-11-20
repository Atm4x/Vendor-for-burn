use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::{
    F5Error, Result,
    cfm::{CfmSampleMode, OdeMethod, SamplingConfig},
};

pub trait DiTVelocity {
    fn predict(
        &mut self,
        noisy_mel: Tensor<3>,
        reference_mel: Tensor<3>,
        text_tokens: Tensor<2, Int>,
        time: f32,
        cfg_strength: f32,
    ) -> Result<Tensor<3>>;

    fn release_request_cache(&mut self);
}

pub fn integrate<N: DiTVelocity>(
    network: &mut N,
    noise: Tensor<3>,
    reference_mel: Tensor<3>,
    text_tokens: Tensor<2, Int>,
    reference_frames: usize,
    sampling: SamplingConfig,
) -> Result<Tensor<3>> {
    integrate_with_channels(
        network,
        noise,
        reference_mel,
        text_tokens,
        reference_frames,
        sampling,
        100,
    )
}

pub fn integrate_with_edit_mask<N: DiTVelocity>(
    network: &mut N,
    noise: Tensor<3>,
    original_mel: Tensor<3>,
    text_tokens: Tensor<2, Int>,
    keep_frames: &[bool],
    sampling: SamplingConfig,
) -> Result<Tensor<3>> {
    integrate_with_strategy(
        network,
        noise,
        original_mel,
        text_tokens,
        RestoreStrategy::FrameMask(keep_frames),
        sampling,
        100,
        OdeMethod::Euler,
        CfmSampleMode::default(),
    )
}

pub fn integrate_with_ode_method<N: DiTVelocity>(
    network: &mut N,
    noise: Tensor<3>,
    reference_mel: Tensor<3>,
    text_tokens: Tensor<2, Int>,
    reference_frames: usize,
    sampling: SamplingConfig,
    method: OdeMethod,
) -> Result<Tensor<3>> {
    integrate_with_strategy(
        network,
        noise,
        reference_mel,
        text_tokens,
        RestoreStrategy::Prefix(reference_frames),
        sampling,
        100,
        method,
        CfmSampleMode::default(),
    )
}

pub fn integrate_with_edit_mask_ode<N: DiTVelocity>(
    network: &mut N,
    noise: Tensor<3>,
    original_mel: Tensor<3>,
    text_tokens: Tensor<2, Int>,
    keep_frames: &[bool],
    sampling: SamplingConfig,
    method: OdeMethod,
) -> Result<Tensor<3>> {
    integrate_with_strategy(
        network,
        noise,
        original_mel,
        text_tokens,
        RestoreStrategy::FrameMask(keep_frames),
        sampling,
        100,
        method,
        CfmSampleMode::default(),
    )
}

pub fn integrate_with_sample_mode<N: DiTVelocity>(
    network: &mut N,
    noise: Tensor<3>,
    reference_mel: Tensor<3>,
    text_tokens: Tensor<2, Int>,
    reference_frames: usize,
    sampling: SamplingConfig,
    method: OdeMethod,
    mode: CfmSampleMode,
) -> Result<Tensor<3>> {
    integrate_with_strategy(
        network, noise, reference_mel, text_tokens,
        RestoreStrategy::Prefix(reference_frames), sampling, 100, method, mode,
    )
}

pub(crate) fn integrate_with_channels<N: DiTVelocity>(
    network: &mut N,
    noise: Tensor<3>,
    reference_mel: Tensor<3>,
    text_tokens: Tensor<2, Int>,
    reference_frames: usize,
    sampling: SamplingConfig,
    mel_channels: usize,
) -> Result<Tensor<3>> {
    integrate_with_strategy(
        network,
        noise,
        reference_mel,
        text_tokens,
        RestoreStrategy::Prefix(reference_frames),
        sampling,
        mel_channels,
        OdeMethod::Euler,
        CfmSampleMode::default(),
    )
}

enum RestoreStrategy<'a> {
    Prefix(usize),
    FrameMask(&'a [bool]),
}

fn integrate_with_strategy<N: DiTVelocity>(
    network: &mut N,
    mut noise: Tensor<3>,
    reference_mel: Tensor<3>,
    text_tokens: Tensor<2, Int>,
    restoration: RestoreStrategy<'_>,
    sampling: SamplingConfig,
    mel_channels: usize,
    method: OdeMethod,
    mode: CfmSampleMode,
) -> Result<Tensor<3>> {
    let result = (|| {
        let timeline = mode.timeline(sampling)?;
        let shape = noise.dims();
        if reference_mel.dims() != shape || shape[2] != mel_channels {
            return Err(F5Error::InvalidShape(
                "CFM tensors must have matching mel channels and batch dimensions",
            ));
        }
        if text_tokens.dims()[0] != shape[0] {
            return Err(F5Error::InvalidShape("Text and mel batch sizes must match"));
        }
        if mode.duplicate_test {
            let prefix = match restoration {
                RestoreStrategy::Prefix(prefix) => prefix,
                RestoreStrategy::FrameMask(_) => shape[1],
            };
            if prefix == 0 || prefix > shape[1] {
                return Err(F5Error::InvalidShape("Invalid duplicate-test reference length"));
            }
            let copied = prefix.min(shape[1] - prefix);
            let mut parts = vec![
                Tensor::<3>::zeros([shape[0], prefix, mel_channels], &noise.device())
            ];
            if copied != 0 {
                parts.push(reference_mel.clone().slice([0..shape[0], 0..copied, 0..mel_channels]));
            }
            if prefix + copied < shape[1] {
                parts.push(Tensor::<3>::zeros(
                    [shape[0], shape[1] - prefix - copied, mel_channels], &noise.device()
                ));
            }
            noise = noise * (1.0 - mode.t_inter as f32) + Tensor::cat(parts, 1) * mode.t_inter as f32;
        }
        let (condition, frame_mask) = match restoration {
            RestoreStrategy::Prefix(reference_frames) => {
                if reference_frames == 0 || reference_frames >= shape[1] {
                    return Err(F5Error::InvalidShape(
                        "CFM reference length must be between zero and total frames",
                    ));
                }
                (reference_mel.clone(), None)
            }
            RestoreStrategy::FrameMask(keep) => {
                if shape[0] != 1 || keep.len() != shape[1] {
                    return Err(F5Error::InvalidShape(
                        "Speech edit frame mask must match a single mel sequence",
                    ));
                }
                let values: Vec<f32> = keep
                    .iter()
                    .flat_map(|&value| {
                        std::iter::repeat_n(if value { 1.0 } else { 0.0 }, mel_channels)
                    })
                    .collect();
                let mask = Tensor::<3>::from_data(
                    TensorData::new(values, [1, shape[1], mel_channels]),
                    &noise.device(),
                );
                (reference_mel.clone() * mask.clone(), Some(mask))
            }
        };
        let condition = if mode.no_ref_audio {
            Tensor::<3>::zeros(shape, &noise.device())
        } else {
            condition
        };
        for times in timeline.windows(2) {
            let delta = times[1] - times[0];
            let velocity = network.predict(
                noise.clone(),
                condition.clone(),
                text_tokens.clone(),
                times[0],
                sampling.cfg_strength,
            )?;
            if velocity.dims() != shape {
                return Err(F5Error::InvalidShape("DiT must output [B, frames, 100]"));
            }
            let velocity = match method {
                OdeMethod::Euler => velocity,
                OdeMethod::Midpoint => {
                    let mid = noise.clone() + velocity * (delta * 0.5);
                    let mid_velocity = network.predict(
                        mid,
                        condition.clone(),
                        text_tokens.clone(),
                        times[0] + delta * 0.5,
                        sampling.cfg_strength,
                    )?;
                    if mid_velocity.dims() != shape {
                        return Err(F5Error::InvalidShape(
                            "DiT midpoint must output [B, frames, 100]",
                        ));
                    }
                    mid_velocity
                }
            };
            noise = noise + velocity * delta;
        }
        if let Some(mask) = frame_mask {
            let unmasked = Tensor::<3>::ones(shape, &noise.device()) - mask;
            Ok(condition + noise * unmasked)
        } else if let RestoreStrategy::Prefix(reference_frames) = restoration {
            let reference_prefix =
                condition.slice([0..shape[0], 0..reference_frames, 0..shape[2]]);
            let generated_suffix =
                noise.slice([0..shape[0], reference_frames..shape[1], 0..shape[2]]);
            Ok(Tensor::cat(vec![reference_prefix, generated_suffix], 1))
        } else {
            unreachable!()
        }
    })();
    network.release_request_cache();
    result
}

#[cfg(feature = "cubecl-cuda")]
pub fn cuda_device(index: usize) -> Device {
    Device::cuda(index)
}

#[cfg(feature = "cubecl-wgpu")]
pub fn wgpu_device() -> Device {
    Device::wgpu(Default::default())
}

#[cfg(feature = "cubecl-cpu")]
pub fn cpu_device() -> Device {
    Device::cpu()
}

#[cfg(all(test, feature = "cubecl-cpu"))]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    struct ConstantVelocity {
        calls: usize,
        released: bool,
    }

    impl DiTVelocity for ConstantVelocity {
        fn predict(
            &mut self,
            noisy: Tensor<3>,
            _: Tensor<3>,
            _: Tensor<2, Int>,
            _: f32,
            _: f32,
        ) -> Result<Tensor<3>> {
            self.calls += 1;
            Ok(Tensor::ones(noisy.dims(), &noisy.device()))
        }

        fn release_request_cache(&mut self) {
            self.released = true;
        }
    }

    #[test]
    fn cpu_speech_edit_euler_and_midpoint_match_independent_pytorch() {
        struct AffineVelocity;
        impl DiTVelocity for AffineVelocity {
            fn predict(
                &mut self,
                x: Tensor<3>,
                condition: Tensor<3>,
                _: Tensor<2, Int>,
                time: f32,
                cfg: f32,
            ) -> Result<Tensor<3>> {
                let conditional = x.clone() * 0.11 + condition.clone() * 0.05 + time * 0.3;
                let unconditional = x * 0.07 - condition * 0.02 - time * 0.1;
                Ok(conditional.clone() + (conditional - unconditional) * cfg)
            }
            fn release_request_cache(&mut self) {}
        }
        let meta: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/speech_edit_ode_pytorch_expected.json"
        ))
        .unwrap();
        let expected =
            include_bytes!("../../tests/fixtures/speech_edit_ode_pytorch_expected.f32le");
        let device = Device::cpu();
        let frames = meta["frames"].as_u64().unwrap() as usize;
        let keep: Vec<bool> = meta["keep"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let original: Vec<f32> = (0..frames * 100).map(|i| i as f32 / 200.0 - 0.1).collect();
        let initial: Vec<f32> = (0..frames * 100).map(|i| i as f32 / 1000.0 - 0.3).collect();
        for (index, case) in meta["cases"].as_array().unwrap().iter().enumerate() {
            let steps = case["steps"].as_u64().unwrap() as usize;
            let sampling = SamplingConfig {
                steps,
                cfg_strength: case["cfg"].as_f64().unwrap() as f32,
                sway_sampling_coef: case["sway"].as_f64().map(|v| v as f32),
                empirically_pruned_steps: case["epss"].as_bool().unwrap(),
            };
            let method = match case["method"].as_str().unwrap() {
                "euler" => OdeMethod::Euler,
                "midpoint" => OdeMethod::Midpoint,
                _ => unreachable!(),
            };
            let output = integrate_with_edit_mask_ode(
                &mut AffineVelocity,
                Tensor::from_data(TensorData::new(initial.clone(), [1, frames, 100]), &device),
                Tensor::from_data(TensorData::new(original.clone(), [1, frames, 100]), &device),
                Tensor::<2, Int>::zeros([1, 4], &device),
                &keep,
                sampling,
                method,
            )
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
            let offset = case["offset"].as_u64().unwrap() as usize;
            assert_eq!(output.len(), case["count"].as_u64().unwrap() as usize);
            let mut worst = 0.0_f32;
            for (j, actual) in output.iter().enumerate() {
                let idx = (offset + j) * 4;
                let oracle = f32::from_le_bytes(expected[idx..idx + 4].try_into().unwrap());
                worst = worst.max((actual - oracle).abs());
            }
            assert!(
                worst < 2e-5,
                "case {index} {method:?} PyTorch parity: max_abs={worst}"
            );
        }
    }

    #[test]
    fn cpu_speech_edit_preserves_disjoint_regions_and_masks_network_condition() {
        let device = Device::cpu();
        let frames = 8;
        let mut network = ConstantVelocity {
            calls: 0,
            released: false,
        };
        let original = vec![9.0; frames * 100];
        let condition = Tensor::from_data(TensorData::new(original, [1, frames, 100]), &device);
        let keep = [true, true, false, false, true, false, true, false];
        let output = integrate_with_edit_mask(
            &mut network,
            Tensor::zeros([1, frames, 100], &device),
            condition,
            Tensor::<2, Int>::zeros([1, 3], &device),
            &keep,
            SamplingConfig {
                steps: 4,
                cfg_strength: 1.0,
                sway_sampling_coef: None,
                empirically_pruned_steps: false,
            },
        )
        .unwrap();
        let values = output.into_data().try_to_vec::<f32>().unwrap();
        for (frame, is_kept) in keep.into_iter().enumerate() {
            let expected = if is_kept { 9.0 } else { 1.0 };
            assert!(
                values[frame * 100..(frame + 1) * 100]
                    .iter()
                    .all(|&v| (v - expected).abs() < 1e-6)
            );
        }
        assert_eq!(network.calls, 4);
        assert!(network.released);
    }

    #[test]
    fn cpu_cfm_keeps_reference_and_integrates_generated_suffix() {
        let device = Device::cpu();
        let mut model = ConstantVelocity {
            calls: 0,
            released: false,
        };
        let noise = Tensor::zeros([1, 6, 100], &device);
        let mut condition = vec![0.0; 600];
        condition[..200].fill(3.5);
        let condition = Tensor::from_data(TensorData::new(condition, [1, 6, 100]), &device);
        let text = Tensor::<2, Int>::zeros([1, 4], &device);
        let sampling = SamplingConfig {
            steps: 5,
            cfg_strength: 2.0,
            sway_sampling_coef: None,
            empirically_pruned_steps: false,
        };
        let output = integrate(&mut model, noise, condition, text, 2, sampling).unwrap();
        let values = output.into_data().try_to_vec::<f32>().unwrap();
        assert!(values[..200].iter().all(|&v| (v - 3.5).abs() < 1e-6));
        assert!(values[200..].iter().all(|&v| (v - 1.0).abs() < 1e-5));
        assert_eq!(model.calls, 5);
        assert!(model.released);
    }
}
